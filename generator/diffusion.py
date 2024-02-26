import logging
import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
import typing
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from pytorch_lightning import LightningModule
from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from diffusers.training_utils import EMAModel
from diffusers import UNet2DModel

from generator.diffusion_utils import ConditionalUnet1D
from dynamics.sim_test_mj import sim_test_batch
from dynamics.sim_test_mj_3d import sim_test_batch_3d
from dynamics.metrics import metric2objective, convergence_mode_three_class, slicer

NoiseScheduler = Union[DDPMScheduler, DDIMScheduler]
NoiseSchedulerOutput = Union[DDPMSchedulerOutput, DDIMSchedulerOutput]
NoisePredictionNet = Union[UNet2DModel, ConditionalUnet1D]

SCALE_2D = 0.001
SCALE_2D_CONV = 10.0
SCALE_3D = 0.5
SCALE_3D_CONV = 0.8

class Diffusion(LightningModule):
    def __init__(
        self,
        noise_pred_net: NoisePredictionNet,
        noise_scheduler: NoiseScheduler,
        num_inference_steps: int,
        num_epochs: int=10000,
        mode: str="point",  # point_3d, point
        input_dim: int=1,
        num_points: int=10,
        H: int=32,
        W: int=32,
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 0,
        ema_power: float = 0.75,
        ema_update_after_step: int = 0,
        num_timesteps_per_batch: int = 1,
        action_groups: Optional[Dict[str, slice]] = None,
        float32_matmul_precision: str = "high",
        class_cond: bool = False,
        classifier_model: Optional[torch.nn.DataParallel] = None,
        grid_size: int = 360,
        num_pos: int = 5,
        object_vertices: Optional[torch.Tensor] = None,
        object_ids: Optional[List[int]] = None,
        num_cpus: int = 32,
        sub_batch_size: int = 1024,
        pts_x_dim: int = 7,
        pts_z_dim: int = 3,
        render_video: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        if os.environ.get("TORCH_COMPILE", "0") == "0":
            logging.info("Not compiling model. To enable, set `TORCH_COMPILE=1`")
            self.ema_nets = nn.ModuleDict(
                {
                    "noise_pred_net": noise_pred_net,
                }
            )
        else:
            logging.info("`torch.compile`-ing the model")
            # cache text features before compiling
            self.ema_nets = nn.ModuleDict(
                {
                    "noise_pred_net": torch.compile(noise_pred_net, mode="max-autotune"),
                }  # type: ignore
            )
        self.ema = EMAModel(
            model=self.ema_nets,
            power=ema_power,
            update_after_step=ema_update_after_step,
        )
        self.mode = mode
        self.input_dim = input_dim
        self.num_points = num_points
        self.pts_x_dim = pts_x_dim
        self.pts_z_dim = pts_z_dim
        self.H = H
        self.W = W
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.num_timesteps_per_batch = num_timesteps_per_batch
        self.num_epochs = num_epochs   
        self.action_groups = action_groups if action_groups is not None else {}
        torch.set_float32_matmul_precision(float32_matmul_precision)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        self.class_cond = class_cond
        self.seed = seed
        if class_cond:
            self.classifier_model = classifier_model
            self.grid_size = grid_size
            self.num_pos = num_pos
            self.object_vertices = object_vertices
            self.object_ids = object_ids    # int for 2d, str for 3d
            self.num_cpus = num_cpus
            self.use_sub_batch = self.mode == 'point_3d'
            self.sub_batch_size = sub_batch_size
            self.render_video = render_video
            self.threshold = torch.Tensor([0.02, 0.001, 0.001]).to(device=self.device) if self.mode == 'point_3d' else torch.Tensor([0.03, 0.002, 0.003]).to(device=self.device)
            self.std = torch.Tensor([0.0312, 0.0016, 0.0026]).to(device=self.device) if self.mode == 'point_3d' else torch.Tensor([0.0565, 0.0026, 0.0047]).to(device=self.device)
            self.threshold_std = self.threshold / self.std

    @property
    def noise_pred_net(self) -> NoisePredictionNet:
        return typing.cast(
            NoisePredictionNet, self.ema_nets.get_submodule("noise_pred_net")
        )

    def get_stats(self, tensor_data) -> Dict[str, Any]:
        batch_size = tensor_data.shape[0]

        # prepare target action sequence
        # normalized_input = self.normalize(tensor_data).to(device=self.device)
        input = tensor_data.repeat(self.num_timesteps_per_batch, 1, 1)  # already normalized to [-1,1]
        # sample noise to add
        noise = torch.randn((batch_size * self.num_timesteps_per_batch, self.num_points, self.input_dim), device=self.device,)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,  # type: ignore
            (batch_size * self.num_timesteps_per_batch,),
            device=self.device,
        ).long()

        # add noise to the clean inputs according to the noise magnitude at each
        # diffusion iteration (this is the forward diffusion process)
        noisy_inputs = self.noise_scheduler.add_noise(
            original_samples=typing.cast(torch.FloatTensor, input),
            noise=typing.cast(torch.FloatTensor, noise),
            timesteps=typing.cast(torch.IntTensor, timesteps),
        )

        # predict the noise residual
        if self.mode == "point" or self.mode == "point_3d":
            noise_pred = self.noise_pred_net(
                noisy_inputs,
                timesteps,
            )
        else:
            noise_pred = self.noise_pred_net(
                noisy_inputs,
                timesteps,
            ).sample

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        return {"loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]}
    
    def training_step(self, tensor_data, batch_idx):
        stats = self.get_stats(tensor_data.to(device=self.device, non_blocking=True))
        total_loss = sum(v for k, v in stats.items() if "loss" in k)
        self.log_dict(
            {f"train/{k}": v for k, v in stats.items()},
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        return total_loss
    
    def validation_step(self, tensor_data, batch_idx):
        batch_size = tensor_data.shape[0]
        sample = tensor_data.clone().detach()    # noise, (B, num_points, input_dim)
        rs = np.random.RandomState(self.seed)
        noise = torch.from_numpy(rs.randn(batch_size, self.num_points, self.input_dim)).float().to(device=self.device)
        timesteps = self.num_inference_steps * torch.ones((batch_size,), dtype=torch.int64, device=self.device)
        sample = self.noise_scheduler.add_noise(
            original_samples=typing.cast(torch.FloatTensor, sample),
            noise=typing.cast(torch.FloatTensor, noise),
            timesteps=typing.cast(torch.IntTensor, timesteps),
        )

        noise_pred_loss = 0.0
        # denoising loop
        for i, t in enumerate(self.noise_scheduler.timesteps):
            timesteps = t * torch.ones((batch_size,), dtype=torch.int64, device=self.device)
            with torch.no_grad():
                if self.mode == "point" or self.mode == "point_3d":
                    noise_pred = self.noise_pred_net(sample, timesteps)
                else:
                    noise_pred = self.noise_pred_net(sample, timesteps).sample
                noise_pred_loss += nn.functional.mse_loss(noise_pred, noise).item()
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            # inspect denoising process (visualize intermediate results)
            if batch_idx == 0:
                os.makedirs(os.path.join(self.logger.save_dir, 'val_vis'), exist_ok=True)
                if self.mode == 'point':
                    plt.clf()
                    f = plt.figure()
                    ax = f.add_subplot(211)
                    ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
                    ax.scatter(np.linspace(-1.0, 1.0, self.num_points // 2), sample[0, :self.num_points//2, 0].cpu().numpy())
                    ax = f.add_subplot(212)
                    ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
                    ax.scatter(np.linspace(-1.0, 1.0, self.num_points // 2), sample[0, self.num_points//2:, 0].cpu().numpy())
                    plt.savefig(os.path.join(self.logger.save_dir, 'val_vis', '%d_%d.png' % (self.current_epoch, i)))
                    plt.close()
                elif self.mode == "point_3d":
                    x = np.linspace(-1.0, 1.0, self.pts_x_dim)
                    z = np.linspace(-1.0, 1.0, self.pts_z_dim)
                    x_n, z_n = np.meshgrid(x, z)
                    x_n = x_n.T.reshape(-1)
                    z_n = z_n.T.reshape(-1)
                    plt.clf()
                    f = plt.figure()
                    ax = f.add_subplot(211, projection='3d')
                    ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0))
                    ax.scatter(x_n, sample[0, :self.num_points//2, 0].cpu().numpy(), z_n, s=2)
                    ax = f.add_subplot(212, projection='3d')
                    ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0))
                    ax.scatter(x_n, sample[0, self.num_points//2:, 0].cpu().numpy(), z_n, s=2)
                    plt.savefig(os.path.join(self.logger.save_dir, 'val_vis', '%d_%d.png' % (self.current_epoch, i)))
                    plt.close()
        noise_pred_loss /= self.num_inference_steps
        loss = nn.functional.mse_loss(sample, tensor_data)
        accuracy = torch.mean(torch.abs(sample - tensor_data) < 0.01, dtype=torch.float)
        self.log_dict(
            {
                "val/noise pred loss": noise_pred_loss,
                "val/denoise loss": loss,
                "val/accuracy": accuracy,
            },
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        # unguided sample
        if (self.current_epoch > 0 or self.on_validation_batch_start) and batch_idx == 0:
            noise_sample = noise.clone().detach()
            imgs = []
            for i, t in enumerate(self.noise_scheduler.timesteps):
                timesteps = t * torch.ones((batch_size,), dtype=torch.int64, device=self.device)
                with torch.no_grad():
                    if self.mode == "point" or self.mode == "point_3d":
                        noise_pred = self.noise_pred_net(noise_sample, timesteps)
                    else:
                        noise_pred = self.noise_pred_net(noise_sample, timesteps).sample
                noise_sample = self.noise_scheduler.step(noise_pred, t, noise_sample).prev_sample
                # inspect denoising process (visualize intermediate results)
                os.makedirs(os.path.join(self.logger.save_dir, 'val_vis_noise'), exist_ok=True)
                if self.mode == 'point':
                    for idx, s in enumerate(noise_sample):
                        plt.clf()
                        f = plt.figure()
                        ax = f.add_subplot(211)
                        ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
                        ax.scatter(np.linspace(-1.0, 1.0, self.num_points // 2), s[:self.num_points//2, 0].cpu().numpy())
                        ax = f.add_subplot(212)
                        ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
                        ax.scatter(np.linspace(-1.0, 1.0, self.num_points // 2), s[self.num_points//2:, 0].cpu().numpy())
                        plt.savefig(os.path.join(self.logger.save_dir, 'val_vis_noise', '%d_%d_%d.png' % (self.current_epoch, batch_idx*batch_size+idx, i)))
                        plt.close()
                        if t == self.noise_scheduler.timesteps[-1]:
                            imgs.append(os.path.join(self.logger.save_dir, 'val_vis_noise', '%d_%d_%d.png' % (self.current_epoch, batch_idx*batch_size+idx, i)))
                elif self.mode == "point_3d":
                    x = np.linspace(-1.0, 1.0, self.pts_x_dim)
                    z = np.linspace(-1.0, 1.0, self.pts_z_dim)
                    x_n, z_n = np.meshgrid(x, z)
                    x_n = x_n.T.reshape(-1)
                    z_n = z_n.T.reshape(-1)
                    for idx, s in enumerate(noise_sample):
                        plt.clf()
                        f = plt.figure()
                        ax = f.add_subplot(111, projection='3d')
                        ax.set(xlim=(-1.0, 1.0), ylim=(-2.0, 2.0), zlim=(-1.0, 1.0))
                        left_pts = s[:self.num_points//2, 0].cpu().numpy() - 1.0
                        right_pts = s[self.num_points//2:, 0].cpu().numpy() + 1.0
                        ax.scatter(x_n, left_pts, z_n, s=2, c='orange')
                        ax.scatter(x_n, right_pts, z_n, s=2, c='green')
                        plt.grid(b=None)
                        plt.savefig(os.path.join(self.logger.save_dir, 'val_vis_noise', '%d_%d_%d.png' % (self.current_epoch, batch_idx*batch_size+idx, i)))
                        plt.close()
                        if t == self.noise_scheduler.timesteps[-1]:
                            imgs.append(os.path.join(self.logger.save_dir, 'val_vis_noise', '%d_%d_%d.png' % (self.current_epoch, batch_idx*batch_size+idx, i)))
            if self.class_cond:
                if self.object_vertices is None:
                    raise ValueError('object vertices not provided')
                # unguided sample
                ori_ranges = [[-1.0, 1.0]]
                if True:
                    num_objects = len(self.object_ids)
                    num_grippers = noise_sample.shape[0]
                    if self.mode == "point_3d":
                        gripper_imgs, metrics, profiles, profiles_x, profiles_y, finals, videos, _ = sim_test_batch_3d(noise_sample.cpu().numpy(), self.object_ids, os.path.join(self.logger.save_dir, 'val_vis_noise'), render=self.render_video, num_cpus=self.num_cpus)
                        imgs_all = gripper_imgs
                    else:
                        _, metrics, profiles, profiles_x, profiles_y, finals, videos, _ = sim_test_batch(noise_sample.cpu().numpy(), self.object_ids, os.path.join(self.logger.save_dir, 'val_vis_noise'), render=self.render_video, num_cpus=self.num_cpus)
                        imgs_all = [imgs[idx] for _ in range(num_objects) for idx in range(len(imgs))]
                for opt_obj in ['convergence', 'shift_up', 'shift_down', 'shift_left', 'shift_right', 'rotate_clockwise', 'rotate_counterclockwise', 'rotate', 'clockwise_up', 'clockwise_left', 'counterclockwise_up', 'counterclockwise_left']:
                    for ori_range in ori_ranges:
                        if True:
                            metrics_unguided = [{k: metric[k][int((ori_range[0]+1)*180):int((ori_range[1]+1)*180)] for k in metric.keys()} for metric in metrics]
                            objectives_unguided = [metric2objective(metric, opt_obj) for metric in metrics_unguided]
                            print('objectives_unguided', len(objectives_unguided))
                            average_objectives = {k: np.mean([objective[k] for objective in objectives_unguided]) for k in objectives_unguided[0].keys()}
                            all_best_ids = self.get_best_ids(objectives_unguided, num_grippers, num_objects, opt_obj=opt_obj)
                            objectives_best = [{k: objectives_unguided[best_ids[k]][k] for k in objectives_unguided[0].keys()} for best_ids in all_best_ids]
                            average_best_objectives = {k: np.mean([objectives_best[i][k] for i in range(len(objectives_best))]) for k in objectives_best[0].keys()}
                            average_obj_objectives = [{k: np.mean([objectives_unguided[i*num_grippers+idx][k] for i in range(num_objects)]) for k in objectives_unguided[0].keys()} for idx in range(num_grippers)]
                            best_average_obj_ids = self.get_average_best_ids(average_obj_objectives, opt_obj=opt_obj)
                            if self.render_video:
                                self.logger.log_table(
                                    key = "val/unguided_sample/%s_orirange=%.3f_%.3f" % (opt_obj, ori_range[0], ori_range[1]),
                                    columns = ["object_idx", "gripper_idx", "gripper", "objective", "profile", "profile_x", "profile_y", "final", "video"],
                                    data = [[-1, -1, wandb.Image(255*np.ones((128, 128, 3))), average_objectives, wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), [wandb.Video(v) for v in videos[0]]]] 
                                    + [[-1, -1, wandb.Image(255*np.ones((128, 128, 3))), average_best_objectives, wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), [wandb.Video(v) for v in videos[0]]]] 
                                    + [[-1, best_average_obj_ids, wandb.Image(imgs[best_average_obj_ids]), average_obj_objectives[best_average_obj_ids], wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), [wandb.Video(v) for v in videos[0]]]]
                                    + [[i // num_grippers, i % num_grippers, wandb.Image(gripper), objective, wandb.Image(profile), wandb.Image(profile_x), wandb.Image(profile_y), wandb.Image(final), [wandb.Video(v) for v in video[int((ori_range[0]+1)*5):int((ori_range[1]+1)*5)]]] for i, (gripper, objective, profile, profile_x, profile_y, final, video) in enumerate(zip(imgs_all, objectives_unguided, profiles, profiles_x, profiles_y, finals, videos))],
                                )
                            else:
                                self.logger.log_table(
                                    key = "val/unguided_sample/%s_orirange=%.3f_%.3f" % (opt_obj, ori_range[0], ori_range[1]),
                                    columns = ["object_idx", "gripper_idx", "gripper", "objective", "profile", "profile_x", "profile_y", "final"],
                                    data = [[-1, -1, wandb.Image(255*np.ones((128, 128, 3))), average_objectives, wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3)))]] 
                                    + [[-1, -1, wandb.Image(255*np.ones((128, 128, 3))), average_best_objectives, wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3)))]] 
                                    + [[-1, best_average_obj_ids, wandb.Image(imgs[best_average_obj_ids]), average_obj_objectives[best_average_obj_ids], wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3)))]]
                                    + [[i // num_grippers, i % num_grippers, wandb.Image(gripper), objective, wandb.Image(profile), wandb.Image(profile_x), wandb.Image(profile_y), wandb.Image(final)] for i, (gripper, objective, profile, profile_x, profile_y, final) in enumerate(zip(imgs_all, objectives_unguided, profiles, profiles_x, profiles_y, finals))],
                                )
                        if opt_obj != 'convergence':
                            self.guided_sample_multi_object(batch_idx, batch_size, noise, self.logger.save_dir, opt_obj=opt_obj, ori_range=ori_range)
                        self.guided_sample(batch_idx, batch_size, noise, self.logger.save_dir, opt_obj=opt_obj, ori_range=ori_range, unguided_sample=noise_sample)

    def clean_grad(self):
        for param in self.classifier_model.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def get_best_ids(self, objectives_unguided, num_grippers, num_objects, opt_obj='rotate'):
        all_best_ids = []
        for idx in range(num_objects):
            best_ids = self.get_best_ids_all_metrics([objectives_unguided[i] for i in range(idx*num_grippers, (idx+1)*num_grippers)], opt_obj=opt_obj)
            best_ids = {k: v+idx*num_grippers for k, v in best_ids.items()}
            all_best_ids.append(best_ids)
        return all_best_ids

    def get_average_best_ids(self, objectives, opt_obj='rotate'):
        if opt_obj == 'rotate' or opt_obj == 'rotate_in_place':
            best_ids = np.argmin([objective['num_zero_classes'] for objective in objectives])
        elif opt_obj == 'rotate_clockwise':
            best_ids = np.argmax([objective['num_clockwise_classes'] for objective in objectives])
        elif opt_obj == 'rotate_counterclockwise':
            best_ids = np.argmax([objective['num_counterclockwise_classes'] for objective in objectives])
        elif opt_obj == 'shift_up':
            best_ids = np.argmax([objective['num_up_classes'] for objective in objectives])
        elif opt_obj == 'shift_down':
            best_ids = np.argmax([objective['num_down_classes'] for objective in objectives])
        elif opt_obj == 'shift_left':
            best_ids = np.argmax([objective['num_left_classes'] for objective in objectives])
        elif opt_obj == 'shift_right':
            best_ids = np.argmax([objective['num_right_classes'] for objective in objectives])
        elif opt_obj == 'convergence':
            best_ids = np.argmax([objective['max_convergence_range_5deg'] for objective in objectives])
        elif opt_obj == 'clockwise_up':
            best_ids = np.argmax([objective['num_clockwise_up_classes'] for objective in objectives])
        elif opt_obj == 'clockwise_down':
            best_ids = np.argmax([objective['num_clockwise_down_classes'] for objective in objectives])
        elif opt_obj == 'clockwise_left':
            best_ids = np.argmax([objective['num_clockwise_left_classes'] for objective in objectives])
        elif opt_obj == 'clockwise_right':
            best_ids = np.argmax([objective['num_clockwise_right_classes'] for objective in objectives])
        elif opt_obj == 'counterclockwise_up':
            best_ids = np.argmax([objective['num_counterclockwise_up_classes'] for objective in objectives])
        elif opt_obj == 'counterclockwise_down':
            best_ids = np.argmax([objective['num_counterclockwise_down_classes'] for objective in objectives])
        elif opt_obj == 'counterclockwise_left':
            best_ids = np.argmax([objective['num_counterclockwise_left_classes'] for objective in objectives])
        elif opt_obj == 'counterclockwise_right':
            best_ids = np.argmax([objective['num_counterclockwise_right_classes'] for objective in objectives])
        else:
            raise ValueError('opt obj not supported')
        return best_ids

    def get_best_ids_all_metrics(self, objectives, opt_obj='rotate'):
        if opt_obj == 'rotate' or opt_obj == 'rotate_in_place':
            best_ids = {'num_zero_classes': np.argmin([objective['num_zero_classes'] for objective in objectives]), 'delta_theta_abs': np.argmax([objective['delta_theta_abs'] for objective in objectives]), 'final_delta_theta_abs': np.argmax([objective['final_delta_theta_abs'] for objective in objectives])}
        elif opt_obj == 'rotate_clockwise':
            best_ids = {'num_clockwise_classes': np.argmax([objective['num_clockwise_classes'] for objective in objectives]), 'delta_theta': np.argmin([objective['delta_theta'] for objective in objectives]),'final_delta_theta': np.argmin([objective['final_delta_theta'] for objective in objectives])}
        elif opt_obj == 'rotate_counterclockwise':
            best_ids = {'num_counterclockwise_classes': np.argmax([objective['num_counterclockwise_classes'] for objective in objectives]), 'delta_theta': np.argmax([objective['delta_theta'] for objective in objectives]), 'final_delta_theta': np.argmax([objective['final_delta_theta'] for objective in objectives])}
        elif opt_obj == 'shift_up':
            best_ids = {'num_up_classes': np.argmax([objective['num_up_classes'] for objective in objectives]), 'delta_pos_x': np.argmin([objective['delta_pos_x'] for objective in objectives]), 'final_pos_x': np.argmin([objective['final_pos_x'] for objective in objectives])}
        elif opt_obj == 'shift_down':
            best_ids = {'num_down_classes': np.argmax([objective['num_down_classes'] for objective in objectives]), 'delta_pos_x': np.argmax([objective['delta_pos_x'] for objective in objectives]), 'final_pos_x': np.argmax([objective['final_pos_x'] for objective in objectives])}
        elif opt_obj == 'shift_left':
            best_ids = {'num_left_classes': np.argmax([objective['num_left_classes'] for objective in objectives]), 'delta_pos_y': np.argmin([objective['delta_pos_y'] for objective in objectives]), 'final_pos_y': np.argmin([objective['final_pos_y'] for objective in objectives])}
        elif opt_obj == 'shift_right':
            best_ids = {'num_right_classes': np.argmax([objective['num_right_classes'] for objective in objectives]), 'delta_pos_y': np.argmax([objective['delta_pos_y'] for objective in objectives]), 'final_pos_y': np.argmax([objective['final_pos_y'] for objective in objectives])}
        elif opt_obj == 'convergence':
            best_ids = {'max_convergence_range_3deg': np.argmax([objective['max_convergence_range_3deg'] for objective in objectives]),'max_convergence_range_5deg': np.argmax([objective['max_convergence_range_5deg'] for objective in objectives]), 'max_convergence_range_10deg': np.argmax([objective['max_convergence_range_10deg'] for objective in objectives])}
        elif opt_obj == 'clockwise_up':
            best_ids = {'num_clockwise_up_classes': np.argmax([objective['num_clockwise_up_classes'] for objective in objectives]), 'num_clockwise_classes': np.argmax([objective['num_clockwise_classes'] for objective in objectives]), 'delta_theta': np.argmin([objective['delta_theta'] for objective in objectives]), 'final_delta_theta': np.argmin([objective['final_delta_theta'] for objective in objectives]), 'num_up_classes': np.argmax([objective['num_up_classes'] for objective in objectives]), 'delta_pos_x': np.argmin([objective['delta_pos_x'] for objective in objectives]), 'final_pos_x': np.argmin([objective['final_pos_x'] for objective in objectives])}
        elif opt_obj == 'clockwise_down':
            best_ids = {'num_clockwise_down_classes': np.argmax([objective['num_clockwise_down_classes'] for objective in objectives]), 'num_clockwise_classes': np.argmax([objective['num_clockwise_classes'] for objective in objectives]), 'delta_theta': np.argmin([objective['delta_theta'] for objective in objectives]), 'final_delta_theta': np.argmin([objective['final_delta_theta'] for objective in objectives]), 'num_down_classes': np.argmax([objective['num_down_classes'] for objective in objectives]), 'delta_pos_x': np.argmax([objective['delta_pos_x'] for objective in objectives]), 'final_pos_x': np.argmax([objective['final_pos_x'] for objective in objectives])}
        elif opt_obj == 'clockwise_left':
            best_ids = {'num_clockwise_left_classes': np.argmax([objective['num_clockwise_left_classes'] for objective in objectives]), 'num_clockwise_classes': np.argmax([objective['num_clockwise_classes'] for objective in objectives]), 'delta_theta': np.argmin([objective['delta_theta'] for objective in objectives]), 'final_delta_theta': np.argmin([objective['final_delta_theta'] for objective in objectives]), 'num_left_classes': np.argmax([objective['num_left_classes'] for objective in objectives]), 'delta_pos_y': np.argmin([objective['delta_pos_y'] for objective in objectives]), 'final_pos_y': np.argmin([objective['final_pos_y'] for objective in objectives])}
        elif opt_obj == 'clockwise_right':
            best_ids = {'num_clockwise_right_classes': np.argmax([objective['num_clockwise_right_classes'] for objective in objectives]), 'num_clockwise_classes': np.argmax([objective['num_clockwise_classes'] for objective in objectives]), 'delta_theta': np.argmin([objective['delta_theta'] for objective in objectives]), 'final_delta_theta': np.argmin([objective['final_delta_theta'] for objective in objectives]), 'num_right_classes': np.argmax([objective['num_right_classes'] for objective in objectives]), 'delta_pos_y': np.argmax([objective['delta_pos_y'] for objective in objectives]), 'final_pos_y': np.argmax([objective['final_pos_y'] for objective in objectives])}
        elif opt_obj == 'counterclockwise_up':
            best_ids = {'num_counterclockwise_up_classes': np.argmax([objective['num_counterclockwise_up_classes'] for objective in objectives]), 'num_counterclockwise_classes': np.argmax([objective['num_counterclockwise_classes'] for objective in objectives]), 'delta_theta': np.argmax([objective['delta_theta'] for objective in objectives]), 'final_delta_theta': np.argmax([objective['final_delta_theta'] for objective in objectives]), 'num_up_classes': np.argmax([objective['num_up_classes'] for objective in objectives]), 'delta_pos_x': np.argmin([objective['delta_pos_x'] for objective in objectives]), 'final_pos_x': np.argmin([objective['final_pos_x'] for objective in objectives])}
        elif opt_obj == 'counterclockwise_down':
            best_ids = {'num_counterclockwise_down_classes': np.argmax([objective['num_counterclockwise_down_classes'] for objective in objectives]), 'num_counterclockwise_classes': np.argmax([objective['num_counterclockwise_classes'] for objective in objectives]), 'delta_theta': np.argmax([objective['delta_theta'] for objective in objectives]), 'final_delta_theta': np.argmax([objective['final_delta_theta'] for objective in objectives]), 'num_down_classes': np.argmax([objective['num_down_classes'] for objective in objectives]), 'delta_pos_x': np.argmax([objective['delta_pos_x'] for objective in objectives]), 'final_pos_x': np.argmax([objective['final_pos_x'] for objective in objectives])}
        elif opt_obj == 'counterclockwise_left':
            best_ids = {'num_counterclockwise_left_classes': np.argmax([objective['num_counterclockwise_left_classes'] for objective in objectives]), 'num_counterclockwise_classes': np.argmax([objective['num_counterclockwise_classes'] for objective in objectives]), 'delta_theta': np.argmax([objective['delta_theta'] for objective in objectives]), 'final_delta_theta': np.argmax([objective['final_delta_theta'] for objective in objectives]), 'num_left_classes': np.argmax([objective['num_left_classes'] for objective in objectives]), 'delta_pos_y': np.argmin([objective['delta_pos_y'] for objective in objectives]), 'final_pos_y': np.argmin([objective['final_pos_y'] for objective in objectives])}
        elif opt_obj == 'counterclockwise_right':
            best_ids = {'num_counterclockwise_right_classes': np.argmax([objective['num_counterclockwise_right_classes'] for objective in objectives]), 'num_counterclockwise_classes': np.argmax([objective['num_counterclockwise_classes'] for objective in objectives]), 'delta_theta': np.argmax([objective['delta_theta'] for objective in objectives]), 'final_delta_theta': np.argmax([objective['final_delta_theta'] for objective in objectives]), 'num_right_classes': np.argmax([objective['num_right_classes'] for objective in objectives]), 'delta_pos_y': np.argmax([objective['delta_pos_y'] for objective in objectives]), 'final_pos_y': np.argmax([objective['final_pos_y'] for objective in objectives])}
        else:
            raise ValueError('opt obj not supported')
        if opt_obj != 'convergence':
            best_ids['success_rate'] = np.argmax([objective['success_rate'] for objective in objectives])
        return best_ids
    
    def deltas_to_objective(self, deltas, opt_obj, centers=None):
        if opt_obj == 'rotate':
            objective = deltas[..., 0]**2
        elif opt_obj == 'rotate_clockwise':
            objective = -deltas[..., 0]
        elif opt_obj == 'rotate_counterclockwise':
            objective = deltas[..., 0]
        elif opt_obj == 'shift_up':
            objective = -deltas[..., 1]
        elif opt_obj == 'shift_down':
            objective = deltas[..., 1]
        elif opt_obj == 'shift_left':
            objective = -deltas[..., 2]
        elif opt_obj == 'shift_right':
            objective = deltas[..., 2]
        elif opt_obj == 'convergence':
            objective = []
            for i, center in enumerate(centers):
                delta_theta = deltas[i*self.grid_size*self.num_pos**2:(i+1)*self.grid_size*self.num_pos**2, 0]
                left_delta = slicer(delta_theta, center*self.num_pos**2-(self.grid_size//2)*self.num_pos**2, center*self.num_pos**2)
                right_delta = slicer(-delta_theta, center*self.num_pos**2, center*self.num_pos**2+(self.grid_size//2)*self.num_pos**2)
                objective.append(torch.cat([left_delta, right_delta], dim=0))
            objective = torch.cat(objective, dim=0)
        elif opt_obj == 'clockwise_up':
            objective = -deltas[..., 0] - deltas[..., 1]
        elif opt_obj == 'clockwise_down':
            objective = -deltas[..., 0] + deltas[..., 1]
        elif opt_obj == 'clockwise_left':
            objective = -deltas[..., 0] - deltas[..., 2]
        elif opt_obj == 'clockwise_right':
            objective = -deltas[..., 0] + deltas[..., 2]
        elif opt_obj == 'counterclockwise_up':
            objective = deltas[..., 0] - deltas[..., 1]
        elif opt_obj == 'counterclockwise_down':
            objective = deltas[..., 0] + deltas[..., 1]
        elif opt_obj == 'counterclockwise_left':
            objective = deltas[..., 0] - deltas[..., 2]
        elif opt_obj == 'counterclockwise_right':
            objective = deltas[..., 0] + deltas[..., 2]
        else:
            raise ValueError('opt obj not supported')
        return objective

    def cond_fn(self, x, t, opt_obj='rotate', object_vertices=None, ori_range=[-1.0, 1.0], convergence_centers=None):
        self.clean_grad()
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            batch_size = x.shape[0]
            sample_ori, sample_x, sample_y = torch.meshgrid(torch.linspace(ori_range[0], ori_range[1], self.grid_size), torch.linspace(-1, 1, self.num_pos), torch.linspace(-1, 1, self.num_pos))
            sample_ori = torch.concat([ori.repeat(batch_size).reshape((-1, 1)) for ori in sample_ori.reshape(-1)], dim=0)
            sample_x = torch.concat([s_x.repeat(batch_size) for s_x in sample_x.reshape(-1)], dim=0)
            sample_y = torch.concat([s_y.repeat(batch_size) for s_y in sample_y.reshape(-1)], dim=0)
            sample_pos = torch.stack([sample_x, sample_y], dim=-1).to(device=self.device)
            if self.mode == 'point':
                pts = torch.concat([x for _ in range(self.grid_size*self.num_pos**2)], dim=0).reshape((batch_size*self.grid_size*self.num_pos**2, -1))
                object_vertices_all = torch.stack([object_vertices.reshape(-1).to(device=self.device) for _ in range(batch_size*self.grid_size*self.num_pos**2)], dim=0)
                timesteps = torch.concat([t for _ in range(self.grid_size*self.num_pos**2)], dim=0)
                logits = self.classifier_model(pts, sample_ori, sample_pos, timesteps.float() / self.noise_scheduler.config.num_train_timesteps, object_vertices=object_vertices_all)
            elif self.mode == 'point_3d':
                pts = torch.concat([torch.linspace(-1.0, 1.0, self.num_points // 2).repeat(batch_size*2, 1).reshape((batch_size, 1, -1)).to(device=self.device), x.moveaxis(-1, -2), torch.linspace(-1.0, 1.0, self.num_points // 2).repeat(batch_size*2, 1).reshape((batch_size, 1, -1)).to(device=self.device)], dim=1)
                pts = torch.concat([pts for _ in range(self.grid_size*self.num_pos**2)], dim=0)
                object_vertices_all = torch.stack([object_vertices.to(device=self.device) for _ in range(batch_size*self.grid_size*self.num_pos**2)], dim=0).moveaxis(-1, -2)
                timesteps = torch.concat([t for _ in range(self.grid_size*self.num_pos**2)], dim=0)
                if self.use_sub_batch:
                    grad = 0.0
                    for i in range(0, batch_size*self.grid_size*self.num_pos**2, self.sub_batch_size):
                        logits = self.classifier_model(pts[i:i+self.sub_batch_size], sample_ori[i:i+self.sub_batch_size], sample_pos[i:i+self.sub_batch_size], timesteps[i:i+self.sub_batch_size].float() / self.noise_scheduler.config.num_train_timesteps, object_vertices=object_vertices_all[i:i+self.sub_batch_size])
                        log_probs = self.deltas_to_objective(logits, opt_obj, centers=convergence_centers)
                        grad += torch.autograd.grad(log_probs.sum(), x)[0]
                    return grad
                logits = self.classifier_model(pts, sample_ori, sample_pos, timesteps.float() / self.noise_scheduler.config.num_train_timesteps, object_vertices=object_vertices_all)
            else:
                raise ValueError('model type not supported')
            log_probs = self.deltas_to_objective(logits, opt_obj, centers=convergence_centers)
            return torch.autograd.grad(log_probs.sum(), x)[0]

    def get_convergence_centers(self, unguided_sample, object_vertices, batch_size, ori_range=[-1.0, 1.0]):
        # make a forward pass to get the profile
        with torch.no_grad():
            sample_ori = torch.linspace(ori_range[0], ori_range[1], self.grid_size)
            sample_ori = torch.concat([ori.repeat(batch_size).reshape((-1, 1)) for ori in sample_ori], dim=0)
            sample_pos = torch.zeros((batch_size*self.grid_size, 2), dtype=torch.float32, device=self.device)
            if self.mode == 'point':
                pts = torch.concat([unguided_sample for _ in range(self.grid_size)], dim=0).reshape((batch_size*self.grid_size, -1))
                object_vertices_all = torch.stack([object_vertices.reshape(-1).to(device=self.device) for _ in range(batch_size*self.grid_size)], dim=0)
                timesteps = torch.zeros((batch_size*self.grid_size,), dtype=torch.float32, device=self.device)
                logits = self.classifier_model(pts, sample_ori, sample_pos, timesteps, object_vertices=object_vertices_all).detach().clone()
            elif self.mode == 'point_3d':
                pts = torch.concat([torch.linspace(-1.0, 1.0, self.num_points // 2).repeat(batch_size*2, 1).reshape((batch_size, 1, -1)).to(device=self.device), unguided_sample.moveaxis(-1, -2), torch.linspace(-1.0, 1.0, self.num_points // 2).repeat(batch_size*2, 1).reshape((batch_size, 1, -1)).to(device=self.device)], dim=1)
                pts = torch.concat([pts for _ in range(self.grid_size)], dim=0)
                object_vertices_all = torch.stack([object_vertices.to(device=self.device) for _ in range(batch_size*self.grid_size)], dim=0).moveaxis(-1, -2)
                timesteps = torch.zeros((batch_size*self.grid_size,), dtype=torch.float32, device=self.device)
                if self.use_sub_batch:
                    logits = []
                    for i in range(0, batch_size*self.grid_size, self.sub_batch_size):
                        logits_sub = self.classifier_model(pts[i:i+self.sub_batch_size], sample_ori[i:i+self.sub_batch_size], sample_pos[i:i+self.sub_batch_size], timesteps[i:i+self.sub_batch_size], object_vertices=object_vertices_all[i:i+self.sub_batch_size])
                        logits.append(logits_sub.detach().clone())
                    logits = torch.cat(logits, dim=0)
                else:
                    logits = self.classifier_model(pts, sample_ori, sample_pos, timesteps, object_vertices=object_vertices_all).detach().clone()
            else:
                raise ValueError('model type not supported')
        profiles_ori = torch.Tensor([2 if l > self.threshold_std[0] else 0 if l < -self.threshold_std[0] else 1 for l in logits[..., 0]]).to(device=self.device)
        max_length_centers = []
        for i in range(batch_size):
            profile = profiles_ori[torch.arange(i, batch_size*self.grid_size, batch_size)]
            lengths, centers = convergence_mode_three_class(profile)
            max_length_centers.append(centers[torch.argmax(lengths)])
        max_length_centers = torch.stack(max_length_centers, dim=0)
        return max_length_centers

    def guided_sample(self, batch_idx, batch_size, noise, save_dir, opt_obj='rotate', ori_range=[-1.0, 1.0], unguided_sample=None):
        all_imgs = []
        all_objectives = []
        all_profiles = []
        all_finals = []
        all_videos = []
        all_gripper_dirs = []
        num_objects = len(self.object_ids)
        if self.mode == 'point':
            if opt_obj == 'convergence':
                classifier_scale = SCALE_2D_CONV
            else:
                classifier_scale = SCALE_2D
        elif self.mode == 'point_3d':
            if opt_obj == 'convergence':
                classifier_scale = SCALE_3D_CONV
            else:
                classifier_scale = SCALE_3D
        else:
            classifier_scale = 0.001
        for idx, obj_vertices in enumerate(self.object_vertices):
            if opt_obj == 'convergence':
                convergence_centers = self.get_convergence_centers(unguided_sample, obj_vertices, batch_size, ori_range=ori_range)
                print('centers', convergence_centers)
            else:
                convergence_centers = None
            object_idx = self.object_ids[idx]
            result_save_dir = os.path.join(save_dir, 'vis_guided', '%s_orirange=%.3f_%.3f' % (opt_obj, ori_range[0], ori_range[1]))
            os.makedirs(result_save_dir, exist_ok=True)
            sample = noise.clone().detach()    # noise, (B, num_points, input_dim) / (B, input_dim, H, W)
            for i, t in enumerate(self.noise_scheduler.timesteps):
                timesteps = t * torch.ones((batch_size,), dtype=torch.int64, device=sample.device)
                noise_pred = self.noise_pred_net(sample, timesteps)
                grad = self.cond_fn(sample, timesteps, opt_obj=opt_obj, object_vertices=obj_vertices, ori_range=ori_range, convergence_centers=convergence_centers)    # (B, num_points, input_dim)
                noise_pred = noise_pred - (1 - self.noise_scheduler.alphas_cumprod[t]).sqrt() * grad * classifier_scale
                sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            if self.mode == "point_3d":
                gripper_imgs, metrics, profiles, profiles_x, profiles_y, finals, videos, save_gripper_dirs = sim_test_batch_3d(sample.cpu().numpy(), [object_idx], os.path.join(result_save_dir, str(object_idx)), render=self.render_video, num_cpus=self.num_cpus, num_rot=int((ori_range[1]-ori_range[0])*180), ori_range=ori_range, render_last=(not self.render_video))
            else:
                gripper_imgs, metrics, profiles, profiles_x, profiles_y, finals, videos, save_gripper_dirs = sim_test_batch(sample.cpu().numpy(), [object_idx], os.path.join(result_save_dir, str(object_idx)), render=self.render_video, num_cpus=self.num_cpus, num_rot=int((ori_range[1]-ori_range[0])*180), ori_range=ori_range, render_last=(not self.render_video))
            if len(metrics) == 0:
                continue
            objectives = [metric2objective(metric, opt_obj) for metric in metrics]
            if opt_obj == 'rotate' or opt_obj == 'rotate_clockwise' or opt_obj == 'rotate_counterclockwise' or opt_obj == 'convergence' or opt_obj == 'clockwise_up' or opt_obj == 'clockwise_down' or opt_obj == 'clockwise_left' or opt_obj == 'clockwise_right' or opt_obj == 'counterclockwise_up' or opt_obj == 'counterclockwise_down' or opt_obj == 'counterclockwise_left' or opt_obj == 'counterclockwise_right':
                obj_profiles = profiles
            elif opt_obj == 'shift_up' or opt_obj == 'shift_down':
                obj_profiles = profiles_x
            elif opt_obj == 'shift_left' or opt_obj == 'shift_right':
                obj_profiles = profiles_y
            else:
                raise ValueError('opt obj not supported')
            best_ids_all_metrics = self.get_best_ids_all_metrics(objectives, opt_obj=opt_obj)
            best_objectives = {k: objectives[best_ids_all_metrics[k]] for k in best_ids_all_metrics.keys()}
            best_imgs = {k: gripper_imgs[best_ids_all_metrics[k]] for k in best_ids_all_metrics.keys()}
            best_profiles = {k: obj_profiles[best_ids_all_metrics[k]] for k in best_ids_all_metrics.keys()}
            best_finals = {k: finals[best_ids_all_metrics[k]] for k in best_ids_all_metrics.keys()}
            best_videos = {k: videos[best_ids_all_metrics[k]] for k in best_ids_all_metrics.keys()}
            best_gripper_dirs = {k: save_gripper_dirs[best_ids_all_metrics[k]] for k in best_ids_all_metrics.keys()}
            all_videos.append(best_videos)
            all_imgs.append(best_imgs)
            all_objectives.append(best_objectives)
            all_profiles.append(best_profiles)
            all_finals.append(best_finals)
            all_gripper_dirs.append(best_gripper_dirs)
        average_best_objectives = {k: np.mean([objective[k][k] for objective in all_objectives]) for k in all_objectives[0].keys()}
        if self.render_video:
            self.logger.log_table(
                key = "val/guided_sample/%s_orirange=%.3f_%.3f" % (opt_obj, ori_range[0], ori_range[1]),
                columns = ["object_idx", "gripper", "objective", "profile", "final", "video", "gripper_dir"],
                data = [[-1, wandb.Image(255*np.ones((128, 128, 3))),  average_best_objectives, wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), [wandb.Video(v) for v in all_videos[0][all_objectives[0].keys()[0]]], ""]] 
                + [[i, wandb.Image(all_imgs[i][k]), all_objectives[i][k], wandb.Image(all_profiles[i][k]), wandb.Image(all_finals[i][k]), [wandb.Video(v) for v in all_videos[i][k]], all_gripper_dirs[i][k]] for i in range(num_objects) for k in all_objectives[i].keys()],
            )
        else:
            self.logger.log_table(
                key = "val/guided_sample/%s_orirange=%.3f_%.3f" % (opt_obj, ori_range[0], ori_range[1]),
                columns = ["object_idx", "gripper", "objective", "profile", "final", "last_img", "gripper_dir"],
                data = [[-1, wandb.Image(255*np.ones((128, 128, 3))),  average_best_objectives, wandb.Image(255*np.ones((128, 128, 3))), wandb.Image(255*np.ones((128, 128, 3))), [wandb.Image(255*np.ones((128, 128, 3)))], ""]]
                + [[i, wandb.Image(all_imgs[i][k]), all_objectives[i][k], wandb.Image(all_profiles[i][k]), wandb.Image(all_finals[i][k]), [wandb.Image(img) for img in all_videos[i][k]], all_gripper_dirs[i][k]] for i in range(num_objects) for k in all_objectives[i].keys()],
            ) 

    def guided_sample_multi_object(self, batch_idx, batch_size, noise, save_dir, opt_obj='rotate', ori_range=[-1.0, 1.0]):
        result_save_dir = os.path.join(save_dir, 'vis_guided', '%s_orirange=%.3f_%.3f' % (opt_obj, ori_range[0], ori_range[1]))
        os.makedirs(result_save_dir, exist_ok=True)
        sample = noise.clone().detach()
        num_objects = len(self.object_ids)
        all_imgs = []
        all_samples = []
        all_objectives = []
        all_gripper_dirs = []
        all_videos = []
        if self.mode == 'point':
            classifier_scale = SCALE_2D
        elif self.mode == 'point_3d':
            classifier_scale = SCALE_3D
        else:
            classifier_scale = 0.001
        for i, t in enumerate(self.noise_scheduler.timesteps):
            timesteps = t * torch.ones((batch_size,), dtype=torch.int64, device=sample.device)
            noise_pred = self.noise_pred_net(sample, timesteps)
            grad = 0.0
            for idx, obj_vertices in enumerate(self.object_vertices):
                g = self.cond_fn(sample, timesteps, opt_obj=opt_obj, object_vertices=obj_vertices, ori_range=ori_range)    # (B, num_points, input_dim)
                grad += g
            grad /= len(self.object_vertices)
            noise_pred = noise_pred - (1 - self.noise_scheduler.alphas_cumprod[t]).sqrt() * grad * classifier_scale
            # eps = eps - (1 - alpha_bar).sqrt() * cond_fn()
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
            for gripper_idx, s in enumerate(sample):
                if self.mode == 'point_3d':
                    x = np.linspace(-1.0, 1.0, self.pts_x_dim)
                    z = np.linspace(-1.0, 1.0, self.pts_z_dim)
                    x_n, z_n = np.meshgrid(x, z)
                    x_n = x_n.T.reshape(-1)
                    z_n = z_n.T.reshape(-1)
                    plt.clf()
                    f = plt.figure()
                    ax = f.add_subplot(111, projection='3d')
                    ax.set(xlim=(-1.0, 1.0), ylim=(-2.0, 2.0), zlim=(-1.0, 1.0))
                    left_pts = s[:self.num_points//2, 0].cpu().numpy() - 1.0
                    right_pts = s[self.num_points//2:, 0].cpu().numpy() + 1.0
                    ax.scatter(x_n, left_pts, z_n, s=2, c='orange')
                    ax.scatter(x_n, right_pts, z_n, s=2, c='green')
                    plt.grid(b=None)
                else:
                    plt.clf()
                    f = plt.figure()
                    ax = f.add_subplot(211)
                    ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
                    ax.scatter(np.linspace(-1.0, 1.0, self.num_points // 2), s[:self.num_points//2, 0].cpu().numpy())
                    ax = f.add_subplot(212)
                    ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
                    ax.scatter(np.linspace(-1.0, 1.0, self.num_points // 2), s[self.num_points//2:, 0].cpu().numpy())
                plt.savefig(os.path.join(result_save_dir, 'allobj_%d_%d.png' % (batch_idx*batch_size+gripper_idx, i)))
                plt.close()
        all_samples = sample.cpu().numpy()
        print('all_samples:', all_samples.shape)
            
        for idx, s in enumerate(all_samples):
            s = np.expand_dims(s, axis=0)
            if self.mode == "point_3d":
                gripper_imgs, metrics, _, _, _, _, videos, save_gripper_dirs = sim_test_batch_3d(s, self.object_ids, os.path.join(result_save_dir, 'allobj_%d' % idx), render=self.render_video, num_cpus=self.num_cpus, num_rot=int((ori_range[1]-ori_range[0])*180), ori_range=ori_range, render_last=(not self.render_video))
            else:
                gripper_imgs, metrics, _, _, _, _, videos, save_gripper_dirs = sim_test_batch(s, self.object_ids, os.path.join(result_save_dir, 'allobj_%d' % idx), render=self.render_video, num_cpus=self.num_cpus, num_rot=int((ori_range[1]-ori_range[0])*180), ori_range=ori_range, render_last=(not self.render_video))
            if len(metrics) != num_objects:
                continue
            objectives = [metric2objective(metric, opt_obj) for metric in metrics]
            average_objectives = {k: np.mean([objective[k] for objective in objectives]) for k in objectives[0].keys()}
            all_objectives.append(average_objectives)
            all_gripper_dirs.append(save_gripper_dirs[0])
            all_imgs.append(gripper_imgs[0])
            all_videos.append(sum(videos, []))
        best_ids_all_metrics = self.get_best_ids_all_metrics(all_objectives, opt_obj=opt_obj)
        objective_keys = best_ids_all_metrics.keys()
        best_objectives = {k: all_objectives[best_ids_all_metrics[k]] for k in objective_keys}
        best_imgs = {k: all_imgs[best_ids_all_metrics[k]] for k in objective_keys}
        best_gripper_dirs = {k: all_gripper_dirs[best_ids_all_metrics[k]] for k in objective_keys}
        best_videos = {k: all_videos[best_ids_all_metrics[k]] for k in objective_keys}
        if self.render_video:
            self.logger.log_table(
                key = "val/guided_sample/allobj_%s_orirange=%.3f_%.3f" % (opt_obj, ori_range[0], ori_range[1]),
                columns = ["gripper", "objective", "video", "gripper_dir"],
                data = [[wandb.Image(best_imgs[k]), best_objectives[k], [wandb.Video(v) for v in best_videos[k]], best_gripper_dirs[k]] for k in objective_keys],
            )
        else:
            self.logger.log_table(
                key = "val/guided_sample/allobj_%s_orirange=%.3f_%.3f" % (opt_obj, ori_range[0], ori_range[1]),
                columns = ["gripper", "objective",  "last_img", "gripper_dir"],
                data = [[wandb.Image(best_imgs[k]), best_objectives[k], [wandb.Image(img) for img in best_videos[k]], best_gripper_dirs[k]] for k in objective_keys],
            )

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.ema_nets.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=0.0)
        return [self.optimizer], [self.lr_scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        try:
            self.ema.step(self.ema_nets)
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                self.ema.averaged_model = self.ema.averaged_model.to(self.device)
            else:
                raise e
        self.log(
            "train/ema_decay",
            self.ema.decay,
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if os.environ.get("TORCH_COMPILE", "0") == "0":
            # need to handle torch compile, for instance:
            # noise_pred_net._orig_mod.final_conv.1.bias
            # noise_pred_net.final_conv.1.bias
            checkpoint["state_dict"] = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
            checkpoint["state_dict"]["ema_model"] = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["state_dict"]["ema_model"].items()
            }
        return super().on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        retval = super().load_state_dict(state_dict, strict=False)
        self.ema.averaged_model.load_state_dict(state_dict["ema_model"], strict=False)
        return retval

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        print('on save checkpoint')
        checkpoint["state_dict"]["ema_model"] = self.ema.averaged_model.state_dict()
        super().on_save_checkpoint(checkpoint)
