import os
import typing
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

import torch
import torch.nn as nn 

from dynamics.profile_forward_3d import ProfileForward3DModel
from dynamics.profile_forward_2d import ProfileForward2DModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.use_sub_batch = args.use_sub_batch
        self.sub_batch_size = args.sub_bs
        self.grid_size = args.grid_size
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.ckpt_path = args.checkpoint_path
        self.fingers_3d = args.fingers_3d
        if self.fingers_3d:
            self.gripperpts_dim = args.ctrlpts_dim
            self.object_vertices_dim = args.object_max_num_vertices
        else:
            self.gripperpts_dim = args.ctrlpts_dim
            self.object_vertices_dim = 2*args.object_max_num_vertices
        self.loss_fn = nn.MSELoss()
        self.num_timesteps_per_batch = args.num_timesteps_per_batch
        self.num_inference_steps = args.num_inference_steps
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps,beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')  # squared cosine beta schedule
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
    def create_model(self):
        if self.fingers_3d:
            self.model = nn.DataParallel(ProfileForward3DModel(output_ch=3, params_ch=self.gripperpts_dim).cuda())
        else:
            self.model = nn.DataParallel(ProfileForward2DModel(output_ch=3, params_ch=self.gripperpts_dim, object_ch=self.object_vertices_dim).cuda())
        for param in self.model.parameters():
            param.requires_grad = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-2*self.learning_rate)
        if self.ckpt_path is not None:
            print('loading checkpoint from', self.ckpt_path)
            self.model.load_state_dict(torch.load(self.ckpt_path))
        print('done')

    def step(self, ctrl, score, input_ori=None, input_pos=None, object_vertices=None):
        self.model.train()
        if self.fingers_3d:
            input_ctrl_all = ctrl.repeat(self.num_timesteps_per_batch, 1, 1)  # already normalized to [-1,1]
            object_vertices_all = object_vertices.repeat(self.num_timesteps_per_batch, 1, 1)
        else:
            input_ctrl_all = ctrl.repeat(self.num_timesteps_per_batch, 1)  # already normalized to [-1,1]
            object_vertices_all = object_vertices.repeat(self.num_timesteps_per_batch, 1)
        input_ori_all = input_ori.repeat(self.num_timesteps_per_batch, 1)
        input_pos_all = input_pos.repeat(self.num_timesteps_per_batch, 1)
        score_all = score.repeat(self.num_timesteps_per_batch, 1)

        # sample noise to add
        if self.fingers_3d:
            noise = torch.cat([torch.zeros((input_ctrl_all.shape[0]*self.num_timesteps_per_batch, 1, self.gripperpts_dim)), torch.randn((input_ctrl_all.shape[0]*self.num_timesteps_per_batch, 1, self.gripperpts_dim)), torch.zeros((input_ctrl_all.shape[0]*self.num_timesteps_per_batch, 1, self.gripperpts_dim))], dim=1).cuda()
        else:
            noise = torch.randn((input_ctrl_all.shape[0]*self.num_timesteps_per_batch, self.gripperpts_dim),).cuda()
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,  # type: ignore
            (input_ctrl_all.shape[0],),
        ).long().cuda()
        noisy_ctrl_all = self.noise_scheduler.add_noise(
            original_samples=typing.cast(torch.FloatTensor, input_ctrl_all),
            noise=typing.cast(torch.FloatTensor, noise),
            timesteps=typing.cast(torch.IntTensor, timesteps),
        )
        timesteps = timesteps.float() / self.noise_scheduler.config.num_train_timesteps # rescale to [0,1]
        if self.use_sub_batch:
            all_loss = 0.0
            all_pred = []
            for i in range(0, noisy_ctrl_all.shape[0], self.sub_batch_size):
                pred = self.model(noisy_ctrl_all[i:i+self.sub_batch_size], input_ori_all[i:i+self.sub_batch_size], input_pos_all[i:i+self.sub_batch_size], timesteps[i:i+self.sub_batch_size], 
                object_vertices=object_vertices_all[i:i+self.sub_batch_size])
                loss = self.loss_fn(pred, score_all[i:i+self.sub_batch_size])
                all_loss += loss.item()
                all_pred.append(pred.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            all_loss /= (noisy_ctrl_all.shape[0] / self.sub_batch_size)
            all_pred = torch.cat(all_pred, dim=0)
            return all_loss, all_pred
        else:
            pred = self.model(noisy_ctrl_all, input_ori_all, input_pos_all, timesteps=timesteps, object_vertices=object_vertices_all)
            loss = self.loss_fn(pred, score)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred.detach()

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

    def inference(self, ctrl, score, input_ori=None, input_pos=None, object_vertices=None):
        self.model.eval()
        with torch.no_grad():
            if self.fingers_3d:
                input_ctrl_all = ctrl.repeat(self.num_timesteps_per_batch, 1, 1)
                object_vertices_all = object_vertices.repeat(self.num_timesteps_per_batch, 1, 1)
            else:
                input_ctrl_all = ctrl.repeat(self.num_timesteps_per_batch, 1)  # already normalized to [-1,1]
                object_vertices_all = object_vertices.repeat(self.num_timesteps_per_batch, 1)
            input_ori_all = input_ori.repeat(self.num_timesteps_per_batch, 1)
            input_pos_all = input_pos.repeat(self.num_timesteps_per_batch, 1)
            score_all = score.repeat(self.num_timesteps_per_batch, 1)

            # sample noise to add
            if self.fingers_3d:
                noise = torch.cat([torch.zeros((input_ctrl_all.shape[0]*self.num_timesteps_per_batch, 1, self.gripperpts_dim)), torch.randn((input_ctrl_all.shape[0]*self.num_timesteps_per_batch, 1, self.gripperpts_dim)), torch.zeros((input_ctrl_all.shape[0]*self.num_timesteps_per_batch, 1, self.gripperpts_dim))], dim=1).cuda()
            else:
                noise = torch.randn((input_ctrl_all.shape[0]*self.num_timesteps_per_batch, self.gripperpts_dim),).cuda()
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,  # type: ignore
                (input_ctrl_all.shape[0],),
            ).long().cuda()
            noisy_ctrl_all = self.noise_scheduler.add_noise(
                original_samples=typing.cast(torch.FloatTensor, input_ctrl_all),
                noise=typing.cast(torch.FloatTensor, noise),
                timesteps=typing.cast(torch.IntTensor, timesteps),
            )
            timesteps = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
            if self.use_sub_batch:
                all_loss = 0.0
                all_pred = []
                for i in range(0, noisy_ctrl_all.shape[0], self.sub_batch_size):
                    pred = self.model(noisy_ctrl_all[i:i+self.sub_batch_size], input_ori_all[i:i+self.sub_batch_size], input_pos_all[i:i+self.sub_batch_size], timesteps[i:i+self.sub_batch_size], object_vertices=object_vertices_all[i:i+self.sub_batch_size])
                    loss = self.loss_fn(pred, score_all[i:i+self.sub_batch_size])
                    all_loss += loss.item()
                    all_pred.append(pred.detach())
                all_loss /= (noisy_ctrl_all.shape[0] / self.sub_batch_size)
                all_pred = torch.cat(all_pred, dim=0)
                return all_pred, all_loss
            else:
                pred = self.model(noisy_ctrl_all, input_ori_all, input_pos_all, timesteps, object_vertices=object_vertices_all)
                loss = self.loss_fn(pred, score)
        return pred, loss.item()