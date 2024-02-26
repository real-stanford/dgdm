import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dynamics.dataloader import DynamicsDataset
from dynamics.trainer import Trainer
from dynamics.parser import parse

def validate(args, val_loader, trainer, threshold_std=[0.641, 0.625, 0.3846]):
    print('validation:')
    average_val_loss = 0
    average_val_accuracy = 0
    average_val_accuracy_x = 0
    average_val_accuracy_y = 0
    with torch.no_grad():
        for batch in val_loader:
            score = batch['scores']     # [batch_size, num_ori*num_pos, 3]
            input_ori = batch['input_ori'].reshape((-1, 1)).cuda()
            input_pos = batch['input_pos'].reshape((-1, 2)).cuda()
            object_vertices = None
            if args.fingers_3d:
                ctrlpts = torch.cat([batch['ctrlpts'] for _ in range(score.size(1))], 0).moveaxis(-1, -2).cuda()
                object_vertices = torch.cat([batch['object_vertices'] for _ in range(score.size(1))], 0).moveaxis(-1, -2).cuda()
            else:
                ctrlpts = torch.cat([batch['ctrlpts'][..., 1] for _ in range(score.size(1))], 1).reshape((input_ori.shape[0], -1)).cuda()
                object_vertices = torch.cat([batch['object_vertices'] for _ in range(score.size(1))], 1).reshape((input_ori.shape[0], -1)).cuda()
            score = score.reshape((-1, 3)).cuda()
            pred, loss = trainer.inference(None, ctrlpts, score, input_ori, input_pos, object_vertices)
            accuracy = torch.mean(torch.Tensor([2 if score_ori > threshold_std[0] else 0 if score_ori < -threshold_std[0] else 1 for score_ori in score[..., 0]]) == torch.Tensor([2 if pred_ori > threshold_std[0] else 0 if pred_ori < -threshold_std[0] else 1 for pred_ori in pred[..., 0]]), dtype=torch.float32)
            accuracy_x = torch.mean(torch.Tensor([2 if score_x > threshold_std[1] else 0 if score_x < -threshold_std[1] else 1 for score_x in score[..., 1]]) == torch.Tensor([2 if pred_x > threshold_std[1] else 0 if pred_x < -threshold_std[1] else 1 for pred_x in pred[..., 1]]), dtype=torch.float32)
            accuracy_y = torch.mean(torch.Tensor([2 if score_y > threshold_std[2] else 0 if score_y < -threshold_std[2] else 1 for score_y in score[..., 2]]) == torch.Tensor([2 if pred_y > threshold_std[2] else 0 if pred_y < -threshold_std[2] else 1 for pred_y in pred[..., 2]]), dtype=torch.float32)
            average_val_accuracy_x += accuracy_x
            average_val_accuracy_y += accuracy_y
            average_val_loss += loss
            average_val_accuracy += accuracy
    average_val_loss /= len(val_loader)
    print('average val loss:', average_val_loss)
    average_val_accuracy /= len(val_loader)
    print('average val accuracy:', average_val_accuracy)
    average_val_accuracy_x /= len(val_loader)
    print('average val accuracy x:', average_val_accuracy_x)
    average_val_accuracy_y /= len(val_loader)
    print('average val accuracy y:', average_val_accuracy_y)
    return average_val_loss, average_val_accuracy, average_val_accuracy_x, average_val_accuracy_y

def train(args):
    wandb.init(
        project='dynamics model',
        config=args,
        dir=args.save_dir,
        name=args.wandb_id,
    )
    gripper_pts_max_x = 0.12
    gripper_pts_min_x = -0.12
    if args.fingers_3d:
        gripper_pts_max_y = 0
        gripper_pts_min_y = -0.1
        object_pts_max_x = 0.1
        object_pts_min_x = -0.1
        object_pts_max_y = 0.1
        object_pts_min_y = -0.1
    else:
        gripper_pts_max_y = 0.015
        gripper_pts_min_y = -0.045
        object_pts_max_x = 0.05
        object_pts_min_x = -0.05
        object_pts_max_y = 0.05
        object_pts_min_y = -0.05
    gripper_pts_max_z = 0.12
    gripper_pts_min_z = 0.0
    object_pts_max_z = 0.12
    object_pts_min_z = 0.0
    train_dataset = DynamicsDataset(
        dataset_dir=args.data_dir, 
        object_mesh_dir=args.object_mesh_dir,
        fingers_3d=args.fingers_3d, 
        gripper_pts_max_x=gripper_pts_max_x, 
        gripper_pts_min_x=gripper_pts_min_x, 
        gripper_pts_max_y=gripper_pts_max_y, 
        gripper_pts_min_y=gripper_pts_min_y, 
        gripper_pts_max_z=gripper_pts_max_z, 
        gripper_pts_min_z=gripper_pts_min_z, 
        object_max_num_vertices=args.object_max_num_vertices, 
        object_pts_max_x=object_pts_max_x, 
        object_pts_min_x=object_pts_min_x, 
        object_pts_max_y=object_pts_max_y, 
        object_pts_min_y=object_pts_min_y, 
        object_pts_max_z=object_pts_max_z, 
        object_pts_min_z=object_pts_min_z)
    threshold_std = train_dataset.threshold / train_dataset.std
    val_dataset = DynamicsDataset(
        dataset_dir=args.test_data_dir, 
        object_mesh_dir=args.object_mesh_dir,
        fingers_3d=args.fingers_3d,
        gripper_pts_max_x=gripper_pts_max_x, 
        gripper_pts_min_x=gripper_pts_min_x, 
        gripper_pts_max_y=gripper_pts_max_y, 
        gripper_pts_min_y=gripper_pts_min_y, 
        gripper_pts_max_z=gripper_pts_max_z, 
        gripper_pts_min_z=gripper_pts_min_z, 
        object_max_num_vertices=args.object_max_num_vertices, 
        object_pts_max_x=object_pts_max_x, 
        object_pts_min_x=object_pts_min_x, 
        object_pts_max_y=object_pts_max_y, 
        object_pts_min_y=object_pts_min_y, 
        object_pts_max_z=object_pts_max_z, 
        object_pts_min_z=object_pts_min_z)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    trainer = Trainer(args)
    trainer.create_model()

    # short-cut for testing
    if args.mode == 'validate':
        if args.checkpoint_path is None:
            raise ValueError('checkpoint path is not specified')
        validate(args, val_loader, trainer, threshold_std=threshold_std)
        return

    # train model
    if args.mode == 'train':
        best_val_loss = float('inf')
        last_best_epoch = 0
        for epoch in tqdm(range(args.num_epochs)):
            average_loss = 0
            average_accuracy = 0
            average_accuracy_x = 0
            average_accuracy_y = 0
            for idx_batch, batch in enumerate(tqdm(train_loader)):
                score = batch['scores']     # [batch_size, num_ori*num_pos, 3]
                input_ori = batch['input_ori'].reshape((-1, 1)).cuda()
                input_pos = batch['input_pos'].reshape((-1, 2)).cuda()
                object_vertices = None
                if args.fingers_3d:
                    ctrlpts = torch.cat([batch['ctrlpts'] for _ in range(score.size(1))], 0).moveaxis(-1, -2).cuda()
                    object_vertices = torch.cat([batch['object_vertices'] for _ in range(score.size(1))], 0).moveaxis(-1, -2).cuda()
                else:
                    ctrlpts = torch.cat([batch['ctrlpts'][..., 1] for _ in range(score.size(1))], 1).reshape((input_ori.shape[0], -1)).cuda()
                    object_vertices = torch.cat([batch['object_vertices'] for _ in range(score.size(1))], 1).reshape((input_ori.shape[0], -1)).cuda()
                score = score.reshape((-1, 3)).cuda()
                loss, pred = trainer.step(ctrlpts, score, input_ori, input_pos, object_vertices)

                accuracy = torch.mean(torch.Tensor([2 if score_ori > threshold_std[0] else 0 if score_ori < -threshold_std[0] else 1 for score_ori in score[..., 0]]) == torch.Tensor([2 if pred_ori > threshold_std[0] else 0 if pred_ori < -threshold_std[0] else 1 for pred_ori in pred[..., 0]]), dtype=torch.float32)
                accuracy_x = torch.mean(torch.Tensor([2 if score_x > threshold_std[1] else 0 if score_x < -threshold_std[1] else 1 for score_x in score[..., 1]]) == torch.Tensor([2 if pred_x > threshold_std[1] else 0 if pred_x < -threshold_std[1] else 1 for pred_x in pred[..., 1]]), dtype=torch.float32)
                accuracy_y = torch.mean(torch.Tensor([2 if score_y > threshold_std[2] else 0 if score_y < -threshold_std[2] else 1 for score_y in score[..., 2]]) == torch.Tensor([2 if pred_y > threshold_std[2] else 0 if pred_y < -threshold_std[2] else 1 for pred_y in pred[..., 2]]), dtype=torch.float32)
                average_accuracy_x += accuracy_x
                average_accuracy_y += accuracy_y
                average_loss += loss
                average_accuracy += accuracy
                wandb.log({
                    'train/lr': trainer.optimizer.param_groups[0]['lr'],
                    'train/batch loss': loss,
                    'train/batch accuracy ori': accuracy,
                    'train/batch accuracy x': accuracy_x,
                    'train/batch accuracy y': accuracy_y,
                })
                if idx_batch % args.save_ckpt_step == 0:
                    os.makedirs(args.save_dir, exist_ok=True)
                    trainer.save_checkpoint(os.path.join(args.save_dir, '%d_%d.pt' % (epoch, idx_batch)))
            trainer.lr_scheduler.step()
            average_loss /= len(train_loader)
            print('epoch:', epoch, 'loss:', average_loss)
            average_accuracy /= len(train_loader)
            print('epoch:', epoch, 'accuracy:', average_accuracy)
            average_accuracy_x /= len(train_loader)
            print('epoch:', epoch, 'accuracy x:', average_accuracy_x)
            average_accuracy_y /= len(train_loader)
            print('epoch:', epoch, 'accuracy y:', average_accuracy_y)
            wandb.log({
                'train/average loss': average_loss,
                'train/average accuracy ori': average_accuracy,
                'train/average accuracy x': average_accuracy_x,
                'train/average accuracy y': average_accuracy_y,
            })
            if epoch % args.val_step == 0:
                val_loss, val_accuracy, val_accuracy_x, val_accuracy_y = validate(args, val_loader, trainer, threshold_std=threshold_std)
                wandb.log({
                    'val/average loss': val_loss,
                    'val/average accuracy ori': val_accuracy,
                    'val/average accuracy x': val_accuracy_x,
                    'val/average accuracy y': val_accuracy_y,
                })
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(args.save_dir, exist_ok=True)
                    trainer.save_checkpoint(os.path.join(args.save_dir, 'best.pt'))
                    last_best_epoch = epoch
                else:
                    if epoch - last_best_epoch >= args.patience:
                        print('early stopping...')
                        break
    wandb.finish()

if __name__ == '__main__':
    args = parse()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
    


