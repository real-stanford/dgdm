import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
import torch
import torch.nn as nn

from dynamics.profile_forward_2d import get_embedder, timestep_embedding
from dynamics.models.pointnet2 import PointNet2

class ProfileForward3DModel(nn.Module):
    def __init__(self, W=256, params_ch=1250, ori_ch=1, pos_ch=2, output_ch=3):
        super(ProfileForward3DModel, self).__init__()
        self.W = W
        self.ori_ch = ori_ch
        self.pos_ch = pos_ch
        self.output_ch = output_ch
        self.ori_embed, ori_embed_dim = get_embedder(ori_ch, 4, 0, scalar_factor=1)
        self.pos_embed, pos_embed_dim = get_embedder(pos_ch, 4, 0, scalar_factor=1)
        self.ori_ch = ori_embed_dim
        self.pos_ch = pos_embed_dim
        self.pose_embed_dim = ori_embed_dim + pos_embed_dim
        self.time_embed_dim = W
        self.time_encoder = nn.Sequential(
            nn.Linear(W // 2, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.object_encoder = PointNet2(W)
        self.object_encode_dim = W
        self.gripper_encoder = nn.Sequential(
            nn.Linear(params_ch, W),
            nn.ReLU(),
            nn.Linear(W, W),
        )
        self.gripper_encode_dim = W
        self.linears = nn.Sequential(
            nn.Linear(self.gripper_encode_dim + self.pose_embed_dim + self.time_embed_dim + self.object_encode_dim, W * 2),
            nn.BatchNorm1d(W * 2),
            nn.ReLU(),
            nn.Linear(W * 2, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
        )
        self.output = nn.Linear(W, output_ch)
        
    def forward(self, x_ctrl, x_ori, x_pos, timesteps=None, object_vertices=None):
        '''
        input: 
            ctrlpts [batch_size, 3, 1250] / [batch_size, 3, 42]
            ori [batch_size, 1]
            pos [batch_size, 2]
            timesteps [batch_size,]
            object_pts [batch_size, 1024, 3]
        output: 
            profile [batch_size, 9]
        '''
        x_ctrl = self.gripper_encoder(x_ctrl[:, 1, :])
        x_ori = self.ori_embed(x_ori)
        x_pos = self.pos_embed(x_pos)
        x_pose = torch.cat([x_ori, x_pos], dim=1)
        x_object, _ = self.object_encoder(object_vertices)
        time_emb = timestep_embedding(timesteps, self.time_embed_dim)
        x = self.linears(torch.cat([x_object, x_ctrl, x_pose, time_emb], dim=1))
        x = self.output(x)
        return x