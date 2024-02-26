import numpy as np
import torch
from torch.utils.data import Dataset

class GripperDataset(Dataset):
    def __init__(self, gripper_pts, gripper_pts_max_x, gripper_pts_min_x, gripper_pts_max_y, gripper_pts_min_y):
        self.gripper_pts = gripper_pts
        self.gripper_pts_max_x = gripper_pts_max_x
        self.gripper_pts_min_x = gripper_pts_min_x
        self.gripper_pts_max_y = gripper_pts_max_y
        self.gripper_pts_min_y = gripper_pts_min_y
        
    def __len__(self):
        return len(self.gripper_pts)
    
    def __getitem__(self, idx):
        # IMPORTANT: normalize the input to [-1, 1]
        ctrlpts = self.gripper_pts[idx, :, 1].astype(np.float32)
        ctrlpts = (ctrlpts - self.gripper_pts_min_y) / (self.gripper_pts_max_y - self.gripper_pts_min_y) * 2.0 - 1.0
        return ctrlpts.reshape((-1, 1))