import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dynamics.utils import sample_pts_from_mesh

class DynamicsDataset(Dataset):
    def __init__(self, dataset_dir, object_mesh_dir, fingers_3d, gripper_pts_max_x, gripper_pts_min_x, gripper_pts_max_y, gripper_pts_min_y, gripper_pts_max_z, gripper_pts_min_z, object_max_num_vertices=10, object_pts_max_x=0.05, object_pts_min_x=-0.05, object_pts_max_y=0.05, object_pts_min_y=-0.05, object_pts_max_z=0.05, object_pts_min_z=-0.05):
        self.fingers_3d = fingers_3d
        if fingers_3d:
            self.threshold = np.array([0.02, 0.001, 0.001])
            self.std = np.array([0.0312, 0.0016, 0.0026])
        else:
            self.threshold = np.array([0.03, 0.002, 0.003])
            self.std = np.array([0.0565, 0.0026, 0.0047])
        self.gripper_pts_max_x = gripper_pts_max_x
        self.gripper_pts_min_x = gripper_pts_min_x
        self.gripper_pts_max_y = gripper_pts_max_y
        self.gripper_pts_min_y = gripper_pts_min_y
        self.gripper_pts_max_z = gripper_pts_max_z
        self.gripper_pts_min_z = gripper_pts_min_z
        self.object_max_num_vertices = object_max_num_vertices
        self.object_pts_max_x = object_pts_max_x
        self.object_pts_min_x = object_pts_min_x
        self.object_pts_max_y = object_pts_max_y
        self.object_pts_min_y = object_pts_min_y
        self.object_pts_max_z = object_pts_max_z
        self.object_pts_min_z = object_pts_min_z
        self.data_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.npz'):
                    self.data_files.append(os.path.join(root, file))
        self.object_pts = {}    # used for caching object points
        self.object_mesh_dir = object_mesh_dir

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx], allow_pickle=True)['arr_0'].item()
        # normalize with std (already zero-mean)
        train_scores = np.stack([data['delta_theta']/self.std[0], data['delta_pos'][:, 0]/self.std[1], data['delta_pos'][:, 1]/self.std[2]], axis=1)
        train_scores = torch.from_numpy(train_scores).float()
        train_ctrlpts = data['ctrlpts']
        train_ctrlpts[..., 0] = (train_ctrlpts[..., 0] - self.gripper_pts_min_x) / (self.gripper_pts_max_x - self.gripper_pts_min_x) * 2.0 - 1.0
        train_ctrlpts[..., 1] = (train_ctrlpts[..., 1] - self.gripper_pts_min_y) / (self.gripper_pts_max_y - self.gripper_pts_min_y) * 2.0 - 1.0
        if self.fingers_3d:
            train_ctrlpts[..., 2] = (train_ctrlpts[..., 2] - self.gripper_pts_min_z) / (self.gripper_pts_max_z - self.gripper_pts_min_z) * 2.0 - 1.0
        train_ctrlpts = torch.from_numpy(train_ctrlpts).float()
        train_input_ori = data['obj_theta'] / np.pi - 1.0
        train_input_pos = data['obj_pos'][..., :2] / 0.03
        train_input_ori = torch.from_numpy(train_input_ori).float()
        train_input_pos = torch.from_numpy(train_input_pos).float()
        if self.fingers_3d:
            object_name = data['object_name']
            if object_name not in self.object_pts.keys():
                mesh_file = os.path.join(self.object_mesh_dir, object_name, 'model.obj')
                object_vertices = sample_pts_from_mesh(mesh_file, self.object_max_num_vertices)
                object_vertices[..., 0] = (object_vertices[..., 0] - self.object_pts_min_x) / (self.object_pts_max_x - self.object_pts_min_x) * 2.0 - 1.0
                object_vertices[..., 1] = (object_vertices[..., 1] - self.object_pts_min_y) / (self.object_pts_max_y - self.object_pts_min_y) * 2.0 - 1.0
                object_vertices[..., 2] = (object_vertices[..., 2] - self.object_pts_min_z) / (self.object_pts_max_z - self.object_pts_min_z) * 2.0 - 1.0
                self.object_pts[object_name] = object_vertices
            else:
                object_vertices = self.object_pts[object_name]
            object_vertices = torch.from_numpy(object_vertices).float()
        else:
            object_vertices = data['object_vertices']
            object_vertices[..., 0] = (object_vertices[..., 0] - self.object_pts_min_x) / (self.object_pts_max_x - self.object_pts_min_x) * 2.0 - 1.0
            object_vertices[..., 1] = (object_vertices[..., 1] - self.object_pts_min_y) / (self.object_pts_max_y - self.object_pts_min_y) * 2.0 - 1.0
            object_vertices = torch.from_numpy(object_vertices).float()
            object_vertices = torch.cat([object_vertices, torch.zeros(self.object_max_num_vertices - object_vertices.shape[0], 2)], dim=0)
        return {
            'ctrlpts': train_ctrlpts,
            'scores': train_scores,
            'input_ori': train_input_ori,
            'input_pos': train_input_pos,
            'object_vertices': object_vertices,
        }