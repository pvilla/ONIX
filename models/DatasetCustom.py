"""
    This is an example of preparing the dataset for training.
    We show the case of phase+attenuation training, for the case of attenuation only, please remove the phase part.
"""

from random import randint
import numpy as np
import torch
from models.mat_calc import get_world_mat
from models.utils import data_reg


class CustomDataset(torch.utils.data.Dataset):
    """Load data [n_views,H,W]"""

    def __init__(self, opt):
        super().__init__()

        self.n_views = opt.n_views
        path = opt.load_path
        self.total_views = 8  #  total number of views used in the training. Change it to match the experimental setup

        images = np.load(
            path
        )  # For ph+att: [dataset_size, num_projections, 2, H, W], otherwise [dataset_size, num_projections, H, W]
        att = images[:, :, :1]
        att = data_reg(att)
        ph = images[:, :, 1:]
        ph = data_reg(ph)  # remove if not using phase
        images = np.concatenate((att, ph), axis=2)  # remove if not using phase

        self.images_pool = torch.from_numpy(images)[
            : images.shape[0], : self.total_views
        ]
        self.generate_poses()

    def generate_poses(self):
        ProjAngles = np.linspace(0.0, 180.0, 10)[
            : self.total_views
        ]  # NB: Modify this part to match the experimental setup
        theta_x = 0
        theta_z = 0
        theta_y = ProjAngles
        world_mat = []
        for i in range(ProjAngles.shape[0]):
            world_mat.append(get_world_mat(theta_x, theta_y[i], theta_z))
        world_matrix = np.asarray(world_mat).astype(float)
        self.tform_cam2world = torch.from_numpy(world_matrix).float()  # [100,4,4]

    def __len__(self):
        return self.images_pool.shape[0]

    def __getitem__(self, index):
        self.imgs = self.images_pool[index]
        self.poses = self.tform_cam2world
        return self.imgs, self.poses
