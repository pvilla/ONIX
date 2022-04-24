"""
    Prepare the dataset for testing. We use only one object for testing.
    This is an example for the case of phase+attenuation training, for the case of attenuation only, please remove the phase part.
"""

from random import randint
import numpy as np
import torch
from models.mat_calc import get_world_mat
from models.utils import data_reg


class TestDataset(torch.utils.data.Dataset):
    """Load data"""

    def __init__(self, opt):
        super().__init__()
        self.n_views = opt.n_views
        path = opt.load_path

        # Load images
        images = np.load(
            path
        )  # For ph+att: [dataset_size, num_projections, 2, H, W], otherwise [dataset_size, num_projections, H, W]

        att = images[:, :, :1]
        att = data_reg(att)
        ph = images[:, :, 1:]
        ph = data_reg(ph)  # remove if not using phase
        images = np.concatenate((att, ph), axis=2)  # remove if not using phase

        self.images_pool = torch.from_numpy(images)
        self.generate_poses()

    def generate_poses(self):
        ProjAngles = np.linspace(
            0.0, 180.0, 10
        )  # match the angles of the experiments/simula
        theta_x = 0
        theta_z = 0
        theta_y = ProjAngles
        world_mat = []
        for i in range(ProjAngles.shape[0]):
            world_mat.append(get_world_mat(theta_x, theta_y[i], theta_z))
        world_matrix = np.asarray(world_mat).astype(float)
        self.tform_cam2world = torch.from_numpy(world_matrix).float()  # [100,4,4]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # self.obj_idx = randint(0,self.images_pool.shape[0]-1)
        self.obj_idx = 5  # We use a fixed validation object in each epoch. Change to random index if needed.
        self.imgs = self.images_pool[self.obj_idx : self.obj_idx + 1]
        self.poses = self.tform_cam2world
        return self.imgs, self.poses, self.obj_idx
