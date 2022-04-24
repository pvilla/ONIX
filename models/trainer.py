"""
    Trainer for ONIX
"""

import sys
import os
import itertools
from random import randint
from collections import OrderedDict
from abc import ABC
import numpy as np
import torch
import torch.nn.functional as F
from models.encoder import SpatialEncoder
from models.resnetfc import ResnetFC
from models.DatasetCustom import CustomDataset
from models.DatasetTest import TestDataset
from models.utils import (
    init_weights,
    get_embedding_function,
    get_minibatches,
    get_ray_bundle,
    repeat_interleave,
    sample_pdf,
    cumprod_exclusive,
    save_tensor_plot,
)


class TrainModel(ABC):
    """ONIX model"""

    def __init__(self, opt):
        self.num_objs = 0
        self.opt = opt
        self.dtype = torch.float
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_train = True
        self.loss_names = ["coarse", "fine", "total"]
        self.generate_3D = False  # for eval

        self.seed_torch = 20210823
        self.seed_rand = 20210823
        torch.manual_seed(self.seed_torch)
        np.random.seed(self.seed_rand)
        self.val_names = ["val"]

        if opt.in_channel == 1:
            self.with_phase = False
        elif opt.in_channel == 2:
            self.with_phase = True
        else:
            raise ValueError("Only one or two input channels are supported yet!")

    def save_parameters(self):
        with open(self.save_log, "w+") as f:
            print(self.opt.__dict__, file=f)

    def load_data(self):
        print("start loading data....")
        if self.is_train:
            self.train_dataset = CustomDataset(self.opt)
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.opt.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )
        else:
            self.train_loader = []
        self.test_dataset = TestDataset(self.opt)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        print("finish loading data")

    def init_in_out(self):
        d_in = self.opt.num_encoding_fn_xyz * 6
        if self.opt.include_input_xyz:
            d_in += 3
        self.d_latent = self.latent_size if self.opt.use_encoder else 0
        self.d_in = d_in
        self.d_out = self.opt.d_out

    def init_model(self):
        if self.is_train:
            self.lambda_mse = self.opt.lambda_mse
            self.clip_max = self.opt.clip_max
        if not self.generate_3D:
            self.save_run = f"{self.opt.run_path}/{self.opt.run_name}"
            self.save_log = f"{self.save_run}/log.txt"
            self.save_val = f"{self.save_run}/val.txt"
            self.create_dir_if_not_exist(self.save_run)
        else:
            save_3d_path = f"{self.opt.model_path}/eval"
            self.save_run = save_3d_path
            self.create_dir_if_not_exist(save_3d_path)

        if self.opt.use_encoder:
            self.encoder = SpatialEncoder(
                in_channel=self.opt.in_channel,
                backbone=self.opt.backbone,
                pretrained=self.opt.encoder_pretrain,
                num_layers=self.opt.encoder_num_layers,
                use_first_pool=self.opt.use_first_pool,
            )
            self.latent_size = self.encoder.latent_size
            self.encoder.to(self.device)

        self.init_in_out()
        self.model_coarse = ResnetFC(
            d_in=self.d_in,
            d_latent=self.d_latent,
            d_out=self.opt.d_out,
            n_blocks=self.opt.n_blocks,
            d_hidden=self.opt.d_hidden,
            combine_layer=self.opt.combine_layer,
            combine_type=self.opt.combine_type,
        )
        self.model_coarse.to(self.device)
        init_weights(self.model_coarse, "normal", init_gain=0.02)
        trainable_parameters = list(self.model_coarse.parameters())
        if self.opt.use_encoder:
            trainable_parameters += list(self.encoder.parameters())

        if self.opt.num_fine > 0:
            self.model_fine = ResnetFC(
                d_in=self.d_in,
                d_latent=self.d_latent,
                d_out=self.opt.d_out,
                n_blocks=self.opt.n_blocks,
                d_hidden=self.opt.d_hidden,
                combine_layer=self.opt.combine_layer,
                combine_type=self.opt.combine_type,
            )
            self.model_fine.to(self.device)
            init_weights(self.model_fine, "normal", init_gain=0.02)
            trainable_parameters += list(self.model_fine.parameters())
        else:
            self.model_fine = None

        self.init_embedding()
        if self.is_train:
            self.optimizer = torch.optim.Adam(
                trainable_parameters, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
        self.load_data()
        if self.is_train:
            self.total_step = len(self.train_loader)

    def init_embedding(self):
        self.encode_position_fn = get_embedding_function(
            num_encoding_functions=self.opt.num_encoding_fn_xyz,
            include_input=self.opt.include_input_xyz,
            log_sampling=self.opt.log_sampling_xyz,
        )

    def run_one_iter_of_nerf(
        self,
        mode="train",
    ):
        batches = get_minibatches(self.rays, chunksize=self.opt.chunksize)
        pred = [
            self.predict_and_render_radiance(batch) for (i, batch) in enumerate(batches)
        ]
        synthesized_images = list(zip(*pred))
        synthesized_images = [
            torch.cat(image, dim=-1) if image[0] is not None else (None)
            for image in synthesized_images
        ]
        if mode == "validation":
            restore_shapes = [
                self.ray_directions.shape[:-1],
                self.ray_directions.shape[:-1],
                self.ray_directions.shape[:-1],
            ]
            if self.model_fine:
                restore_shapes += restore_shapes
            synthesized_images = [
                image.view(shape) if image is not None else None
                for (image, shape) in zip(synthesized_images, restore_shapes)
            ]
            if self.model_fine:
                return tuple(synthesized_images)
            else:
                if self.with_phase:
                    return tuple(synthesized_images + [None])
                else:
                    return tuple(synthesized_images)

        return tuple(synthesized_images)

    def set_input(self, input_data):
        """images: [n_views, H, W]; poses: [n_views, 4, 4]"""
        self.all_images = (
            input_data[0].to(self.device, dtype=self.dtype).squeeze(0)
        )  # 100,256,256
        self.all_poses = (
            input_data[1].to(self.device, dtype=self.dtype).squeeze(0)
        )  # 100,4,4

    def get_encode_views(self):
        """Generate"""
        self.SB, self.NV, _, self.H, self.W = self.all_images.shape
        if self.opt.n_views > self.NV:
            raise ValueError(
                f"Selected number of input should be less than the number of available input! Only {self.NV} inputs are available!"
            )
        # Choose encode views
        self.r_index = torch.randperm(self.NV)[: self.opt.n_views].sort()[
            0
        ]  # select n views from the input
        self.encode_images = self.all_images[:, self.r_index]
        self.encode_pose = self.all_poses[:, self.r_index]
        self.encode_pose = self.encode_pose.reshape(-1, 4, 4)

    def get_ray_function(self):
        """Get ray origin and ray direction"""
        pose_target = self.all_poses[..., :3, :4].to(self.device)
        self.ray_origins, self.ray_directions = get_ray_bundle(
            self.H, self.W, pose_target[0]
        )

    def get_ray_sample_new(self):
        self.get_encode_views()
        self.get_ray_function()
        pix_inds = torch.randint(
            0, self.NV * self.H * self.W, (self.opt.num_random_rays,)
        )  # [1024]
        self.ray_origins = self.ray_origins.reshape(-1, self.ray_origins.shape[-1])[
            pix_inds
        ].to(self.device)
        self.ray_directions = self.ray_directions.reshape(
            -1, self.ray_directions.shape[-1]
        )[pix_inds].to(self.device)
        all_att_gt = []
        all_rays = []
        if self.with_phase:
            all_ph_gt = []
        for obj_idx in range(self.SB):
            if self.with_phase:
                att_gt = (
                    self.all_images[obj_idx, :, 0].reshape(-1)[pix_inds].to(self.device)
                )
                ph_gt = (
                    self.all_images[obj_idx, :, 1].reshape(-1)[pix_inds].to(self.device)
                )
            else:
                att_gt = self.all_images[obj_idx].reshape(-1)[pix_inds].to(self.device)
            rays = self.get_ray_batches()
            all_att_gt.append(att_gt)
            if self.with_phase:
                all_ph_gt.append(ph_gt)
            all_rays.append(rays)
        all_att_gt = torch.stack(all_att_gt)  # (SB, ray_batch_size, 2)
        all_rays = torch.stack(all_rays)
        self.target_att = all_att_gt.reshape(-1)
        if self.with_phase:
            all_ph_gt = torch.stack(all_ph_gt)
            self.target_ph = all_ph_gt.reshape(-1)
        self.rays = all_rays
        self.select_inds = pix_inds

    def get_ray_sample_gradient(self):
        """Sample rays by gradient"""
        self.get_encode_views()
        # Gradient sampling
        all_att_gt = []
        if self.with_phase:
            all_ph_gt = []
        all_rays = []
        all_prob = []
        for obj_idx in range(self.SB):
            self.get_ray_function()
            all_views = self.all_images[obj_idx].squeeze()
            if self.with_phase:
                all_gradient = self.get_gradient_batch(
                    all_views[:, self.opt.gradient_sampling_channel]
                )
            else:
                all_gradient = self.get_gradient_batch(all_views)
            gradient_flat = all_gradient.reshape(-1)
            sampling_weighted_gradient = torch.multinomial(
                gradient_flat, self.opt.num_random_rays
            )
            self.ray_origins = self.ray_origins.reshape(-1, self.ray_origins.shape[-1])[
                sampling_weighted_gradient
            ].to(self.device)
            self.ray_directions = self.ray_directions.reshape(
                -1, self.ray_directions.shape[-1]
            )[sampling_weighted_gradient].to(self.device)

            gradient_sample_prob = gradient_flat[sampling_weighted_gradient]
            if self.with_phase:
                att_gt = (
                    self.all_images[obj_idx, :, 0]
                    .reshape(-1)[sampling_weighted_gradient]
                    .to(self.device)
                )
                ph_gt = (
                    self.all_images[obj_idx, :, 1]
                    .reshape(-1)[sampling_weighted_gradient]
                    .to(self.device)
                )
            else:
                att_gt = (
                    self.all_images[obj_idx]
                    .reshape(-1)[sampling_weighted_gradient]
                    .to(self.device)
                )
            rays = self.get_ray_batches()
            all_att_gt.append(att_gt)
            if self.with_phase:
                all_ph_gt.append(ph_gt)
            all_rays.append(rays)
            all_prob.append(gradient_sample_prob)
        all_att_gt = torch.stack(all_att_gt)  # (SB, ray_batch_size, 2)
        all_rays = torch.stack(all_rays)
        all_prob = torch.stack(all_prob)
        self.rays = all_rays
        self.select_inds = sampling_weighted_gradient
        self.gradient_sample_prob = all_prob.reshape(-1)
        self.target_att = all_att_gt.reshape(-1)
        if self.with_phase:
            all_ph_gt = torch.stack(all_ph_gt)
            self.target_ph = all_ph_gt.reshape(-1)

    @staticmethod
    def get_gradient_numpy(img):
        """Get gradient using numpy"""
        sx, sy = [np.gradient(img)[i] + 1e-8 for i in range(2)]
        # Get square root of sum of squares
        gradient = np.hypot(sx, sy)
        return gradient

    def get_gradient_batch(self, images):
        """input (C,H,W) tensor, output gradient (C,H,W)"""
        images = images.cpu().numpy()
        gradient = np.empty_like(images)
        for i in range(images.shape[0]):
            gradient[i] = self.get_gradient_numpy(images[i])
        # gradient = torch.tensor(gradient)
        gradient = torch.from_numpy(gradient).to(dtype=self.dtype, device=self.device)
        return gradient

    def optimization(self):
        """The optimization function. Could be biased with gradient sampling"""
        self.is_train = True
        if self.opt.use_gradient_sampling:
            self.get_ray_sample_gradient()
        else:
            self.get_ray_sample_new()
        self.optimizer.zero_grad()
        self.model_coarse.train()
        if self.model_fine:
            self.model_fine.train()
        if self.opt.use_encoder:
            self.encoder.train()
        if self.with_phase:
            self.atten_coarse, self.atten_fine, self.ph_coarse, self.ph_fine = (
                None,
                None,
                None,
                None,
            )
            (
                self.atten_coarse,
                self.ph_coarse,
                self.atten_fine,
                self.ph_fine,
            ) = self.run_one_iter_of_nerf(mode="train")
            self.target_complex = torch.cat((self.target_att, self.target_ph), -1)
            self.complex_coarse = torch.cat((self.atten_coarse, self.ph_coarse), -1)
            self.coarse_loss = torch.nn.functional.mse_loss(
                self.complex_coarse, self.target_complex
            )
        else:
            self.atten_coarse, self.atten_fine = None, None
            self.atten_coarse, self.atten_fine = self.run_one_iter_of_nerf(
                mode="train",
            )
            self.coarse_loss = torch.nn.functional.mse_loss(
                self.atten_coarse, self.target_att
            )
        self.fine_loss = 0.0
        if self.atten_fine is not None:
            # NOT TESTED! We only used coarse model.
            if self.with_phase:
                self.complex_fine = torch.cat((self.atten_fine, self.ph_fine), -1)
                self.fine_loss = torch.nn.functional.mse_loss(
                    self.complex_fine, self.target_complex
                )
            else:
                self.fine_loss = torch.nn.functional.mse_loss(
                    self.atten_fine, self.target_att
                )
        self.total_loss = self.coarse_loss + self.fine_loss
        self.total_loss = self.total_loss * self.lambda_mse
        self.total_loss.backward()
        if self.clip_max != 0:
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.model_coarse.parameters(), self.encoder.parameters()
                ),
                self.clip_max,
            )

        self.optimizer.step()

    def validation(self):
        self.is_train = False
        if self.with_phase:
            self.phase_o = []
        self.attenuation_o = []
        if self.opt.num_val_views != 0:
            self.val_idx = randint(
                self.all_images.shape[1] - self.opt.num_val_views,
                self.all_images.shape[1] - 1,
            )
        else:
            self.val_idx = randint(0, self.all_images.shape[1] - 1)
        self.get_ray_validation()
        self.model_coarse.eval()
        if self.model_fine:
            self.model_fine.eval()
        if self.opt.use_encoder:
            self.encoder.eval()

        if self.with_phase:
            self.atten_coarse, self.atten_fine, self.ph_coarse, self.ph_fine = (
                None,
                None,
                None,
                None,
            )
            # target_ray_values = None
            (
                self.atten_coarse,
                self.ph_coarse,
                self.atten_fine,
                self.ph_fine,
            ) = self.run_one_iter_of_nerf(
                mode="validation",
            )

            self.image_coarse_att = self.atten_coarse.reshape(
                [self.all_images.shape[-2], self.all_images.shape[-1]]
            )[None, None, :, :]
            self.image_coarse_ph = self.ph_coarse.reshape(
                [self.all_images.shape[-2], self.all_images.shape[-1]]
            )[None, None, :, :]

            self.image_coarse = torch.cat(
                (self.image_coarse_att, self.image_coarse_ph), 1
            )

            if self.atten_fine is not None:
                # NOT TESTED! We only used coarse model.
                self.image_fine_att = self.atten_fine.reshape(
                    [self.all_images.shape[-2], self.all_images.shape[-1]]
                )[None, None, :, :]
                self.image_fine_ph = self.ph_fine.reshape(
                    [self.all_images.shape[-2], self.all_images.shape[-1]]
                )[None, None, :, :]
                self.image_fine = torch.cat(
                    (self.image_fine_att, self.image_fine_ph), 1
                )

            self.gt = self.gt.squeeze()[None, :, :, :]
        else:
            self.atten_coarse, self.atten_fine = None, None
            self.atten_coarse, self.atten_fine = self.run_one_iter_of_nerf(
                mode="validation",
            )
            self.image_coarse = self.atten_coarse.reshape(
                [self.all_images.shape[-2], self.all_images.shape[-1]]
            )[None, None, :, :]
            if self.atten_fine is not None:
                self.image_fine = self.atten_fine.reshape(
                    [self.all_images.shape[-2], self.all_images.shape[-1]]
                )[None, None, :, :]

            self.gt = self.gt.squeeze()[None, None, :, :]

        # save validation loss
        self.val_loss = torch.nn.functional.mse_loss(self.gt, self.image_coarse)
        if self.generate_3D:
            print(f"Valiation loss is: {self.val_loss:.6f}")
            self.attenuation_o = torch.stack(self.attenuation_o, 0)
            self.reshape_3d(
                self.attenuation_o, "attenuation.npy"
            )  # Save 3D as npy files in the working directory
            if self.with_phase:
                self.phase_o = torch.stack(self.phase_o, 0)
                self.reshape_3d(self.phase_o, "phase.npy")

    def reshape_3d(self, x, savename):
        x = x.detach().cpu().numpy()
        x = x.reshape(-1, x.shape[-1])
        x = x.reshape(*self.all_images.shape[-2:], self.all_images.shape[-1])
        np.save(savename, x)

    def get_ray_validation(self):
        SB, NV, _, H, W = self.all_images.shape
        if self.opt.n_views > NV:
            raise ValueError(
                f"Selected number of input should be less than the number of available input! Only {NV} inputs are available!"
            )
        self.image_shape = torch.tensor([H, W])
        try:
            self.r_index = torch.tensor(self.opt.val_view_idx)
        except:
            self.r_index = torch.randperm(NV - self.opt.num_val_views)[
                : self.opt.n_views
            ].sort()[0]

        self.encode_images = self.all_images[:, self.r_index]
        self.encode_pose = self.all_poses[self.r_index]
        pose_target = self.all_poses[self.val_idx : self.val_idx + 1, :3, :4].to(
            self.device
        )
        self.encode_pose = self.encode_pose[None, ...].repeat(SB, 1, 1, 1)
        self.encode_pose = self.encode_pose.reshape(-1, 4, 4)
        self.gt = self.all_images[:, self.val_idx : self.val_idx + 1]

        self.ray_origins, self.ray_directions = get_ray_bundle(H, W, pose_target)
        pix_inds = torch.arange(H * W)  # [1024]
        self.ray_origins = self.ray_origins.reshape(-1, self.ray_origins.shape[-1])[
            pix_inds
        ].to(self.device)
        self.ray_directions = self.ray_directions.reshape(
            -1, self.ray_directions.shape[-1]
        )[pix_inds].to(self.device)
        all_rays_gt = []
        all_rays = []
        for obj_idx in range(SB):
            if self.with_phase:
                rays_gt = (
                    self.all_images[obj_idx]
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .reshape(-1, 2)[pix_inds]
                    .to(self.device)
                )
            else:
                rays_gt = self.all_images[obj_idx].reshape(-1)[pix_inds].to(self.device)
            rays = self.get_ray_batches()

            all_rays_gt.append(rays_gt)
            all_rays.append(rays)
        all_rays_gt = torch.stack(all_rays_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)
        self.target_s = all_rays_gt.reshape(-1)
        self.rays = all_rays
        self.target_s = self.gt.reshape(-1, 1)[pix_inds].to(self.device)
        self.select_inds = pix_inds
        self.SB = SB

    def get_val_losses(self):
        errors_list = OrderedDict()
        for name in self.val_names:
            if isinstance(name, str):
                errors_list[name] = float(getattr(self, name + "_loss"))
        return errors_list

    def get_current_losses(self):
        errors_list = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_list[name] = float(getattr(self, name + "_loss"))
        return errors_list

    def encoder_index(self, uv, latent):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :return (B, L, N) L is latent size
        """
        uv = uv * 2
        uv = uv.unsqueeze(2)  # (B, N, 1, 2) (8, 2048, 1, 2)
        samples = F.grid_sample(
            latent,
            uv,
            align_corners=True,
            mode=self.opt.index_interp,
            padding_mode=self.opt.index_padding,
        )
        return samples[:, :, :, 0]  # (B, C, N)

    def encode(self, pts, images):

        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
        else:
            self.num_views_per_obj = 1

        latent = self.encoder(images)

        if not self.opt.use_camera_space_pnts:
            uv = -pts[:, :, :2] / (pts[:, :, 2:] + 1.5)
        else:
            uv = -pts[:, :, :2]
            uv[:, :, :1] = -uv[:, :, :1]
        uv = uv.to(self.dtype)
        latent = self.encoder_index(uv, latent)
        latent = latent.transpose(1, 2).reshape(-1, self.latent_size)
        return latent

    def get_ray_batches(self):
        z_near = self.opt.z_near * torch.ones_like(self.ray_directions[..., :1])
        z_far = self.opt.z_far * torch.ones_like(self.ray_directions[..., :1])
        self.rays = torch.cat(
            (self.ray_origins, self.ray_directions, z_near, z_far), dim=-1
        ).view(
            (-1, 8)
        )  # remove 1st dimension
        return self.rays

    def get_ray_pts(self, batch):
        ro, rd = batch[..., :3], batch[..., 3:6]
        bounds = batch[..., 6:8].view((batch.shape[0], -1, 1, 2))  # [2048, 1, 2]
        near, far = bounds[..., 0], bounds[..., 1]  # [2048,1]
        t_vals = torch.linspace(
            0.0,
            1.0,
            self.opt.num_coarse,
            dtype=self.dtype,
            device=self.device,
        )
        if not self.opt.lindisp:
            self.z_vals = near * (1.0 - t_vals) + far * t_vals
        else:
            self.z_vals = 1.0 / (
                1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals
            )  # [2048,64]
        if self.opt.perturb:
            # Get intervals between samples.
            mids = 0.5 * (self.z_vals[..., 1:] + self.z_vals[..., :-1])
            upper = torch.cat((mids, self.z_vals[..., -1:]), dim=-1)
            lower = torch.cat((self.z_vals[..., :1], mids), dim=-1)
            # Stratified samples in those intervals.
            t_rand = torch.rand(self.z_vals.shape, dtype=self.dtype, device=self.device)
            self.z_vals = lower + (upper - lower) * t_rand
        pts = ro[..., None, :] + rd[..., None, :] * self.z_vals[..., :, None]
        return pts

    def main_network(self, pts, coarse=True):
        # Convert camera pose matrix to a "transform matrix"
        if self.opt.use_camera_space_pnts:
            rot = self.encode_pose[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
            trans = -torch.bmm(rot, self.encode_pose[:, :3, 3:])  # (B, 3, 1)
            poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        pts_flat = pts.reshape((self.SB, -1, pts.shape[-1]))
        xyz = repeat_interleave(pts_flat, self.opt.n_views)  # (SB*NS, B, 3)

        if self.opt.use_camera_space_pnts:
            # Transform query points into the camera spaces of the input views
            xyz_rot = torch.matmul(poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
            xyz = xyz_rot + poses[:, None, :3, 3]
        if self.opt.use_encoder:
            latent = self.encode(xyz, self.encode_images)  # [2048,512]

        embedded = self.encode_position_fn(xyz.reshape(-1, 3))  # [2x132072,63]
        mlp_input = embedded  # [2048x64x2,90] --> output [2x1024x64,2]

        if self.opt.use_encoder:
            mlp_input = torch.cat((latent, mlp_input), dim=-1)  # [2048x64x2,512+90]

        _, B, _ = pts_flat.shape
        NS = self.opt.n_views
        # Run main NeRF network
        if coarse or self.model_fine is None:
            mlp_output = self.model_coarse(
                mlp_input,
                combine_inner_dims=(NS, B),
            )  # [1024x64x2,2]
        else:
            mlp_output = self.model_fine(
                mlp_input,
                combine_inner_dims=(NS, B),
            )  # [1024x64x2,2]

        mlp_output = mlp_output.reshape(-1, mlp_output.shape[-1])  #
        return mlp_output

    def predict_and_render_radiance(self, batch):
        pts = self.get_ray_pts(batch)
        radiance_field = self.main_network(pts, coarse=True)
        if self.is_train:
            radiance_field_noise_std = self.radiance_field_noise_std
        else:
            radiance_field_noise_std = 0.0
        if self.with_phase:
            (att_coarse, ph_coarse, weights,) = self.volume_render_xray_propagation(
                radiance_field,
                batch[0, ..., 3:6],
                self.z_vals,
                radiance_field_noise_std=radiance_field_noise_std,
            )
            att_fine, ph_fine = None, None
        else:
            (att_coarse, weights,) = self.volume_render_xray_propagation(
                radiance_field,
                batch[0, ..., 3:6],
                self.z_vals,
                radiance_field_noise_std=radiance_field_noise_std,
            )
            att_fine = None
        if self.opt.num_fine > 0:
            SB, R, K = self.z_vals.shape
            z_vals_mid = 0.5 * (self.z_vals[..., 1:] + self.z_vals[..., :-1])
            weights = weights[..., 1:-1].reshape([SB, R, K - 2])
            z_vals = torch.zeros([SB, R, self.opt.num_fine + self.opt.num_coarse]).to(
                self.z_vals
            )
            for obj_idx in range(SB):
                z_samples = sample_pdf(
                    z_vals_mid[obj_idx],
                    weights[obj_idx],
                    self.opt.num_fine,
                    det=(self.opt.perturb == 0.0),
                )
                z_samples = z_samples.detach()
                z_vals[obj_idx], _ = torch.sort(
                    torch.cat((self.z_vals[obj_idx], z_samples), dim=-1), dim=-1
                )
            ro, rd = batch[..., :3], batch[..., 3:6]
            pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
            radiance_field = self.main_network(pts, coarse=False)
            if self.with_phase:
                att_fine, ph_fine, _, = self.volume_render_xray_propagation(
                    radiance_field,
                    batch[0, ..., 3:6],
                    z_vals,
                    radiance_field_noise_std=radiance_field_noise_std,
                )
            else:
                att_fine, _, = self.volume_render_xray_propagation(
                    radiance_field,
                    batch[0, ..., 3:6],
                    z_vals,
                    radiance_field_noise_std=radiance_field_noise_std,
                )
        if self.with_phase:
            return att_coarse, ph_coarse, att_fine, ph_fine
        else:
            return att_coarse, att_fine

    def volume_render_xray_propagation(
        self,
        radiance_field: torch.Tensor,
        ray_directions: torch.Tensor,
        depth_values: torch.Tensor,
        radiance_field_noise_std=0.0,
    ):
        """
        Render the image using the law of X-ray propagation
        """
        one_e_10 = torch.tensor([1e-10], dtype=self.dtype, device=self.device)
        dists = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                one_e_10.expand(depth_values[..., :1].shape),
            ),
            dim=-1,
        )  # [2048,64]
        radiance_field = radiance_field.reshape(
            -1, dists.shape[-1], radiance_field.shape[-1]
        )
        dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)  # [2048,64]
        dists = dists.reshape(-1, dists.shape[-1])

        noise = 0.0
        if radiance_field_noise_std > 0.0:
            noise = (
                torch.randn(
                    radiance_field[..., 0].shape,
                    dtype=self.dtype,
                    device=self.device,
                )
                * radiance_field_noise_std
            )
        sigma_a = torch.nn.functional.relu(
            radiance_field[..., 0] + noise
        )  # (width, height, num_samples) # [2048, 64]
        if self.with_phase:
            phase = torch.nn.functional.relu(radiance_field[..., 1] + noise)
        alpha = 1.0 - torch.exp(-sigma_a * dists)
        weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

        if self.generate_3D:
            self.attenuation_o.append(sigma_a * dists)
            if self.with_phase:
                self.phase_o.append(phase * dists)
        attenuation_map = (sigma_a * dists).sum(
            dim=-1
        )  # Compute attenuation of each sample along each ray
        if self.with_phase:
            phase_map = (phase * dists).sum(dim=-1)
            return attenuation_map, phase_map, weights
        else:
            return attenuation_map, weights

    def print_current_losses(self, epoch, iters, losses):
        message = "Epoch [{}/{}], Step [{}/{}]".format(
            epoch + 1, self.opt.num_epochs, iters + 1, self.total_step
        )
        for name, loss in losses.items():
            message += ", {:s}: {:.3f}".format(name, loss)
        print(message)
        with open(
            self.save_log,
            "a",
            encoding="utf-8",
        ) as f:
            print(message, file=f)

    def print_val_losses(self, epoch, iters, losses):
        message = "Epoch [{}/{}], Step [{}/{}]".format(
            epoch + 1, self.opt.num_epochs, iters + 1, self.total_step
        )
        for name, loss in losses.items():
            message += ", {:s}: {:.3f}".format(name, loss)
        print(message)
        with open(self.save_val, "a+") as f:
            print(message, file=f)

    @staticmethod
    def create_dir_if_not_exist(path):
        if os.path.exists(path):
            print("Warning: Overwriting folder: {}".format(path))
        if not os.path.exists(path):
            os.makedirs(path)

    def save_inputs(self):
        for i in range(self.all_images.shape[0]):
            save_tensor_plot(self.all_images[i], self.save_run, f"input_images_{i}")

    def visual_iter(self, epoch, iteration, obj_id):
        save_name = "{:03d}epoch_{:04d}step".format(epoch + 1, iteration + 1)
        save_tensor_plot(
            self.atten_coarse.detach().reshape(
                [self.all_images.shape[-2], self.all_images.shape[-1]]
            ),
            self.save_run,
            f"{save_name}_coarse_att",
        )
        if self.with_phase:
            save_tensor_plot(
                self.ph_coarse.detach().reshape(
                    [self.all_images.shape[-2], self.all_images.shape[-1]]
                ),
                self.save_run,
                f"{save_name}_coarse_ph",
            )
            save_tensor_plot(
                self.gt.squeeze()[0],
                self.save_run,
                f"{save_name}_gt_att_obj{obj_id}_view{self.val_idx}_enc{self.r_index}",
            )
            save_tensor_plot(
                self.gt.squeeze()[1],
                self.save_run,
                f"{save_name}_gt_ph_obj{obj_id}_view{self.val_idx}_enc{self.r_index}",
            )
        else:
            save_tensor_plot(
                self.gt.squeeze(),
                self.save_run,
                f"{save_name}_gt_obj{obj_id}_view{self.val_idx}_enc{self.r_index}",
            )

        if self.atten_fine is not None:
            save_tensor_plot(
                self.atten_fine.detach().reshape(
                    [self.all_images.shape[-2], self.all_images.shape[-1]]
                ),
                self.save_run,
                f"{save_name}_fine_att",
            )
            if self.with_phase:
                save_tensor_plot(
                    self.ph_fine.detach().reshape(
                        [self.all_images.shape[-2], self.all_images.shape[-1]]
                    ),
                    self.save_run,
                    f"{save_name}_fine_ph",
                )

    def save_net(self, name, epoch, net, optimizer, loss):
        model_save_name = f"{name}_{epoch}ep.pt"
        path = f"{self.save_run}/save"
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving trained model {}".format(model_save_name))
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            path + f"/{model_save_name}",
        )

    def save_models(self, epoch):
        self.save_net(
            "coarse", epoch + 1, self.model_coarse, self.optimizer, self.coarse_loss
        )
        if self.atten_fine is not None:
            self.save_net(
                "fine", epoch + 1, self.model_fine, self.optimizer, self.fine_loss
            )
        if self.opt.use_encoder:
            self.save_net(
                "encoder", epoch + 1, self.encoder, self.optimizer, self.coarse_loss
            )

    def adjust_learning_rate(self, epoch, optimizer, lr_start):
        """Learning rate decay"""
        lr_new = lr_start * (
            self.opt.lr_decay_factor ** (epoch // self.opt.lr_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

    def adjust_noise_factor(self, epoch):
        self.radiance_field_noise_std = self.opt.radiance_field_noise_std * (
            self.opt.noise_decay_factor ** (epoch / self.opt.noise_decay_steps)
        )

    def update_parameters(self, epoch):
        self.adjust_learning_rate(epoch, self.optimizer, lr_start=self.opt.lr)
        self.adjust_noise_factor(epoch)  # Adjust the noise factor if needed

    def load_trained_models(self, load_path, load_epoch):
        # fine_path = f"{load_path}/save/fine_{load_epoch}ep.pt"
        coarse_path = f"{load_path}/save/coarse_{load_epoch}ep.pt"
        encoder_path = f"{load_path}/save/encoder_{load_epoch}ep.pt"
        coarse_checkpoint = torch.load(coarse_path)
        print("Loading model from ", coarse_path)
        encoder_checkpoint = torch.load(encoder_path)
        self.encoder.load_state_dict(
            encoder_checkpoint["model_state_dict"], strict=False
        )
        self.model_coarse.load_state_dict(
            coarse_checkpoint["model_state_dict"], strict=False
        )
        if self.opt.num_fine > 0:
            self.model_fine.load_state_dict(
                coarse_checkpoint["model_state_dict"], strict=False
            )
        print(f"Finish loading {encoder_path}")