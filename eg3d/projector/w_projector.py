# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule
from tqdm import tqdm
import dnnlib
import PIL.Image
from camera_utils import LookAtPoseSampler

vgg16 = None


def _load_vgg16():
    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    url = './networks/vgg16.pt'
    global vgg16
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval()


def get_vgg16() -> None | RecursiveScriptModule:
    global vgg16
    if vgg16 is None:
        _load_vgg16()
    return vgg16


def initNoises(noise_buffs: list[torch.Tensor]):
    for buf in noise_buffs:
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True
    return noise_buffs


class EG3DInverter:
    def __init__(self,
                 outdir,
                 w_avg_samples=10000,
                 initial_learning_rate=0.01,
                 initial_noise_factor=0.05,
                 lr_rampdown_length=0.25,
                 lr_rampup_length=0.05,
                 noise_ramp_length=0.75,
                 regularize_noise_weight=1e5,
                 device: torch.device = 'cuda',
                 image_log_step=100
                 ):
        self.outdir = outdir
        self.w_avg_samples = w_avg_samples
        self.initial_learning_rate = initial_learning_rate
        self.initial_noise_factor = initial_noise_factor
        self.lr_rampdown_length = lr_rampdown_length
        self.lr_rampup_length = lr_rampup_length
        self.noise_ramp_length = noise_ramp_length
        self.regularize_noise_weight = regularize_noise_weight
        self.device = device
        self.image_log_step = image_log_step
        self.vgg16 = get_vgg16().to(device)
        self.w_type_name = 'w'

    def computeWStats(self, G, initial_w=None):
        w_avg_path = './w_avg.npy'
        w_std_path = './w_std.npy'
        if (not os.path.exists(w_avg_path)) or (not os.path.exists(w_std_path)):
            print(f'Computing W midpoint and stddev using {self.w_avg_samples} samples...')
            z_samples = np.random.RandomState(123).randn(self.w_avg_samples, G.z_dim)

            # use avg look at point
            camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=self.device)
            cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=self.device)
            focal_length = 4.2647  # FFHQ's FOV
            intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=self.device)
            c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            c_samples = c_samples.repeat(self.w_avg_samples, 1)

            w_samples = G.mapping(torch.from_numpy(z_samples).to(self.device), c_samples)  # [N, L, C]
            w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
            w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
            w_std = (np.sum((w_samples - w_avg) ** 2) / self.w_avg_samples) ** 0.5
        else:
            raise Exception(' ')

        start_w = initial_w if initial_w is not None else w_avg
        return start_w, w_std

    def getFeatures(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(0)
        target_images = target.to(self.device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        return self.vgg16(target_images, resize_images=False, return_lpips=True)

    def learningRateSchedule(self, step, num_steps, w_std, optimizer):
        t = step / num_steps
        w_noise_scale = w_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp *= min(1.0, t / self.lr_rampup_length)
        lr = self.initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return w_noise_scale

    def noiseRegularisation(self, noise_buffs, dist):
        reg_loss = 0.0
        for v in noise_buffs:
            noise = v[None, None, :, :] if v.dim() == 2 else v
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return dist + reg_loss * self.regularize_noise_weight

    def next_ws(self, G, step, num_steps, w_opt, w_std, optimizer):
        w_noise_scale = self.learningRateSchedule(step, num_steps, w_std, optimizer)
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        return (w_opt + w_noise).repeat([1, G.backbone.mapping.num_ws, 1])

    def loop(self, G, c, target_features, num_steps, outdir, optimizer, w_opt, w_std, noise_buffs):
        for step in tqdm(range(num_steps)):
            ws = self.next_ws(G, step, num_steps, w_opt, w_std, optimizer)
            synth_images = G.synthesis(ws, c, noise_mode='const')['image']

            if self.image_log_step != 0 and step % self.image_log_step == 0:
                with torch.no_grad():
                    vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')

            # Down sample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255 / 2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            # Features for synth images.
            synth_features = self.vgg16(synth_images, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum()

            loss = self.noiseRegularisation(noise_buffs, dist)

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_buffs:
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    def get_w_all(self, G, initial_w=None):
        start_w, w_std = self.computeWStats(G, initial_w)
        w_opt = torch.tensor(start_w, dtype=torch.float32, device=self.device, requires_grad=True)
        return start_w, w_std, w_opt

    def project(self,
                G, c,
                w_name: str,
                target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
                num_steps=1000,
                initial_w=None
                ):
        # assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
        outdir = f'{self.outdir}/{w_name}_{self.w_type_name}'
        os.makedirs(outdir, exist_ok=True)

        G = copy.deepcopy(G).eval().requires_grad_(False).to(self.device).float()  # type: ignore
        start_w, w_std, w_opt = self.get_w_all(G, initial_w)
        if target.dim() == 4:
            start_w = start_w.repeat(target.shape[0], axis=0)
            w_opt = w_opt.repeat(target.shape[0], 1, 1)
            w_opt = torch.tensor(w_opt)

        noise_buffs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}
        noise_buffs = list(noise_buffs.values())
        if target.dim() == 4:
            for i in range(len(noise_buffs)):
                noise_buffs[i] = noise_buffs[i][None].repeat(target.shape[0], 1, 1, 1)
        optimizer = torch.optim.Adam([w_opt] + noise_buffs, betas=(0.9, 0.999), lr=0.1)
        noise_buffs = initNoises(noise_buffs)

        self.loop(G, c, self.getFeatures(target), num_steps, outdir, optimizer, w_opt, w_std, noise_buffs)
        return w_opt.repeat([1, G.backbone.mapping.num_ws, 1])
