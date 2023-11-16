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

from projector.w_projector import EG3DInverter, initNoises


class EG3DInverterPlus(EG3DInverter):
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
        super().__init__(outdir, w_avg_samples, initial_learning_rate, initial_noise_factor, lr_rampdown_length, lr_rampup_length, noise_ramp_length, regularize_noise_weight, device, image_log_step)
        self.w_type_name = 'w_plus'

    def next_ws(self, G, step, num_steps, w_opt, w_std, optimizer):
        w_noise_scale = self.learningRateSchedule(step, num_steps, w_std, optimizer)
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        return w_opt + w_noise

    def project(self,
                G, c,
                w_name: str,
                target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
                num_steps=1000,
                initial_w=None
                ):
        assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
        outdir = f'{self.outdir}/{w_name}_{self.w_type_name}'
        os.makedirs(outdir, exist_ok=True)

        G = copy.deepcopy(G).eval().requires_grad_(False).to(self.device).float()  # type: ignore
        start_w, w_std = self.computeWStats(G, initial_w)

        noise_buffs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}
        start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)
        w_opt = torch.tensor(start_w, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([w_opt] + list(noise_buffs.values()), betas=(0.9, 0.999), lr=0.1)
        noise_buffs = initNoises(noise_buffs)

        self.loop(G, c, self.getFeatures(target), num_steps, outdir, optimizer, w_opt, w_std, noise_buffs)
        del G
        return w_opt
