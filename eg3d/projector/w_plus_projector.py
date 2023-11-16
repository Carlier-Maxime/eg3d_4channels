# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import numpy as np
import torch

from projector.w_projector import EG3DInverter


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

    def get_w_all(self, G, initial_w=None):
        start_w, w_std = self.computeWStats(G, initial_w)
        start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)
        w_opt = torch.tensor(start_w, dtype=torch.float32, device=self.device, requires_grad=True)
        return start_w, w_std, w_opt
