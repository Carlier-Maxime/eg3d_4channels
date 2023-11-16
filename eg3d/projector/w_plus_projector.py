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

import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from projector.w_projector import computeWStats, get_vgg16, getFeatures, initNoises, learningRateSchedule, noiseRegularisation


def project(
        G,
        c,
        outdir,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        initial_w=None,
        image_log_step=100,
        w_name: str
):
    os.makedirs(f'{outdir}/{w_name}_w_plus', exist_ok=True)
    outdir = f'{outdir}/{w_name}_w_plus'
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore
    start_w, w_std = computeWStats(G, w_avg_samples, device, initial_w)

    # Setup noise inputs.
    noise_buffs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    vgg16 = get_vgg16().to(device)
    target_features = getFeatures(target)

    start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)
    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w_opt] + list(noise_buffs.values()), betas=(0.9, 0.999), lr=0.1)

    noise_buffs = initNoises(noise_buffs)

    for step in tqdm(range(num_steps)):
        w_noise_scale = learningRateSchedule(step, num_steps, w_std, optimizer, initial_learning_rate, initial_noise_factor, lr_rampdown_length, lr_rampup_length, noise_ramp_length)

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)
        synth_images = G.synthesis(ws, c, noise_mode='const')['image']

        if step % image_log_step == 0:
            with torch.no_grad():
                vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        loss = noiseRegularisation(noise_buffs, dist, regularize_noise_weight)

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_buffs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    del G
    return w_opt
