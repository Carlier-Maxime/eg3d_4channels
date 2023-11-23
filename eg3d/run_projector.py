# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import Optional, Tuple, Union

import click
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import dnnlib
import legacy
from projector.w_projector import EG3DInverter
from training.dataset import ImageFolderDataset
from training.triplane import TriPlaneGenerator


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple):
        return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f'cannot parse tuple {s}')


# ----------------------------------------------------------------------------

def inversion(G: TriPlaneGenerator, c, projector, image_names, target, num_steps, outdir, files_basename, sub_folder: str = '.', save_features_map: bool = False):
    ws = projector.project(G, c, image_names=image_names, target=target, num_steps=num_steps, sub_folder=sub_folder)
    if save_features_map:
        features_outdir = f'{outdir}/features_maps/{sub_folder}'
        os.makedirs(features_outdir, exist_ok=True)
        features_maps = G.backbone.synthesis(ws).detach().cpu().numpy()
        for file_basename, features_map in zip(files_basename, features_maps):
            np.save(f'{features_outdir}/{file_basename}.npy', features_map)
    ws = ws.detach().cpu().numpy()
    lmks_outdir = f'{outdir}/latents/{sub_folder}'
    os.makedirs(lmks_outdir, exist_ok=True)
    for file_basename, ws_ in zip(files_basename, ws):
        np.save(f'{lmks_outdir}/{file_basename}.npy', ws_)


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR', default='w', show_default=True)
@click.option('--image_path', help='image_path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--c_path', help='camera parameters path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--dataset', help='path to dataset for inverse all images content', type=str, default=None, metavar='STR', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--num_steps', 'num_steps', type=int, help='Multiplier for depth sampling in volume rendering', default=500, show_default=True)
@click.option('--img-log-step', 'image_log_step', type=int, help='number of step between image log', default=100, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--batch', type=int, help='batch size, used only with dataset', default=16, show_default=True)
@click.option('--limit', type=int, help='limit images, used only with dataset', default=-1, show_default=True)
@click.option('--save-features-map', type=bool, help='save features map', default=False, is_flag=True, show_default=True)
def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        latent_space_type: str,
        image_path: str,
        c_path: str,
        num_steps: int,
        dataset: str,
        image_log_step: int,
        batch: int,
        limit: int,
        save_features_map: bool
):
    """Render a latent vector interpolation video.
    Examples:
    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    Animation length and seed keyframes:
    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.
    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.
    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None:
        G.neural_rendering_resolution = nrr

    projector = EG3DInverter(outdir, device=torch.device('cuda'), w_avg_samples=600, image_log_step=image_log_step, repeat_w=latent_space_type == 'w')

    if dataset is None:
        image = Image.open(image_path).convert('RGB')
        image_name = os.path.basename(image_path)[:-4]
        c_ext = os.path.splitext(os.path.basename(c_path))[1]
        if c_ext == '.npy':
            c = np.load(c_path)
            c = np.reshape(c, (1, 25))
            c = torch.FloatTensor(c).cuda()
        else:
            c = torch.load(c_path)
            c = torch.cat((c[1].flatten(), c[0].flatten())).reshape(1, 25).to(device)

        from_im = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.5, 0.5, 0.5], device=device)
        std = torch.tensor([0.5, 0.5, 0.5], device=device)
        from_im = (from_im - mean[:, None, None]) / std[:, None, None]
        from_im = torch.nn.functional.interpolate(from_im.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        id_image = torch.squeeze((from_im + 1) / 2) * 255
        inversion(G, c, projector, [image_name], id_image[None], num_steps, outdir, [f"{image_name}_{latent_space_type}"], save_features_map=save_features_map)
    else:
        dataset = ImageFolderDataset(dataset, force_rgb=True, use_labels=True)
        dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, pin_memory=True)
        i = 0
        for img, c in dataloader:
            if i >= limit > -1:
                break
            img = img.to(device)
            c = c.to(device)
            print(c.shape, img.shape)
            image_names = []
            for j in range(i, i + batch):
                image_names.append(f'{j:08d}')
            inversion(G, c, projector, image_names, img, num_steps, outdir, [f"{j:08d}_{latent_space_type}" for j in range(i, i+batch)], sub_folder=f"{i//1000:05d}", save_features_map=save_features_map)
            i += batch


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
