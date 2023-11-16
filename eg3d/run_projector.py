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

import dnnlib
import legacy
from projector.w_projector import EG3DInverter
from projector.w_plus_projector import EG3DInverterPlus


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

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR',
              default='w', show_default=True)
@click.option('--image_path', help='image_path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--c_path', help='camera parameters path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--num_steps', 'num_steps', type=int,
              help='Multiplier for depth sampling in volume rendering', default=500, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        latent_space_type: str,
        image_path: str,
        c_path: str,
        num_steps: int
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

    os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None:
        G.neural_rendering_resolution = nrr

    image = Image.open(image_path).convert('RGB')
    image_name = os.path.basename(image_path)[:-4]
    c = np.load(c_path)
    c = np.reshape(c, (1, 25))

    c = torch.FloatTensor(c).cuda()

    from_im = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5], device=device)
    std = torch.tensor([0.5, 0.5, 0.5], device=device)
    from_im = (from_im - mean[:, None, None]) / std[:, None, None]
    from_im = torch.nn.functional.interpolate(from_im.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)

    id_image = torch.squeeze((from_im + 1) / 2) * 255

    if latent_space_type == 'w':
        projector = EG3DInverter(outdir, device=torch.device('cuda'), w_avg_samples=600)
    else:
        projector = EG3DInverterPlus(outdir, device=torch.device('cuda'), w_avg_samples=600, image_log_step=1)
    w = projector.project(G, c, w_name=image_name, target=id_image, num_steps=num_steps)

    w = w.detach().cpu().numpy()
    np.save(f'{outdir}/{image_name}_{latent_space_type}/{image_name}_{latent_space_type}.npy', w)

    PTI_embedding_dir = f'./projector/PTI/embeddings/{image_name}'
    os.makedirs(PTI_embedding_dir, exist_ok=True)

    np.save(f'./projector/PTI/embeddings/{image_name}/{image_name}_{latent_space_type}.npy', w)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
