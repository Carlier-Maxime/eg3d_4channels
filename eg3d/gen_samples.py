# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
from typing import List, Tuple, Union

import PIL.Image
import click
import mrcfile
import numpy as np
import torch
from tqdm import tqdm

from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from gen_utils import parse_range, loadNetwork, create_samples
from torch_utils import misc
from training.generator import Generator


# ----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple):
        return s
    parts = s.split(',')
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    raise ValueError(f'cannot parse 2-vector {s}')


# ----------------------------------------------------------------------------

def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def gen_density_cube(G, ws, size: int, max_batch: int = 1000000, pad: int = 0, verbose: bool = True, with_grad: bool = False):
    samples, voxel_origin, voxel_size = create_samples(N=size, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)  # .reshape(1, -1, 3)
    samples = samples.repeat(ws.shape[0], 1, 1)
    samples = samples.to(ws.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=ws.device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=ws.device)
    transformed_ray_directions_expanded[..., -1] = -1
    planes = G.backbone.synthesis(ws, noise_mode='const')
    planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

    head = 0
    with tqdm(total=samples.shape[1], disable=not verbose) as pbar:
        with torch.enable_grad() if with_grad else torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = G.sample_planes(planes, samples[:, head:head + max_batch], transformed_ray_directions_expanded[:, :samples.shape[1] - head])['sigma']
                sigmas[:, head:head + max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.view((ws.shape[0], size, size, size)).flip(0)
    if pad != 0:
        pad_value = sigmas.min().abs().mul(-1.01)
        sigmas[:, pad] = pad_value
        sigmas[:, -pad:] = pad_value
        sigmas[:, :, :pad] = pad_value
        sigmas[:, :, -pad:] = pad_value
        sigmas[:, :, :, :pad] = pad_value
        sigmas[:, :, :, -pad:] = pad_value
    return sigmas


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--planes', 'save_planes', help='Save Planes into npy file', type=bool, required=False, metavar='BOOL', default=False, show_default=True, is_flag=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--random-camera', help='Use a random camera', type=bool, required=False, metavar='BOOL', default=False, show_default=True, is_flag=True)
@click.option('--single-camera', help='single camera', type=bool, required=False, metavar='BOOL', default=False, show_default=True, is_flag=True)
@click.option('--mapped-latents', help='Save mapped latents', type=bool, required=False, metavar='BOOL', default=False, show_default=True, is_flag=True)
def generate_images(
        network_pkl: str,
        seeds: List[int],
        truncation_psi: float,
        truncation_cutoff: int,
        outdir: str,
        shapes: bool,
        shape_res: int,
        fov_deg: float,
        shape_format: str,
        save_planes: bool,
        reload_modules: bool,
        random_camera: bool,
        single_camera: bool,
        mapped_latents: bool
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pretrained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    device = torch.device('cuda')
    G = loadNetwork(network_pkl, device)

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_data = G
        G = Generator(**G_data.init_kwargs).requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G_data, G if list(G_data.named_parameters())[0][0].startswith("triplane.") else G.triplane, require_all=False)
        G.neural_rendering_resolution = G_data.neural_rendering_resolution
        G.rendering_kwargs = G_data.rendering_kwargs

    outdir_ws = f'{outdir}/mapped_latents'
    outdir_planes = f'{outdir}/planes'
    os.makedirs(outdir, exist_ok=True)
    if mapped_latents:
        os.makedirs(outdir_ws, exist_ok=True)
    if save_planes:
        os.makedirs(outdir_planes, exist_ok=True)

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        if mapped_latents:
            np.save(f'{outdir_ws}/seed{seed:04d}.npy', ws.cpu().numpy())
        if save_planes:
            planes = G.backbone.synthesis(ws)
            np.save(f'{outdir_planes}/seed{seed:04d}.npy', planes.cpu().numpy())
            if not reload_modules:
                planes = None
        else:
            planes = None

        imgs = []
        angle_p = -0.2
        if random_camera:
            angles = (torch.rand((1 if single_camera else 3, 2), device=device) * 2 - 1) * torch.pi / 6
        else:
            angles = [(0, angle_p)] if single_camera else [(.4, angle_p), (0, angle_p), (-.4, angle_p)]
        for angle_y, angle_p in angles:
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            img = G.synthesis(ws, camera_params)['image'] if planes is None else G.synthesis(ws, camera_params, planes=planes)['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)

        img = torch.cat(imgs, dim=2)

        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGBA' if img.shape[-1] == 4 else 'RGB').save(f'{outdir}/seed{seed:04d}.png')

        if shapes:
            sigmas = gen_density_cube(G, ws, shape_res, pad=int(30 * shape_res / 256))[0].cpu().numpy()
            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
            elif shape_format == '.mrc':  # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
