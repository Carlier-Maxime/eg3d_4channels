# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from gen_samples import gen_density_cube


# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, **kwargs):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, D_density, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0,
                 gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.D_density = D_density
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.r1_gamma_init = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def get_G_ws(self, z, c, swapping_prob, update_emas: bool = False, style_mixing: bool = True):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.lt(torch.rand((c.shape[0], 1), device=c.device), swapping_prob), c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if style_mixing and self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.lt(torch.rand([], device=ws.device), self.style_mixing_prob), cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        return ws

    def run_G(self, ws, c, neural_rendering_resolution, update_emas=False):
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                          torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                         dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def run_D_density(self, cube, c, update_emas: bool = False):
        return self.D_density(cube, c, update_emas)

    def density_reg_sigma(self, ws, perturbation_amplitude, nb_points: int = 1000):
        initial_coordinates = torch.rand((ws.shape[0], nb_points, 3), device=ws.device) * 2 - 1  # Front
        perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1 / 256) * perturbation_amplitude  # Behind
        all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        sigma_initial = sigma[:, :sigma.shape[1] // 2]
        sigma_perturbed = sigma[:, sigma.shape[1] // 2:]
        return sigma_initial, sigma_perturbed

    def density_reg_l1(self, ws, gain, perturbation_amplitude):
        sigma_initial, sigma_perturbed = self.density_reg_sigma(ws, perturbation_amplitude, nb_points=1000)
        TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        TVloss.mul(gain).backward()

    def density_reg_monotonic(self, ws, gain, fixed: bool = False):
        sigma_initial, sigma_perturbed = self.density_reg_sigma(ws, self.G.rendering_kwargs['box_warp'], nb_points=2000)
        monotonic_loss = torch.relu((sigma_initial if fixed else sigma_initial.detach()) - sigma_perturbed).mean() * 10
        monotonic_loss.mul(gain).backward()
        self.density_reg_l1(ws, gain, self.G.rendering_kwargs['box_warp'])

    def gen_logits(self, phase: str, ws, gen_c, gain, neural_rendering_resolution, blur_sigma):
        assert phase in ['Gmain', 'Dgen']
        with torch.autograd.profiler.record_function(phase + '_forward'):
            gen_img, _gen_ws = self.run_G(ws, gen_c, neural_rendering_resolution=neural_rendering_resolution, update_emas=phase == 'Dgen')
            gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=phase == 'Dgen')
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            loss = torch.nn.functional.softplus(-gen_logits if phase == 'Gmain' else gen_logits)
            if phase == 'Gmain': training_stats.report('Loss/G/loss', loss)
        with torch.autograd.profiler.record_function(phase + '_backward'):
            loss.mean().mul(gain).backward()
        return loss

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, real_cube=None, **kwargs):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'Gdensity', 'Ddensity']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}
        ws = self.get_G_ws(gen_z, gen_c, swapping_prob=swapping_prob)

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            self.gen_logits('Gmain', ws, gen_c, gain, neural_rendering_resolution, blur_sigma)

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0:
            if self.G.rendering_kwargs['reg_type'] == 'l1':
                self.density_reg_l1(ws, gain, self.G.rendering_kwargs['density_reg_p_dist'])
            elif self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
                self.density_reg_monotonic(ws, gain)
            elif self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
                self.density_reg_monotonic(ws, gain, fixed=True)
            else:
                raise NotImplementedError

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            loss_Dgen = self.gen_logits('Dgen', ws, gen_c, gain, neural_rendering_resolution, blur_sigma)

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        if phase == 'Gdensity':
            with torch.autograd.profiler.record_function(phase + '_forward'):
                gen_cube = gen_density_cube(self.G, ws, self.D_density.img_resolution, verbose=False, with_grad=True)
                gen_logits = self.run_D_density(gen_cube, gen_c)
                loss = torch.nn.functional.softplus(-gen_logits)
                training_stats.report(f'Loss/density/fake', loss)
            with torch.autograd.profiler.record_function(phase + '_backward'):
                loss.mean().mul(gain).backward()

        if phase == 'Ddensity':
            with torch.autograd.profiler.record_function(phase + '_forward'):
                cube = real_cube.detach().requires_grad_(True)
                logits = self.run_D_density(cube, gen_c, update_emas=True)
                loss = torch.nn.functional.softplus(logits)
                training_stats.report(f'Loss/density/real', loss)
            with torch.autograd.profiler.record_function(phase + '_backward'):
                loss.mean().mul(gain).backward()


# ----------------------------------------------------------------------------

class SpaceRegulizer:
    def __init__(self, original_G: torch.nn.Module, lpips_net: torch.nn.Module, device: torch.device, l2_lambda: float = 0.1, lpips_lambda: float = 0.1, alpha: float = 30):
        self.device = device
        self.original_G = original_G
        self.alpha = alpha
        self.morphing_regulizer_alpha = self.alpha
        self.lpips_loss = lpips_net
        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        self.l2_lambda = l2_lambda
        self.lpips_lambda = lpips_lambda

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = self.alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        self.morphing_regulizer_alpha * fixed_w + (1 - self.morphing_regulizer_alpha) * new_w_code

        return result_w

    @staticmethod
    def get_image_from_ws(w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch):
        loss = 0.0
        z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
        c_samples = np.random.randn(num_of_sampled_latents, self.original_G.c_dim)
        w_samples = self.original_G.mapping(z=torch.from_numpy(z_samples).to(self.device), c=torch.from_numpy(c_samples).to(self.device),
                                            truncation_psi=0.5)
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)
            with torch.no_grad():
                old_img = self.original_G.synthesis(w_code, torch.from_numpy(c_samples).to(self.device))
            if self.l2_lambda > 0:
                l2_loss_val = self.l2_loss(old_img, new_img)
                loss += l2_loss_val * self.l2_lambda
            if self.lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * self.lpips_lambda

        return loss / len(territory_indicator_ws)

    def space_regulizer_loss(self, new_G, w_batch, ball_num_of_samples: int = 1):
        ret_val = self.ball_holder_loss_lazy(new_G, ball_num_of_samples, w_batch)
        return ret_val
