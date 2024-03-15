import os
import copy
import pickle
from typing import Dict

import dnnlib
from torch_utils import misc

from training.generator import Generator
from torch_utils.multiprocessing import launch_multiprocessing, init_distributed
import torch
import legacy
import PIL.Image


class BaseCoach:
    def __init__(self, network_pkl: str, dataset, device: torch.device, lr: float, l2_lambda: float = 1, lpips_lambda: float = 1, loss_lmks_lambda: float = 1, use_locality_regularization: bool = False, lpips_type: str = "alex", locality_regularization_interval: int = 1, outdir: str = "output", network_lmks: str | None = None):
        self.loss_lmks_lambda = loss_lmks_lambda
        self.space_regulizer = None
        self.__original_G = None
        self.__G = None
        self.__lmkDetector = None
        self.network_pkl = network_pkl
        self.network_lmks = network_lmks
        self.dataset = dataset
        self.__device = device
        self.lr = lr
        self.w_pivots = {}
        self.l2_lambda = l2_lambda
        self.lpips_lambda = lpips_lambda
        self.use_locality_regularization = use_locality_regularization
        self.lpips_type = lpips_type
        self.locality_regularization_interval = locality_regularization_interval
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.sampling_multiplier = 2
        self._use_ball_holder = True
        self.num_gpus = 1
        self.batch_gpu = 4
        self.seed = 0
        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        self.restart_training()

    def restart_training(self):
        if self.network_lmks is not None:
            with open(self.network_lmks, 'rb') as f:
                self.__lmkDetector = pickle.Unpickler(f).load().to(self.__device).requires_grad_(True)
        with open(self.network_pkl, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.__device)  # type: ignore
            self.__G = Generator(**G.init_kwargs).requires_grad_(False).to(self.__device)
            misc.copy_params_and_buffers(G, self.__G if list(G.named_parameters())[0][0].startswith("triplane.") else self.__G.triplane, require_all=False)
        self.__G.rendering_kwargs['depth_resolution'] = int(self.__G.rendering_kwargs['depth_resolution'] * self.sampling_multiplier)
        self.__G.rendering_kwargs['depth_resolution_importance'] = int(self.__G.rendering_kwargs['depth_resolution_importance'] * self.sampling_multiplier)
        for p in self.__G.parameters():
            p.requires_grad = True
        self.__original_G = copy.deepcopy(self.__G)

    def getOriginalG(self):
        return self.__original_G

    def getCopyG(self):
        return copy.deepcopy(self.__G)

    def getCopyLmkDetector(self):
        return copy.deepcopy(self.__lmkDetector)

    def train(self, coach_instance, num_gpus: int, batch, random_seed: int = 0, use_idr_torch: bool = False, **kwargs):
        args = dnnlib.EasyDict(kwargs)
        args.coach = self
        args.num_gpus = num_gpus
        args.coach_instance = coach_instance
        args.device = self.__device
        self.num_gpus = num_gpus
        self.seed = random_seed
        self.batch_gpu = batch // self.num_gpus
        launch_multiprocessing(BaseCoach.subprocess_fn, args, use_idr_torch=use_idr_torch)

    @staticmethod
    def subprocess_fn(rank, args, temp_dir, local_rank=-1):
        init_distributed(rank, temp_dir, args)
        device = torch.device('cuda', rank if local_rank == -1 else local_rank) if args.num_gpus > 1 else args.device
        G, lmkDetector = args.coach_instance(args.coach, rank, device, logdir=f"{args.coach.outdir}/{args.run_name}", lpips_type=args.coach.lpips_type).train(**args)
        if rank == 0:
            args.coach.__G = G
            args.coach.__lmkDetector = lmkDetector

    def calc_loss(self, lpips, spaceRegulizer, generated_images, real_images, gen_lmks, lmks, new_G, use_ball_holder, w_batch) -> Dict[str, torch.Tensor | None]:
        loss = torch.tensor(0., device=generated_images.device, dtype=torch.float)
        l2_loss_val = None
        loss_lpips = None
        loss_lmks_val = None
        if self.l2_lambda > 0:
            l2_loss_val = self.l2_loss(generated_images, real_images)
            loss += l2_loss_val * self.l2_lambda
        if self.lpips_lambda > 0:
            loss_lpips = lpips(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips).mean()
            loss += loss_lpips * self.lpips_lambda
        if use_ball_holder and self.use_locality_regularization:
            ball_holder_loss_val = spaceRegulizer.space_regulizer_loss(new_G, w_batch)
            loss += ball_holder_loss_val
        if gen_lmks is not None:
            loss_lmks_val = self.l2_loss(gen_lmks.flatten(), lmks.flatten())
            loss += loss_lmks_val * self.loss_lmks_lambda
        return {"loss": loss, "lpips": loss_lpips, "l2": l2_loss_val, "lmks": loss_lmks_val}

    def save_preview(self, run_name: str, name: str, gen_img: torch.Tensor):
        preview_path = f'{self.outdir}/{run_name}/preview_for_{name}.png'
        gen_img = (gen_img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(gen_img.cpu().numpy(), 'RGBA' if gen_img.shape[-1] == 4 else 'RGB').save(preview_path)
