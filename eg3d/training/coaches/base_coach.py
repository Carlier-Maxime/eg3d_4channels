import abc
import os
import copy
from training.loss import SpaceRegulizer
import torch
from lpips import LPIPS


class BaseCoach:
    def __init__(self, network_pkl: str, data_loader, device: torch.device, lr: float, l2_lambda: float = 1, lpips_lambda: float = 1, use_locality_regularization: bool = False, lpips_type: str = "alex", locality_regularization_interval: int = 1, outdir: str = "output"):
        self.optimizer = None
        self.space_regulizer = None
        self.original_G = None
        self.G = None
        self.network_pkl = network_pkl
        self.data_loader = data_loader
        self._device = device
        self.lr = lr
        self.w_pivots = {}
        self.image_counter = 0
        self.l2_lambda = l2_lambda
        self.lpips_lambda = lpips_lambda
        self.use_locality_regularization = use_locality_regularization
        self.lpips_type = lpips_type
        self.locality_regularization_interval = locality_regularization_interval
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.sampling_multiplier = 2

        self.lpips_loss = LPIPS(net=self.lpips_type).to(self._device).eval()
        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        self.restart_training()

    def restart_training(self):
        with open(self.network_pkl, 'rb') as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self._device)  # type: ignore
        self.G.rendering_kwargs['depth_resolution'] = int(self.G.rendering_kwargs['depth_resolution'] * self.sampling_multiplier)
        self.G.rendering_kwargs['depth_resolution_importance'] = int(self.G.rendering_kwargs['depth_resolution_importance'] * self.sampling_multiplier)
        for p in self.G.parameters():
            p.requires_grad = True
        self.original_G = copy.deepcopy(self.G)
        self.space_regulizer = SpaceRegulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    @abc.abstractmethod
    def train(self, run_name: str, nb_steps: int, limit: int = -1, lpips_threshold: float = 0):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        return optimizer

    def calc_loss(self, generated_images, real_images, new_G, use_ball_holder, w_batch):
        loss = 0.0
        l2_loss_val = None
        loss_lpips = None
        if self.l2_lambda > 0:
            l2_loss_val = self.l2_loss(generated_images, real_images)
            loss += l2_loss_val * self.l2_lambda
        if self.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * self.lpips_lambda
        if use_ball_holder and self.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch)
            loss += ball_holder_loss_val
        return loss, l2_loss_val, loss_lpips

    def forward(self, ws):
        return self.G.backbone.synthesis(ws, noise_mode='const')['image']
