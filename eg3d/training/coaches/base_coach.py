import abc
import os
import copy
import pickle

from torch_utils import misc
from training.generator import Generator
from training.loss import SpaceRegulizer
import torch
from lpips import LPIPS
import legacy
import PIL.Image


class BaseCoach:
    def __init__(self, network_pkl: str, data_loader, device: torch.device, lr: float, l2_lambda: float = 1, lpips_lambda: float = 1, loss_lmks_lambda: float = 1, use_locality_regularization: bool = False, lpips_type: str = "alex", locality_regularization_interval: int = 1, outdir: str = "output", network_lmks: str | None = None):
        self.loss_lmks_lambda = loss_lmks_lambda
        self.optimizer = None
        self.space_regulizer = None
        self.original_G = None
        self.G = None
        self.lmkDetector = None
        self.network_pkl = network_pkl
        self.network_lmks = network_lmks
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
        self._step = 0
        self._use_ball_holder = True

        self.lpips_loss = LPIPS(net=self.lpips_type).to(self._device).eval()
        self.l2_loss = torch.nn.MSELoss(reduction='mean')

        self.tf_events = None
        try:
            import torch.utils.tensorboard as tensorboard
        except ImportError:
            tensorboard = None
            print("skipped : tensorboard, module not found")
        self.tensorboard = tensorboard
        self.restart_training()

    def restart_training(self):
        if self.network_lmks is not None:
            with open(self.network_lmks, 'rb') as f:
                self.lmkDetector = pickle.Unpickler(f).load().to(self._device).requires_grad_(True)
        with open(self.network_pkl, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self._device)  # type: ignore
            self.G = Generator(**G.init_kwargs).to(self._device)
            misc.copy_params_and_buffers(G, self.G, require_all=False)
        self.G.rendering_kwargs['depth_resolution'] = int(self.G.rendering_kwargs['depth_resolution'] * self.sampling_multiplier)
        self.G.rendering_kwargs['depth_resolution_importance'] = int(self.G.rendering_kwargs['depth_resolution_importance'] * self.sampling_multiplier)
        for p in self.G.parameters():
            p.requires_grad = True
        self.original_G = copy.deepcopy(self.G)
        self.space_regulizer = SpaceRegulizer(self.original_G, self.lpips_loss, self._device)
        self.optimizer = self.configure_optimizers()

    def train_step(self, imgs: torch.Tensor, ws_pivots: torch.Tensor, camera: torch.Tensor, pts: torch.Tensor | None, lpips_threshold: float = 0) -> torch.Tensor | None:
        imgs = imgs.to(self._device)
        ws_pivots = ws_pivots.to(self._device)
        camera = camera.to(self._device)
        pts = pts.to(self._device)
        gen_imgs, gen_lmks = self.forward(ws_pivots, camera)
        loss, l2_loss_val, loss_lpips, loss_lmks_val = self.calc_loss(gen_imgs, imgs, gen_lmks, pts, self.G, self._use_ball_holder, ws_pivots)
        assert torch.is_tensor(loss)
        if self.tf_events is not None:
            self.tf_events.add_scalar('Loss/loss', loss.item(), self._step)
            if loss_lpips is not None:
                self.tf_events.add_scalar('Loss/lpips', loss_lpips.item(), self._step)
            if l2_loss_val is not None:
                self.tf_events.add_scalar('Loss/l2', l2_loss_val, self._step)
            if loss_lmks_val is not None:
                self.tf_events.add_scalar('Loss/lmks', loss_lmks_val, self._step)

        self.optimizer.zero_grad()
        if loss_lpips <= lpips_threshold:
            return None, None
        loss.backward()
        self.optimizer.step()

        self._use_ball_holder = self._step % self.locality_regularization_interval == 0
        self._step += 1
        return gen_imgs, gen_lmks

    @abc.abstractmethod
    def train(self, run_name: str, nb_steps: int, limit: int = -1, lpips_threshold: float = 0, **kwargs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam((list(self.G.parameters()) + list(self.lmkDetector.parameters())) if self.lmkDetector is not None else self.G.parameters, lr=self.lr)
        return optimizer

    def calc_loss(self, generated_images, real_images, gen_lmks, lmks, new_G, use_ball_holder, w_batch) -> tuple[torch.Tensor, float | None, torch.Tensor | None, float | None]:
        loss = torch.tensor(0., device=generated_images.device, dtype=torch.float)
        l2_loss_val = None
        loss_lpips = None
        loss_lmks_val = None
        if self.l2_lambda > 0:
            l2_loss_val = self.l2_loss(generated_images, real_images)
            loss += l2_loss_val * self.l2_lambda
        if self.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips).mean()
            loss += loss_lpips * self.lpips_lambda
        if use_ball_holder and self.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch)
            loss += ball_holder_loss_val
        if gen_lmks is not None:
            loss_lmks_val = self.l2_loss(gen_lmks, lmks)
            loss += loss_lmks_val * self.loss_lmks_lambda
        return loss, l2_loss_val, loss_lpips, loss_lmks_val

    def forward(self, ws, c):
        planes = self.G.backbone.synthesis(ws)
        gen_lmks = self.lmkDetector(planes) if self.lmkDetector is not None else None
        return self.G.synthesis(ws, c, planes=planes, noise_mode='const')['image'], gen_lmks

    def save_preview(self, run_name: str, name: str, gen_img: torch.Tensor):
        preview_path = f'{self.outdir}/{run_name}/preview_for_{name}.png'
        gen_img = (gen_img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(gen_img.cpu().numpy(), 'RGBA' if gen_img.shape[-1] == 4 else 'RGB').save(preview_path)
