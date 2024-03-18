import abc

import torch
from lpips import LPIPS
from torch.utils.data import DataLoader
from torch_utils import misc
from training.coaches.base_coach import BaseCoach
from training.loss import SpaceRegulizer

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None
    print("skipped : tensorboard, module not found")


class BaseCoachInstance(object):
    def __init__(self, coach_model: BaseCoach, rank: int, device: torch.device, logdir: str | None = None, lpips_type: str = "alex"):
        self.model = coach_model
        self.rank = rank
        self._device = device
        self.set_sampler = misc.SimpleSampler(dataset=self.model.dataset, rank=rank, num_replicas=self.model.num_gpus, seed=self.model.seed)
        self.data_loader = DataLoader(dataset=self.model.dataset, sampler=self.set_sampler, batch_size=self.model.batch_gpu)
        self.G = self.model.getCopyG().to(device)
        self.lmkDetector = self.model.getCopyLmkDetector().to(device)
        self._use_ball_holder = True
        self._local_step = 0
        self._is_master = rank == 0
        self.optimizer = torch.optim.Adam((list(self.G.parameters()) + list(self.lmkDetector.parameters())) if self.lmkDetector is not None else self.G.parameters(), lr=self.model.lr)
        self.tf_events = tensorboard.SummaryWriter(log_dir=logdir) if self._is_master and logdir is not None and tensorboard is not None else None
        self.lpips_loss = LPIPS(net=lpips_type).to(self._device).eval()
        self.space_regulizer = SpaceRegulizer(self.model.getOriginalG(), self.lpips_loss, self._device)

    @abc.abstractmethod
    def train(self, **kwargs):
        pass

    def train_step(self, imgs: torch.Tensor, ws_pivots: torch.Tensor, camera: torch.Tensor, pts: torch.Tensor | None, lpips_threshold: float = 0) -> torch.Tensor | None:
        imgs = imgs.to(self._device)
        ws_pivots = ws_pivots.to(self._device)
        camera = camera.to(self._device)
        pts = pts.to(self._device)
        self.optimizer.zero_grad()
        gen_imgs, gen_lmks = self.forward(ws_pivots, camera)
        all_loss = self.model.calc_loss(self.lpips_loss, self.space_regulizer, gen_imgs, imgs, gen_lmks, pts, self.G, self._use_ball_holder, ws_pivots)
        loss_no_grad = {key: val.clone().detach().requires_grad_(False) for key, val in all_loss.items() if val is not None}
        if all_loss["lpips"] is not None and all_loss["lpips"] <= lpips_threshold: return None, None
        all_loss['loss'].backward()
        params = [param for module in [self.G, self.lmkDetector] for param in module.parameters() if param.numel() > 0 and param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if self.model.num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= self.model.num_gpus
            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)
        self.optimizer.step()
        if self.model.num_gpus > 1:
            for loss in loss_no_grad.values():
                torch.distributed.all_reduce(loss)
                loss /= self.model.num_gpus
        if self._is_master:
            if self.tf_events is not None:
                for key, val in loss_no_grad.items():
                    self.tf_events.add_scalar('Loss/'+key, val.item(), self._local_step)
        self._use_ball_holder = self._local_step % self.model.locality_regularization_interval == 0
        self._local_step += 1
        return gen_imgs, gen_lmks

    def forward(self, ws, c):
        planes = self.G.backbone.synthesis(ws)
        gen_lmks = self.lmkDetector(planes) if self.lmkDetector is not None else None
        return self.G.synthesis(ws, c, planes=planes, noise_mode='const')['image'], gen_lmks

    def restart_training(self):
        if self._is_master: self.model.restart_training()
        self.G = self.model.getCopyG().to(self._device)
        self.lmkDetector = self.model.getCopyLmkDetector().to(self._device)
        for module in [self.G, self.lmkDetector]:
            if module is not None:
                for param in misc.params_and_buffers(module):
                    if param.numel() > 0 and self.model.num_gpus > 1:
                        torch.distributed.broadcast(param, src=0)
