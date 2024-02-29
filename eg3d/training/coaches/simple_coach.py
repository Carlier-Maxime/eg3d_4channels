import os
import torch
from tqdm import tqdm
from training.coaches.base_coach import BaseCoach


class SimpleCoach(BaseCoach):
    def __init__(self, network_pkl: str, data_loader, device: torch.device, lr: float, outdir: str = "output", network_lmks: str | None = None):
        super().__init__(network_pkl, data_loader, device, lr, outdir=outdir, network_lmks=network_lmks)

    def save_snapshot(self, run_name: str, current_step: int, nb_steps: int, final: bool = False):
        torch.save(self.G, f'{self.outdir}/{run_name}/{"final" if final else "snapshot"}_model_{run_name}_multi_id_{current_step}_of_{nb_steps}PTI.pt')
        torch.save(self.lmkDetector, f'{self.outdir}/{run_name}/{"final" if final else "snapshot"}_model_{run_name}_lmks_{current_step}_of_{nb_steps}PTI.pt')
        os.makedirs(f'{self.outdir}/{run_name}/{current_step}_PTI', exist_ok=True)
        for img_name, _, ws_pivots, camera, _ in tqdm(self.data_loader, desc="Snapshot", unit="img", leave=False):
            ws_pivots = ws_pivots.to(self._device)
            camera = camera.to(self._device)
            gen_imgs, gen_lmks = self.forward(ws_pivots, camera)
            for name, gen_img in zip(img_name, gen_imgs):
                self.save_preview(f'{run_name}/{current_step}_PTI', name, gen_img)

    def train(self, run_name: str, nb_epochs: int, steps_per_batch: int, limit: int = -1, lpips_threshold: float = 0, restart_training_between_img_batch: bool = False, snap: int = 4, **kwargs):
        os.makedirs(f'{self.outdir}/{run_name}', exist_ok=True)
        self._use_ball_holder = True
        self._step = 0
        if self.tensorboard is not None:
            self.tf_events = self.tensorboard.SummaryWriter(log_dir=f"{self.outdir}/{run_name}")
        self.restart_training()
        total_steps = nb_epochs*len(self.data_loader)*steps_per_batch
        if total_steps > limit > 0: total_steps = limit
        snap *= 1000
        limit *= 1000
        next_snap = snap
        for _ in tqdm(range(nb_epochs), unit="epoch"):
            for img_name, imgs, ws_pivots, camera, pts in tqdm(self.data_loader, unit="batch", leave=False):
                if restart_training_between_img_batch: self.restart_training()
                if self.image_counter >= limit > 0:
                    self.save_snapshot(run_name, self._step, total_steps, final=True)
                    return

                for _ in tqdm(range(steps_per_batch), unit="step", leave=False):
                    gen_imgs, gen_lmks = self.train_step(imgs, ws_pivots, camera, pts, lpips_threshold)
                    if gen_imgs is None: break
                self.image_counter += len(imgs)
                if self._step >= next_snap:
                    self.save_snapshot(run_name, self._step, total_steps)
                    next_snap += snap
        self.save_snapshot(run_name, self._step, total_steps, final=True)
