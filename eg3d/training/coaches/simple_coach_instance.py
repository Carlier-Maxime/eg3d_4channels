import os
import torch
from tqdm import tqdm

from training.coaches.base_coach import BaseCoach
from training.coaches.base_coach_instance import BaseCoachInstance


class SimpleCoachInstance(BaseCoachInstance):
    def __init__(self, coach_model: BaseCoach, rank: int, device: torch.device, logdir: str | None = None, lpips_type: str = "alex"):
        super().__init__(coach_model, rank, device, logdir, lpips_type)
        self.outdir = self.model.outdir
        self.image_counter = 0

    def save_snapshot(self, run_name: str, current_step: int, nb_steps: int, final: bool = False):
        torch.save(self.G, f'{self.outdir}/{run_name}/{"final" if final else "snapshot"}_model_{run_name}_multi_id_{current_step}_of_{nb_steps}PTI.pt')
        torch.save(self.lmkDetector, f'{self.outdir}/{run_name}/{"final" if final else "snapshot"}_model_{run_name}_lmks_{current_step}_of_{nb_steps}PTI.pt')
        os.makedirs(f'{self.outdir}/{run_name}/{current_step}_PTI', exist_ok=True)
        for img_name, _, ws_pivots, camera, _ in tqdm(self.data_loader, desc="Snapshot", unit="img", leave=False):
            ws_pivots = ws_pivots.to(self._device)
            camera = camera.to(self._device)
            gen_imgs, gen_lmks = self.forward(ws_pivots, camera)
            for name, gen_img in zip(img_name, gen_imgs):
                self.model.save_preview(f'{run_name}/{current_step}_PTI', name, gen_img)

    def train(self, run_name: str, nb_epochs: int, steps_per_batch: int, limit: int = -1, lpips_threshold: float = 0, restart_training_between_img_batch: bool = False, snap: int = 4, **kwargs):
        os.makedirs(f'{self.outdir}/{run_name}', exist_ok=True)
        self._use_ball_holder = True
        self._local_step = 0
        total_steps = nb_epochs*len(self.data_loader)*steps_per_batch
        if total_steps > limit > 0: total_steps = limit
        snap *= 1000
        limit *= 1000
        next_snap = snap
        for _ in tqdm(range(nb_epochs), unit="epoch", disable=not self._is_master):
            for img_name, imgs, ws_pivots, camera, pts in tqdm(self.data_loader, unit="batch", leave=False, disable=not self._is_master):
                if restart_training_between_img_batch: self.restart_training()
                if self.image_counter >= limit > 0:
                    if self._is_master: self.save_snapshot(run_name, self._local_step, total_steps, final=True)
                    return self.G, self.lmkDetector

                for _ in tqdm(range(steps_per_batch), unit="step", leave=False, disable=not self._is_master):
                    gen_imgs, gen_lmks = self.train_step(imgs, ws_pivots, camera, pts, lpips_threshold)
                    if gen_imgs is None: break
                self.image_counter += len(imgs)
                if self._local_step >= next_snap:
                    if self._is_master: self.save_snapshot(run_name, self._local_step, total_steps)
                    next_snap += snap
        if self._is_master: self.save_snapshot(run_name, self._local_step, total_steps, final=True)
        return self.G, self.lmkDetector
