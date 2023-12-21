import os

import torch
from tqdm import tqdm
from training.coaches.base_coach import BaseCoach


class MultiIDCoach(BaseCoach):

    def __init__(self, network_pkl: str, data_loader, device: torch.device, lr: float, outdir: str = "output"):
        super().__init__(network_pkl, data_loader, device, lr, outdir=outdir)

    def save_snapshot(self, run_name: str, current_step: int, nb_steps: int, final: bool = False):
        torch.save(self.G, f'{self.outdir}/{run_name}/{"final" if final else "snapshot"}_model_{run_name}_multi_id_{current_step}_of_{nb_steps}PTI.pt')
        os.makedirs(f'{self.outdir}/{run_name}/{current_step}_PTI', exist_ok=True)
        for img_name, _, ws_pivots, camera in self.data_loader:
            ws_pivots = ws_pivots.to(self._device)
            camera = camera.to(self._device)
            generated_images = self.forward(ws_pivots, camera)
            for name, gen_img in zip(img_name, generated_images):
                self.save_preview(f'{run_name}/{current_step}_PTI', name, gen_img)

    def train(self, run_name: str, nb_steps: int, limit: int = -1, lpips_threshold: float = 0, snapshot_step: int = 100, **kwargs):
        self._use_ball_holder = True
        self._step = 0
        stop = False
        if self.tensorboard is not None:
            self.tf_events = self.tensorboard.SummaryWriter(log_dir=f"{self.outdir}/{run_name}")
        for i in tqdm(range(nb_steps)):
            self.image_counter = 0
            for img_name, imgs, ws_pivots, camera in self.data_loader:
                if self.image_counter >= limit > 0:
                    break
                gen_imgs = self.train_step(imgs, ws_pivots, camera, lpips_threshold)
                if gen_imgs is None:
                    stop = True
                    break
                self.image_counter += len(imgs)
            if i % snapshot_step == 0:
                self.save_snapshot(run_name, i, nb_steps)
            if stop:
                break
        self.save_snapshot(run_name, nb_steps, nb_steps, final=True)
