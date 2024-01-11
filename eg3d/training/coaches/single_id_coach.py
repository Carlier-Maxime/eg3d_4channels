import os
import torch
from tqdm import tqdm
from training.coaches.base_coach import BaseCoach


class SingleIDCoach(BaseCoach):
    def __init__(self, network_pkl: str, data_loader, device: torch.device, lr: float, outdir: str = "output", network_lmks: str | None = None):
        super().__init__(network_pkl, data_loader, device, lr, outdir=outdir, network_lmks=network_lmks)

    def train(self, run_name: str, nb_steps: int, limit: int = -1, lpips_threshold: float = 0, **kwargs):
        os.makedirs(f'{self.outdir}/{run_name}', exist_ok=True)
        self._use_ball_holder = True
        self._step = 0
        if self.tensorboard is not None:
            self.tf_events = self.tensorboard.SummaryWriter(log_dir=f"{self.outdir}/{run_name}")
        for img_name, imgs, ws_pivots, camera, pts in tqdm(self.data_loader):
            self.restart_training()
            if self.image_counter >= limit > 0:
                break

            gen_imgs = []
            for _ in tqdm(range(nb_steps)):
                gen_imgs, gen_lmks = self.train_step(imgs, ws_pivots, camera, pts, lpips_threshold)
                if gen_imgs is None:
                    break
            self.image_counter += len(imgs)

            save_dict = {
                'G_ema': self.G.state_dict(),
                'G_lmks': self.lmkDetector.state_dict()
            }

            for name, gen_img in zip(img_name, gen_imgs):
                torch.save(save_dict, f'{self.outdir}/{run_name}/model_for_{name}.pth')
                self.save_preview(run_name, name, gen_img)
