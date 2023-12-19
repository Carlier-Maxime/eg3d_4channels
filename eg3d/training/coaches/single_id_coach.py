import os
import torch
from tqdm import tqdm
from training.coaches.base_coach import BaseCoach


class SingleIDCoach(BaseCoach):
    def __init__(self, network_pkl: str, data_loader, device: torch.device, lr: float, outdir: str = "output"):
        super().__init__(network_pkl, data_loader, device, lr, outdir=outdir)

    def train(self, run_name: str, nb_steps: int, limit: int = -1, lpips_threshold: float = 0):
        os.makedirs(f'{self.outdir}/{run_name}', exist_ok=True)
        use_ball_holder = True
        i = 0
        for img_name, imgs, ws_pivots in tqdm(self.data_loader):
            self.restart_training()
            if self.image_counter >= limit > 0:
                break

            ws_pivots = ws_pivots.to(self._device)
            imgs = imgs.to(self._device)

            for _ in tqdm(range(nb_steps)):
                generated_images = self.forward(ws_pivots)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, imgs, self.G, use_ball_holder, ws_pivots)
                self.optimizer.zero_grad()

                if loss_lpips <= lpips_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = i % self.locality_regularization_interval == 0
                i += 1

            self.image_counter += 1

            save_dict = {
                'G_ema': self.G.state_dict()
            }
            save_path = f'{self.outdir}/{run_name}/model_for_{img_name}.pth'
            print('final model ckpt save to ', save_path)
            torch.save(save_dict, save_path)
