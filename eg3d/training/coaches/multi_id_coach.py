import torch
from tqdm import tqdm
from training.coaches.base_coach import BaseCoach


class MultiIDCoach(BaseCoach):

    def __init__(self, network_pkl: str, data_loader, device: torch.device, lr: float, outdir: str = "output"):
        super().__init__(network_pkl, data_loader, device, lr, outdir=outdir)

    def train(self, run_name: str, nb_steps: int, limit: int = -1, lpips_threshold: float = 0):
        self.G.synthesis.train()
        self.G.mapping.train()
        use_ball_holder = True

        i = 0
        for _ in tqdm(range(nb_steps)):
            self.image_counter = 0
            for img_name, imgs, ws_pivots in self.data_loader:
                if self.image_counter >= limit > 0:
                    break

                real_images_batch = imgs.to(self._device)
                generated_images = self.forward(ws_pivots)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, self.G, use_ball_holder, ws_pivots)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = i % self.locality_regularization_interval == 0
                i += 1
                self.image_counter += 1

        torch.save(self.G, f'{self.outdir}/model_{run_name}_multi_id.pt')
