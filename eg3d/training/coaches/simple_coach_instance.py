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
        self.snap_grid_size = (16, 9)
        size = int(self.G.img_resolution * 0.01)
        if self._is_master:
            self.indices_circular_mask = torch.arange(-size, size + 1, device=device).repeat(size * 2 + 1, 1)
            circular_mask = self.indices_circular_mask.abs()
            self.indices_circular_mask = self.indices_circular_mask.permute(1, 0) * self.G.img_resolution + self.indices_circular_mask
            circular_mask = circular_mask.permute(1, 0).square().add(circular_mask.square()).sqrt()
            circular_mask = torch.where(circular_mask <= size, True, False)
            self.indices_circular_mask = self.indices_circular_mask[circular_mask].flatten()
        else:
            self.indices_circular_mask = None

    def save_snapshot(self, run_name: str, current_step: int, nb_steps: int, final: bool = False):
        if not self._is_master: return
        outdir = f'{self.outdir}/{run_name}' + (f'/{current_step}_PTI' if not final else '')
        os.makedirs(outdir, exist_ok=True)
        torch.save(self.G, f'{outdir}/{"final" if final else "snapshot"}_model_{run_name}_multi_id_{current_step}_of_{nb_steps}PTI.pt')
        torch.save(self.lmkDetector, f'{outdir}/{"final" if final else "snapshot"}_model_{run_name}_lmks_{current_step}_of_{nb_steps}PTI.pt')
        snap_imgs = []
        snap_lmks = []
        count = self.snap_grid_size[0] * self.snap_grid_size[1]
        bar = tqdm(total=count, desc="Prepare Images for Snapshot", unit="img", leave=False, disable=not self._is_master)
        for img_name, _, ws_pivots, camera, _ in self.data_loader:
            with torch.no_grad():
                ws_pivots = ws_pivots.to(self._device)
                camera = camera.to(self._device)
                gen_imgs, gen_lmks = self.forward(ws_pivots, camera)
                for img, lmk in zip(gen_imgs, gen_lmks):
                    if len(snap_imgs) >= count: break
                    snap_imgs.append(img.cpu())
                    snap_lmks.append(lmk)
            bar.update(len(ws_pivots))
            if len(snap_imgs) >= count: break
        bar.close()
        gw, gh = self.snap_grid_size
        C, H, W = snap_imgs[0].shape
        img_grid = torch.stack(snap_imgs, dim=0).view(gh, gw, C, H, W).permute(0, 1, 3, 4, 2)
        if snap_lmks[0].shape[1] == 2:
            lmks = torch.stack(snap_lmks, dim=0).view(gh, gw, -1, 2).to(torch.int)
            tmp = torch.where(((lmks >= 0) & (lmks < self.G.img_resolution)).all(dim=3))
            lmks = lmks[tmp[0], tmp[1], tmp[2]]
            lmks = lmks[:, 0] * self.G.img_resolution + lmks[:, 1]
            lmks = self.indices_circular_mask.repeat(lmks.shape[0], 1).permute(1, 0).add(lmks)
            mask = (lmks >= 0) & (lmks < self.G.img_resolution**2)
            lmks = ((tmp[0] * gw + tmp[1]) * (H*W) + lmks)[mask]
            img_grid = img_grid.reshape(-1, img_grid.shape[-1])
            img_grid[lmks.cpu()] = torch.tensor([1, -1, -1], dtype=img_grid.dtype)
            img_grid = img_grid.reshape(gh, gw, H, W, C)
        img_grid = img_grid.permute(0, 2, 1, 3, 4).contiguous().view(gh * H, gw * W, C).permute(2, 0, 1)
        self.model.save_preview(outdir, "preview", img_grid)

    def calcul_metrics(self, limit: int = 1):
        if limit == 0: return
        count = limit*1000 // self.model.num_gpus
        if self._is_master: count += limit % self.model.num_gpus
        bar = tqdm(total=count, desc=f"Calcul part of metrics ({self._device})", unit="img", leave=False)
        i = 0
        loss = 0
        div = 0
        for img_name, imgs, ws_pivots, camera, pts in self.data_loader:
            with torch.no_grad():
                if i >= count: break
                imgs = imgs.to(self._device)
                ws_pivots = ws_pivots.to(self._device)
                camera = camera.to(self._device)
                pts = pts.to(self._device)
                gen_imgs, gen_lmks = self.forward(ws_pivots, camera)
                loss += self.model.calc_loss(self.lpips_loss, self.space_regulizer, gen_imgs, imgs, gen_lmks, pts, self.G, self._use_ball_holder, ws_pivots)["loss"]
                i += len(imgs)
                div += 1
                bar.update(len(imgs))
        bar.close()
        loss /= div
        torch.distributed.all_reduce(loss)
        loss /= self.model.num_gpus
        if self._is_master and self.tf_events is not None: self.tf_events.add_scalar("metrics", loss, self._local_step)

    def train(self, run_name: str, nb_epochs: int, steps_per_batch: int, limit: int = -1, lpips_threshold: float = 0, restart_training_between_img_batch: bool = False, snap: int = 4, limit_metrics: int = 0, **kwargs):
        os.makedirs(f'{self.outdir}/{run_name}', exist_ok=True)
        self._use_ball_holder = True
        self._local_step = 0
        total_steps = nb_epochs * len(self.data_loader) * steps_per_batch
        if total_steps > limit > 0: total_steps = limit
        snap = (snap * 1000) / self.model.num_gpus
        limit = (limit * 1000) / self.model.num_gpus
        next_snap = snap
        for _ in tqdm(range(nb_epochs), unit="epoch", disable=not self._is_master):
            for img_name, imgs, ws_pivots, camera, pts in tqdm(self.data_loader, unit="batch", leave=False, disable=not self._is_master):
                if restart_training_between_img_batch: self.restart_training()
                if self.image_counter >= limit > 0:
                    self.save_snapshot(run_name, self._local_step, total_steps, final=True)
                    self.calcul_metrics(limit_metrics)
                    return self.G, self.lmkDetector

                for _ in tqdm(range(steps_per_batch), unit="step", leave=False, disable=not self._is_master):
                    gen_imgs, gen_lmks = self.train_step(imgs, ws_pivots, camera, pts, lpips_threshold)
                    if gen_imgs is None: break
                self.image_counter += len(imgs)
                if self._local_step >= next_snap:
                    self.save_snapshot(run_name, self._local_step, total_steps)
                    self.calcul_metrics(limit_metrics)
                    next_snap += snap
        self.save_snapshot(run_name, self._local_step, total_steps, final=True)
        self.calcul_metrics(limit_metrics)
        return self.G, self.lmkDetector
