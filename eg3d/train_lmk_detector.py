import copy
import os

import click
import torch.nn
import pickle
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dnnlib import EasyDict
from training.dataset import NumpyFolderDataset
from training.landmarkDetection import LandmarkDetector, LandmarkDetectorExperience


def createLmkDetector(opts):
    args = [opts.nb_pts, opts.features_res, opts.channels, opts.pts_dim]
    if opts.detector_type == 'exp':
        lmkDetector = LandmarkDetectorExperience(*args)
    else:
        lmkDetector = LandmarkDetector(*args)
    return lmkDetector.to(opts.device)


@click.command()
# Required.
@click.option('--data', help='Training data', metavar='[ZIP|DIR]', type=str, required=True)
@click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--kimg', help='Number of kimg', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--lr', help='learning rate', metavar='FLOAT', type=click.FloatRange(min=1e-8), required=True)
# Optional
@click.option('--output', help='Where to save the results', metavar='DIR', default='output', show_default=True)
@click.option('--run-dir', help='Name of directory for save this training', metavar='DIR', default=None, show_default=True)
@click.option('--device', help='device used for training', metavar='[cuda|cpu]', type=str, default='cuda', show_default=True)
@click.option('--resume', help='resume from pth or pkl file', metavar='[pth|pkl]', type=str, default=None, show_default=True)
@click.option('--reduce-lr', help='reduce learning rate during training', type=click.Choice(['std', 'exp']), default=None, show_default=True)
@click.option('--to-pth', help='save a network to pth file', type=bool, is_flag=True, default=False, show_default=True)
@click.option('--no-tensorboard', help='disable tensorboard', type=bool, is_flag=True, default=False, show_default=True)
@click.option('--detector-type', help='type of landmark detector used', type=click.Choice(['std', 'exp']), default='std', show_default=True)
@click.option('--nb-pts', help='Number of points', metavar='INT', type=click.IntRange(min=1), default=105, show_default=True)
@click.option('--pts-dim', help='Dimension of points', metavar='INT', type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--features-res', help='Features Resolution', metavar='INT', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--channels', help='Features Channels', metavar='INT', type=click.IntRange(min=1), default=96, show_default=True)
@click.option('--eg3d-network', help='Network EG3D for generate features from a mapped latents provided by a dataset', metavar='PKL', type=str, default=None, show_default=True)
@click.option('--snap', help='Snapshot interval', metavar='KIMG', type=click.IntRange(min=1), default=32, show_default=True)
def main(**kwargs):
    opts = EasyDict(kwargs)
    if opts.detector_type == 'std': assert opts.batch % 4 == 0, 'batch must be a multiple of 4'
    dataset = NumpyFolderDataset(opts.data)
    dataloader = DataLoader(dataset, batch_size=opts.batch, shuffle=True, pin_memory=True)
    if opts.resume is not None:
        print(f"Resume from {opts.resume}...", end="")
        if opts.resume.endswith('.pth'):
            lmkDetector = createLmkDetector(opts)
            w = torch.load(opts.resume)
            lmkDetector.load_state_dict(w)
        elif opts.resume.endswith('.pt'):
            lmkDetector = torch.load(opts.resume)
        else:
            with open(opts.resume, 'rb') as f:
                lmkDetector = pickle.Unpickler(f).load().to(opts.device).requires_grad_(True)
        print(" Done")
    else:
        print("Create Network...", end='')
        lmkDetector = createLmkDetector(opts)
        print(" Done")

    if opts.run_dir is None:
        opts.run_dir = f'{opts.kimg}k_{opts.detector_type}_{opts.features_res}²:{opts.channels}>{opts.nb_pts}_lr{opts.lr:.1e}:{opts.reduce_lr}_batch{opts.batch}'
    opts.output += f'/{opts.run_dir}'

    tf_events = None
    try:
        import torch.utils.tensorboard as tensorboard
    except ImportError:
        tensorboard = None
        print("skipped : tensorboard, module not found")
    if not opts.no_tensorboard and tensorboard is not None:
        tf_events = tensorboard.SummaryWriter(log_dir=opts.output)
    total_params = 0
    for name, param in lmkDetector.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"Layer: {name} | Parameters: {num_params}")
            total_params += num_params

    print(f"Total Trainable Parameters: {total_params}")
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(lmkDetector.parameters(), lr=opts.lr)
    pbar = tqdm(total=opts.kimg * 1000, desc='Training', unit='imgs')
    nb_imgs = 0
    lr = opts.lr
    scheduler = None
    eg3d_network = None
    if opts.eg3d_network is not None:
        if opts.eg3d_network.endswith(".pt"):
            eg3d_network = torch.load(opts.eg3d_network)
        else:
            with open(opts.eg3d_network, 'rb') as f:
                eg3d_network = pickle.Unpickler(f).load()['G'].to(opts.device).requires_grad_(True)
    if opts.reduce_lr == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    def gen_loss(features_map, real_lmks):
        real_lmks = real_lmks.to(opts.device).to(torch.float32)
        features_map = features_map.to(opts.device)
        if eg3d_network is not None:
            features_map = eg3d_network.backbone.synthesis(features_map)
        features_map = features_map.to(torch.float32)
        batch_size = real_lmks.shape[0]
        if batch_size % 4 != 0 and opts.detector_type == 'std':
            real_lmks = real_lmks.repeat(2, 1, 1)[:batch_size + batch_size % 4]
            features_map = features_map.repeat(2, 1, 1, 1)[:batch_size + batch_size % 4]
        lmks = lmkDetector(features_map)
        return criterion(lmks, real_lmks)

    os.makedirs(opts.output, exist_ok=True)

    def save_snap():
        fname = f'{opts.output}/lmkDetector-{nb_imgs // 1000:06d}'
        if opts.resume is not None:
            tmp = opts.resume.split("lmkDetector-")
            if len(tmp) > 1:
                resume = tmp[1].split(".pth")[0].split("-")[0]
                fname += f'-resume-{resume}'
        if opts.to_pth:
            torch.save(lmkDetector.state_dict(), f'{fname}.pth')
        else:
            with open(f"{fname}.pkl", 'wb') as f:
                pickle.dump(copy.deepcopy(lmkDetector).eval().requires_grad_(False).cpu(), f)

    snap_index = 0
    while nb_imgs < opts.kimg * 1000:
        for features_map, real_lmks in dataloader:
            if nb_imgs >= (snap_index * opts.snap * 1000):
                save_snap()
                snap_index += 1
            loss = gen_loss(features_map, real_lmks)
            pbar.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.3e}", loss=f'{float(loss):.3e}')
            if tf_events is not None:
                tf_events.add_scalar('Loss/Train', loss.item(), nb_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nb_imgs += real_lmks.shape[0]
            pbar.update(real_lmks.shape[0])
            if opts.reduce_lr == 'std' and nb_imgs % 1000 == 0:
                for group in optimizer.param_groups:
                    lr /= 10
                    group['lr'] = lr
            elif scheduler is not None:
                scheduler.step()
            if nb_imgs >= opts.kimg * 1000:
                break
    if tf_events is not None:
        tf_events.add_scalar('Loss/Train', gen_loss(*next(iter(dataloader))).item(), nb_imgs)
        tf_events.close()
    save_snap()


if __name__ == '__main__':
    main()
