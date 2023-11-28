import os

import click
import torch.nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dnnlib import EasyDict
from training.dataset import NumpyFolderDataset
from training.landmarkDetection import LandmarkDetector


@click.command()
# Required.
@click.option('--data', help='Training data', metavar='[ZIP|DIR]', type=str, required=True)
@click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--epochs', help='Number of epochs', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--lr', help='learning rate', metavar='FLOAT', type=click.FloatRange(min=1e-8), required=True)
# Optional
@click.option('--output', help='Where to save the results', metavar='DIR', default='output', show_default=True)
@click.option('--device', help='device used for training', metavar='[cuda|cpu]', type=str, default='cuda', show_default=True)
@click.option('--resume', help='resume from pth file', metavar='pth file', type=str, default=None, show_default=True)
@click.option('--reduce-lr', help='reduce learning rate during training', type=click.Choice(['std', 'exp']), default=None, show_default=True)
def main(**kwargs):
    opts = EasyDict(kwargs)
    dataset = NumpyFolderDataset(opts.data)
    dataloader = DataLoader(dataset, batch_size=opts.batch, shuffle=True, pin_memory=True)
    print("Create Network...", end='')
    lmkDetector = LandmarkDetector(105, 256, 96).to(opts.device)
    print(" Done")
    if opts.resume is not None:
        print(f"Resume from {opts.resume}...", end="")
        w = torch.load(opts.resume)
        lmkDetector.load_state_dict(w)
        print(" Done")
    try:
        import torch.utils.tensorboard as tensorboard
        tf_events = tensorboard.SummaryWriter(log_dir=opts.output)
    except ImportError:
        tf_events = None
        print("skipped : tensorboard, module not found")
    total_params = 0
    for name, param in lmkDetector.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"Layer: {name} | Parameters: {num_params}")
            total_params += num_params

    print(f"Total Trainable Parameters: {total_params}")
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(lmkDetector.parameters(), lr=opts.lr)
    pbar = tqdm(total=opts.epochs, desc='Training', unit='epochs')
    nb_epochs = 0
    lr = opts.lr
    scheduler = None
    if opts.reduce_lr == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    while nb_epochs < opts.epochs:
        for features_map, real_lmks in dataloader:
            real_lmks = real_lmks.to(opts.device).to(torch.float32)
            features_map = features_map.to(opts.device).to(torch.float32)
            lmks = lmkDetector(features_map)
            loss = criterion(1 + lmks, 1 + real_lmks)
            pbar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=f'{float(loss)}')
            if tf_events is not None:
                tf_events.add_scalar('Loss/Train', loss.item(), nb_epochs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if opts.reduce_lr == 'std' and nb_epochs % 100 == 0 and nb_epochs > 0:
                for group in optimizer.param_groups:
                    lr /= 10
                    group['lr'] = lr
            elif scheduler is not None:
                scheduler.step()
            nb_epochs += 1
            pbar.update(1)
            if nb_epochs == opts.epochs:
                break
    pbar.close()
    if tf_events is not None:
        tf_events.close()
    os.makedirs(opts.output, exist_ok=True)
    fname = f'{opts.output}/lmkDetector-{nb_epochs}'
    if opts.resume is not None:
        resume = opts.resume.split("lmkDetector-")[1].split(".pth")[0].split("-")[0]
        fname += f'-resume-{resume}'
    torch.save(lmkDetector.state_dict(), f'{fname}.pth')


if __name__ == '__main__':
    main()
