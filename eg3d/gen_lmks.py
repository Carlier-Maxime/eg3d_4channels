import click
import numpy as np
import torch

from dnnlib import EasyDict
from training.landmarkDetection import LandmarkDetector


@click.command()
# Required
@click.option('--network', help='a network used for generate landmarks', type=str, metavar='pth file', required=True)
@click.option('--features', help='a features map', type=str, metavar='npy', required=True)
# Optional
@click.option('--out', help='a output folder', type=str, metavar='DIR', default='output', show_default=True)
@click.option('--device', help='a device used', type=str, metavar='[cuda|cpu]', default='cuda', show_default=True)
def main(**kwargs):
    opts = EasyDict(kwargs)
    features = torch.from_numpy(np.load(opts.features)).to(opts.device)
    print("Load Network...", end='')
    lmkDetector = LandmarkDetector(105, 256, 96).to(opts.device)
    w = torch.load(opts.network)
    lmkDetector.load_state_dict(w)
    print(" Done")
    print("Generate results...", end='')
    with torch.no_grad():
        lmks = lmkDetector(features[None])[0]
    np.save(f'{opts.out}/lmks.npy', lmks.cpu().numpy())
    print(" Done")


if __name__ == '__main__':
    main()
