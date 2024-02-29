from random import choice
from string import ascii_uppercase

import click
from torch.utils.data import DataLoader

import dnnlib
from training.coaches.simple_coach import SimpleCoach
from training.dataset import ImageAndNumpyFolderDataset


@click.command()
@click.option('--network', type=str, required=True, help='Network pkl file')
@click.option('--dataset', type=str, required=True, help='Dataset path : the dataset content images and latents')
@click.option('--network-lmks', type=str, default=None, help='Network pkl file for landmark detector')
@click.option('--outdir', type=str, default="output", help='The output directory')
@click.option('--force-rgb', type=bool, default=False, is_flag=True, help='force RGB images')
@click.option('--run_name', type=str, default=''.join(choice(ascii_uppercase) for i in range(12)), help='run name, is used for saving results')
@click.option('--reset-between-batch', type=bool, default=False, is_flag=True, help='use reset between batch for a specific network per batch')
@click.option('--snap', type=int, default=4, help='the number of thousand step between saving a snapshot')
@click.option('--batch', type=int, default=1, help='batch size')
@click.option('--device', type=str, default='cuda', help='the device used for Pivotal Tuning')
@click.option('--epochs', type=int, default=1, help='number of epochs')
@click.option('--steps-batch', type=int, default=350, help='the number of PTI steps per batch')
@click.option('--limit', type=int, default=-1, help='the maximum number of kimg used')
@click.option('--lpips-threshold', type=float, default=0, help='lpips threshold for stop PTI')
@click.option('--lr', type=float, default=3e-4, help='learning rate')
def run_PTI(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    print("Run Name : "+opts.run_name)
    dataset = ImageAndNumpyFolderDataset(opts.dataset, force_rgb=opts.force_rgb, use_labels=True, require_pts=opts.network_lmks is not None)
    dataloader = DataLoader(dataset, batch_size=opts.batch, shuffle=False)
    coach_args = {
        "network_pkl": opts.network,
        "data_loader": dataloader,
        "device": opts.device,
        "lr": opts.lr,
        "outdir": opts.outdir,
        "network_lmks": opts.network_lmks
    }
    coach = SimpleCoach(**coach_args)
    coach.train(opts.run_name, opts.epochs, opts.steps_batch, limit=opts.limit, snap=opts.snap, restart_training_between_img_batch=opts.reset_between_batch)
    return opts.run_name


if __name__ == '__main__':
    run_PTI()
