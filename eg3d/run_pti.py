from random import choice
from string import ascii_uppercase

import click
from torch.utils.data import DataLoader

import dnnlib
from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
from training.dataset import ImageAndNumpyFolderDataset


@click.command()
@click.option('--network', type=str, required=True, help='Network pkl file')
@click.option('--dataset', type=str, required=True, help='Dataset path : the dataset content images and latents')
@click.option('--network-lmks', type=str, default=None, help='Network pkl file for landmark detector')
@click.option('--outdir', type=str, default="output", help='The output directory')
@click.option('--force-rgb', type=bool, default=False, is_flag=True, help='force RGB images')
@click.option('--run_name', type=str, default=''.join(choice(ascii_uppercase) for i in range(12)), help='run name, is used for saving results')
@click.option('--use-multi-id', type=bool, default=False, is_flag=True, help='use multi id training')
@click.option('--snapshot-step', type=int, default=100, help='the number of steps between saving a snapshot only for multi id')
@click.option('--batch', type=int, default=1, help='batch size')
@click.option('--device', type=str, default='cuda', help='the device used for Pivotal Tuning')
@click.option('--limit', type=int, default=-1, help='the maximum number of images to used')
@click.option('--num_steps', type=int, default=350, help='the number of PTI steps')
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
    coach = MultiIDCoach(**coach_args) if opts.use_multi_id else SingleIDCoach(**coach_args)
    coach.train(opts.run_name, opts.num_steps, limit=opts.limit, snapshot_step=opts.snapshot_step)
    return opts.run_name


if __name__ == '__main__':
    run_PTI()
