from random import choice
from string import ascii_uppercase

import click
from torch.utils.data import DataLoader

import dnnlib
from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
from training.dataset import ImageAndNumpyDataset


@click.command()
@click.option('--network', type=str, required=True, help='Network pkl file')
@click.option('--dataset', type=str, required=True, help='Dataset path : the dataset content images and latents')
@click.option('--outdir', type=str, default="output", help='The output directory')
@click.option('--force-rgb', type=bool, default=False, is_flag=True, help='force RGB images')
@click.option('--run_name', type=str, default=''.join(choice(ascii_uppercase) for i in range(12)), help='run name, is used for saving results')
@click.option('--use-multi-id', type=bool, default=False, is_flag=True, help='use multi id training')
@click.option('--batch', type=int, default=1, help='batch size')
@click.option('--device', type=str, default='cuda', help='the device used for Pivotal Tuning')
@click.option('--limit', type=int, default=-1, help='the maximum number of images to used')
@click.option('--num_steps', type=int, default=350, help='the number of PTI steps')
@click.option('--lpips-threshold', type=float, default=0, help='lpips threshold for stop PTI')
@click.option('--lr', type=float, default=3e-4, help='learning rate')
def run_PTI(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    dataset = ImageAndNumpyDataset(opts.dataset, force_rgb=opts.force_rgb)
    dataloader = DataLoader(dataset, batch_size=opts.batch, shuffle=False)
    coach = MultiIDCoach(opts.network, dataloader, opts.device, opts.lr, outdir=opts.outdir) if opts.use_multi_id else SingleIDCoach(opts.network, dataloader, opts.device, opts.lr, outdir=opts.outdir)
    coach.train(opts.run_name, opts.num_steps, limit=opts.limit)
    return opts.run_name


if __name__ == '__main__':
    run_PTI()
