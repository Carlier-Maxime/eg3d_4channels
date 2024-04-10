import os
import tempfile

import torch

from torch_utils import training_stats


def launch_multiprocessing(subprocess_fn, args, use_idr_torch: bool = False):
    if use_idr_torch:
        import idr_torch
        args.num_gpus = idr_torch.size
        subprocess_fn(rank=idr_torch.rank, local_rank=idr_torch.local_rank, args=args, temp_dir=None)
    else:
        torch.multiprocessing.set_start_method('spawn')
        with tempfile.TemporaryDirectory() as temp_dir:
            if args.num_gpus == 1:
                subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
            else:
                torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)


def init_distributed(rank: int, temp_dir, args):
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init')) if temp_dir is not None else None
        if os.name == 'nt':
            init_method = ('file:///' + init_file.replace('\\', '/')) if init_file is not None else 'env:///'
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}' if init_file is not None else 'env://'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    torch.distributed.barrier()
