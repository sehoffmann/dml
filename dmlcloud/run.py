"""
usage: dmlrun [-h] [--gpus GPUS] [--nprocs NPROCS] script ...

dmlrun is a thin wrapper around torch.distributed.launch that provides a more user-friendly interface.

While torchrun is a powerful tool, it can be a bit clunky to use for testing and debugging. dmlrun aims to make it easier to launch distributed training jobs on a single node.For serious mulit-node training, we recommend using srun or torchrun directly.

positional arguments:
  script                Path to the script to run.
  args                  Arguments to pass to the script.

options:
  -h, --help            show this help message and exit
  --gpus GPUS, -g GPUS  Comma-seperated list of GPU IDs to use for training. Overrides CUDA_VISIBLE_DEVICES.
  --nprocs NPROCS, -n NPROCS
                        Number of GPUs to use for training.

Example:
    dmlrun --gpus 3,7 train.py
    dmlrun --num-gpus 2 train.py --batch-size 64
"""

import argparse
import os


def main():
    description = (
        'dmlrun is a thin wrapper around torch.distributed.launch that provides a more user-friendly interface.\n\n'
        'While torchrun is a powerful tool, it can be a bit clunky to use for testing and debugging. dmlrun aims to make it easier to launch distributed training jobs on a single node.'
        'For serious mulit-node training, we recommend using srun or torchrun directly.'
    )
    epilog = 'Example:\n' '    dmlrun --gpus 3,7 train.py\n' '    dmlrun --num-gpus 2 train.py --batch-size 64'
    parser = argparse.ArgumentParser(
        prog='dmlrun', description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--gpus', '-g', help='Comma-seperated list of GPU IDs to use for training. Overrides CUDA_VISIBLE_DEVICES.'
    )
    parser.add_argument('--nprocs', '-n', type=int, help='Number of GPUs to use for training.')
    parser.add_argument('script', type=str, help='Path to the script to run.')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments to pass to the script.')

    args = parser.parse_args()

    if args.gpus and args.num_gpus:
        raise ValueError('Only one of --gpus or --num-gpus can be specified.')

    if args.gpus:
        ids = args.gpus.split(',')
        if not all(id.isdigit() for id in ids):
            raise ValueError('GPU IDs must be integers.')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        nprocs = len(ids)
    elif args.nprocs:
        nprocs = args.nprocs
    else:
        nprocs = 1

    import torch.distributed.run

    cmdline = [
        '--standalone',
        '--nproc_per_node',
        f'{nprocs}',
    ]

    cmdline += [args.script] + args.args
    print('Executing: torchrun', ' '.join(cmdline), flush=True)
    torch.distributed.run.main(cmdline)


if __name__ == '__main__':
    main()
