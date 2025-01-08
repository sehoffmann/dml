![Dmlcloud Logo](./misc/logo/dmlcloud_color.png)
---------------
[![PyPI Status](https://img.shields.io/pypi/v/dmlcloud)](https://pypi.org/project/dmlcloud/)
[![Documentation Status](https://readthedocs.org/projects/dmlcloud/badge/?version=latest)](https://dmlcloud.readthedocs.io/en/latest/?badge=latest)
[![Test Status](https://img.shields.io/github/actions/workflow/status/sehoffmann/dmlcloud/run_tests.yml?label=tests&logo=github)](https://github.com/sehoffmann/dmlcloud/actions/workflows/run_tests.yml)

A torch library for easy distributed deep learning on HPC clusters. Supports both slurm and MPI. No unnecessary abstractions and overhead. Simple, yet powerful, API.

## Highlights
- Simple, yet powerful, API
- Easy initialization of `torch.distributed`
- Distributed metrics
- Extensive logging and diagnostics
- Wandb support
- Tensorboard support
- A wealth of useful utility functions

## Installation
dmlcloud can be installed directly from PyPI:
```bash
pip install dmlcloud
```

Alternatively, you can install the latest development version directly from Github:
```bash
pip install git+https://github.com/sehoffmann/dmlcloud.git
```

### Documentation

You can find the official documentation at [Read the Docs](https://dmlcloud.readthedocs.io/en/latest/)

## Minimal Example
See [examples/mnist.py](https://github.com/sehoffmann/dmlcloud/blob/develop/examples/mnist.py) for a minimal example on how to train MNIST with multiple GPUS. To run it with 4 GPUs, use
```bash
dmlrun -n 4 python examples/mnist.py
```
`dmlrun` is a thin wrapper around `torchrun` that makes it easier to prototype on a single node.

## Slurm Support
*dmlcloud* automatically looks for slurm environment variables to initialize torch.distributed. On a slurm cluster, you can hence simply use `srun` from within an sbatch script to train on multiple nodes:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpu-bind=none

srun python examples/mnist.py
```

## FAQ

### How is dmlcloud different from similar libraries like *pytorch lightning* or *fastai*?

dmlcloud was designed foremost with one underlying principle:
> **No unnecessary abstractions, just help with distributed training**

As a consequence, dmlcloud code is almost identical to a regular pytorch training loop and only requires a few adjustments here and there.
In contrast, other libraries often introduce extensive API's that can quickly feel overwhelming due to their sheer amount of options.

For instance, **the constructor of `ligthning.Trainer` has 51 arguments! `dml.Pipeline` only has 2.**
