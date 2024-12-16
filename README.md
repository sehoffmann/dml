![Dmlcloud Logo](./misc/logo/dmlcloud_color.png)
---------------
[![PyPI Status](https://img.shields.io/pypi/v/dmlcloud)](https://pypi.org/project/dmlcloud/)
[![Documentation Status](https://readthedocs.org/projects/dmlcloud/badge/?version=latest)](https://dmlcloud.readthedocs.io/en/latest/?badge=latest)
[![Test Status](https://img.shields.io/github/actions/workflow/status/sehoffmann/dmlcloud/run_tests.yml?label=tests&logo=github)](https://github.com/sehoffmann/dmlcloud/actions/workflows/run_tests.yml)

A torch library for easy distributed deep learning on HPC clusters. Supports both slurm and MPI. No unnecessary abstractions and overhead. Simple, yet powerful, API.

## Highlights
- Simple, yet powerful, API
- Easy initialization of `torch.distributed`
- Distributed checkpointing and metrics
- Extensive logging and diagnostics
- Wandb support
- A wealth of useful utility functions

## Installation
dmlcloud can be installed directly from PyPI:
```
pip install dmlcloud
```

Alternatively, you can install the latest development version directly from Github:
```
pip install git+https://github.com/tangentlabs/django-oscar-paypal.git@issue/34/oscar-0.6
```

## Minimal Example
See [examples/barebone_mnist.py](https://github.com/sehoffmann/dmlcloud/blob/develop/examples/barebone_mnist.py) for a minimal and barebone example on how to distributely train MNIST.
To run it on a single node with 4 GPUs, use
```
dmlrun -n 4 examples/barebone_mnist.py
```

`dmlrun` is a thin wrapper around `torchrun` that makes development work on a single node easier.


To run your training across multiple nodes on a slurm cluster instead, you can simply use `srun`:
```
srun --ntasks-per-node [NUM_GPUS] python examples/barebone_mnist.py
```

## Documentation

You can find the official documentation at [Read the Docs](https://dmlcloud.readthedocs.io/en/latest/)
