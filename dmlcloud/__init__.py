"""
Hello world
"""

__version__ = "0.3.3"

###################################
# Sub Packages
###################################

import dmlcloud.data as data
import dmlcloud.git as git
import dmlcloud.slurm as slurm

__all__ = [
    'data',
    'git',
    'slurm',
]


###################################
# Top-level API
###################################


# Pipeline

from .core.pipeline import Pipeline

__all__ += [
    'Pipeline',
]

# Stage

from .core.stage import Stage

__all__ += [
    'Stage',
]

# Callbacks

from .core.callbacks import Callback

__all__ += [
    'Callback',
]

# Distributed helpers

from .core.distributed import (
    all_gather_object,
    broadcast_object,
    deinitialize_torch_distributed,
    gather_object,
    has_environment,
    has_mpi,
    has_slurm,
    init,
    is_root,
    local_node,
    local_rank,
    local_world_size,
    rank,
    root_first,
    root_only,
    seed,
    world_size,
)

__all__ += [
    'has_slurm',
    'has_environment',
    'has_mpi',
    'is_root',
    'root_only',
    'root_first',
    'rank',
    'world_size',
    'local_rank',
    'local_world_size',
    'local_node',
    'all_gather_object',
    'gather_object',
    'broadcast_object',
    'init',
    'deinitialize_torch_distributed',
    'seed',
]

# Metrics

from .core.metrics import Tracker, TrainingHistory

__all__ += [
    Tracker,
    TrainingHistory,
]

# Logging

from .core.logging import (
    critical,
    debug,
    error,
    flush_logger,
    info,
    log,
    logger,
    print_root,
    print_worker,
    reset_logger,
    setup_logger,
    warning,
)

__all__ += [
    'logger',
    'setup_logger',
    'reset_logger',
    'flush_logger',
    'print_root',
    'print_worker',
    'log',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
]

# Model helper

from .core.model import count_parameters, scale_lr, wrap_ddp

__all__ += [
    'wrap_ddp',
    'scale_lr',
    'count_parameters',
]

# Config helper

from .core.config import factory_from_cfg, import_object, obj_from_cfg

__all__ += [
    'import_object',
    'factory_from_cfg',
    'obj_from_cfg',
]
