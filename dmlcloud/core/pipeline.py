import warnings
from datetime import datetime, timedelta
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from dmlcloud.util.wandb import wandb, wandb_is_initialized, wandb_set_startup_timeout
from ..util.logging import experiment_header, general_diagnostics, IORedirector
from . import logging as dml_logging
from .callbacks import CsvCallback
from .checkpoint import CheckpointDir, find_slurm_checkpoint, generate_checkpoint_path
from .distributed import all_gather_object, broadcast_object, init, local_rank, root_only
from .stage import Stage


__all__ = [
    'Pipeline',
]


class Pipeline:
    def __init__(self, config: Optional[Union[OmegaConf, Dict]] = None, name: Optional[str] = None):
        if config is None:
            self.config = OmegaConf.create()
        elif not isinstance(config, OmegaConf):
            self.config = OmegaConf.create(config)
        else:
            self.config = config

        # Auto-init distributed if not already initialized
        if not dist.is_initialized():
            init()

        self.name = name

        self.checkpoint_dir = None
        self.gloo_group = None
        self.io_redirector = None
        self.resumed = None
        self.start_time = None
        self.stop_time = None
        self.current_stage = None

        self.wandb = False
        self._wandb_initalizer = None

        self.stages = []
        self.datasets = {}
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}

    @property
    def checkpointing_enabled(self):
        return self.checkpoint_dir is not None

    def register_optimizer(self, name: str, optimizer, scheduler=None):
        if name in self.optimizers:
            raise ValueError(f'Optimizer with name {name} already exists')
        self.optimizers[name] = optimizer
        if scheduler is not None:
            self.schedulers[name] = scheduler

    def register_dataset(self, name: str, dataset: Union[DataLoader, Dataset, Sequence], verbose: bool = True):
        if name in self.datasets:
            raise ValueError(f'Dataset with name {name} already exists')

        self.datasets[name] = dataset
        if verbose:
            msg = f'Dataset "{name}":\n'
            try:
                length = len(dataset)
                msg += f'    - Batches (Total): ~{length * dist.get_world_size()}\n'
                msg += f'    - Batches (/Worker): {length}\n'
            except TypeError:  # __len__ not implemented
                msg += '    - Batches (Total): N/A\n'
                msg += '    - Batches (/Worker): N/A\n'
            dml_logging.info(msg)

    def append(self, stage: Stage):
        if not isinstance(stage, Stage):
            raise ValueError('stage must be a Stage object')

        stage.pipe = self
        self.stages.append(stage)

    def enable_checkpointing(
        self,
        root: str,
        resume: bool = False,
    ):
        if self.checkpointing_enabled:
            raise ValueError('Checkpointing already enabled')

        path = None
        if resume and CheckpointDir(root).is_valid:
            path = root
            self.resumed = True
        elif resume and find_slurm_checkpoint(root):
            path = find_slurm_checkpoint(root)
            self.resumed = True

        if path is None:  # no need for a barrier here, dir creation happens in _pre_run()
            path = generate_checkpoint_path(root=root, name=self.name, creation_time=self.start_time)
            path = broadcast_object(path)
            self.resumed = False

        self.checkpoint_dir = CheckpointDir(path)

    def enable_wandb(
        self,
        project: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        tags: List[str] | None = None,
        startup_timeout: int = 360,
        **kwargs,
    ):
        import wandb  # import now to avoid potential long import times later on

        @root_only
        def initializer():
            wandb_set_startup_timeout(startup_timeout)
            wandb.init(
                config=OmegaConf.to_container(self.config, resolve=True),
                name=self.name,
                entity=entity,
                project=project if project else self.name,
                group=group,
                tags=tags,
                **kwargs,
            )

        self._wandb_initalizer = initializer
        self.wandb = True

    def barrier(self, timeout=None):
        if self.gloo_group is None:
            dist.barrier()
        else:
            timeout = timedelta(seconds=timeout) if timeout is not None else None
            dist.monitored_barrier(self.gloo_group, timeout=timeout, wait_all_ranks=True)

    def run(self):
        """
        Starts the training and runs all registered stages.
        """
        with _RunGuard(self):
            self._pre_run()
            for stage in self.stages:
                stage.run()
            self._post_run()

    def pre_run(self):
        pass

    def post_run(self):
        pass

    def resume_run(self):
        pass

    @cached_property
    def device(self):
        if torch.cuda.is_available():
            if local_rank() is None:
                warnings.warn(
                    'CUDA is available but no local rank found. Make sure to set CUDA_VISIBLE_DEVICES manually for each rank.'
                )
                return torch.device('cuda')
            else:
                return torch.device('cuda', local_rank())
        else:
            warnings.warn('CUDA is not available. Running on CPU.')
            return torch.device('cpu')

    def _pre_run(self):
        if len(self.stages) == 0:
            raise ValueError('No stages defined. Use append_stage() to add stages to the pipeline.')

        if dist.is_gloo_available():
            self.gloo_group = dist.new_group(backend='gloo')
        else:
            warnings.warn('Gloo backend not available. Barriers will not use custom timeouts.')

        self.barrier(
            timeout=10 * 60
        )  # important to prevent checkpoint dir creation before all processes searched for it
        if self.checkpointing_enabled:
            self._init_checkpointing()

        if self.wandb:
            self._wandb_initalizer()

        self.barrier(timeout=10 * 60)  # make sure everything is set up before starting the run
        self.start_time = datetime.now()

        header = '\n' + experiment_header(self.name, self.checkpoint_dir, self.start_time)
        dml_logging.info(header)

        if self.resumed:
            self._resume_run()

        diagnostics = general_diagnostics()

        diagnostics += '\n* DEVICES:\n'
        devices = all_gather_object(str(self.device))
        diagnostics += '\n'.join(f'    - [Rank {i}] {device}' for i, device in enumerate(devices))

        diagnostics += '\n* CONFIG:\n'
        diagnostics += '\n'.join(f'    {line}' for line in OmegaConf.to_yaml(self.config, resolve=True).splitlines())

        dml_logging.info(diagnostics)

        self.pre_run()

    @root_only
    def _init_checkpointing(self):
        if not self.checkpoint_dir.is_valid:
            self.checkpoint_dir.create()
            self.checkpoint_dir.save_config(self.config)
        self.io_redirector = IORedirector(self.checkpoint_dir.log_file)
        self.io_redirector.install()

        for stage in self.stages:
            stage.add_callback(CsvCallback(self.checkpoint_dir.path / f'metrics_{stage.name}.csv'))

    def _resume_run(self):
        dml_logging.info(f'Resuming training from checkpoint: {self.checkpoint_dir}')
        self.resume_run()

    def _post_run(self):
        self.stop_time = datetime.now()
        dml_logging.info(f'Finished training in {self.stop_time - self.start_time} ({self.stop_time})')
        if self.checkpointing_enabled:
            dml_logging.info(f'Outputs have been saved to {self.checkpoint_dir}')
        self.post_run()

    def _cleanup(self, exc_type, exc_value, traceback):
        """
        Called by _RunGuard to ensure that the pipeline is properly cleaned up
        """
        if exc_type is KeyboardInterrupt:
            dml_logging.info('------- Training interrupted by user -------')
        elif exc_type is not None:
            dml_logging.error(
                '------- Training failed with an exception -------', exc_info=(exc_type, exc_value, traceback)
            )

        if self.wandb and wandb_is_initialized():
            wandb.finish(exit_code=0 if exc_type is None else 1)

        if self.io_redirector is not None:
            self.io_redirector.uninstall()

        return False


class _RunGuard:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        return self.pipeline._cleanup(exc_type, exc_value, traceback)
