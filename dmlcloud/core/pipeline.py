import warnings
from datetime import datetime, timedelta
from functools import cached_property
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from . import logging as dml_logging
from .callbacks import (
    Callback,
    CallbackList,
    CbPriority,
    CheckpointCallback,
    CsvCallback,
    CudaCallback,
    DiagnosticsCallback,
    GitDiffCallback,
    TensorboardCallback,
    WandbCallback,
)
from .checkpoint import CheckpointDir, find_slurm_checkpoint, generate_checkpoint_path
from .distributed import broadcast_object, init, is_root, local_rank
from .stage import Stage


__all__ = [
    'Pipeline',
]


class _RunGuard:
    """
    Context manager that ensures that the pipeline is properly cleaned up in case of an exception or interruption.
    """

    def __init__(self, pipe):
        self.pipe = pipe

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        suppress_exception = False
        if exc_type is KeyboardInterrupt:
            dml_logging.info('------- Training interrupted by user -------')
            suppress_exception = True
        elif exc_type is not None:
            dml_logging.error(
                '------- Training failed with an exception -------', exc_info=(exc_type, exc_value, traceback)
            )

        callbacks = []
        if self.pipe.current_stage is not None:
            callbacks += self.pipe.current_stage.callbacks
        callbacks += self.pipe.callbacks

        for callback in reversed(callbacks):
            callback.cleanup(self.pipe, exc_type, exc_value, traceback)

        return suppress_exception


class _ForwardCallback(Callback):
    """
    Invokes the pre_run, post_run methods of the Pipeline.
    Stage-specific callbacks are managed by the Stage object.
    """

    def pre_run(self, pipe):
        pipe.pre_run()

    def post_run(self, pipe):
        pipe.post_run()


class Pipeline:
    """
    A training pipeline that consists of multiple stages.

    This is the main entry point for training with dmlcloud. The pipeline manages the training process and
    orchestrates the execution of multiple stages. It also provides a way to add callbacks that are executed at
    different points during the training process.

    Use the `append()` method to add stages to the pipeline and `add_callback()` to add callbacks.

    Checkpointing can be enabled with `enable_checkpointing()` and Weights & Biases integration with `enable_wandb()`.

    Once the pipeline is set up, call `run()` to start the training process.
    """

    def __init__(self, config: Optional[Union[OmegaConf, Dict]] = None, name: Optional[str] = None):
        # Auto-init torch.distributed if not already initialized
        if not dist.is_initialized():
            init()

        if config is None:
            self.config = OmegaConf.create()
        elif not isinstance(config, OmegaConf):
            self.config = OmegaConf.create(config)
        else:
            self.config = config

        self.name = name

        self.checkpoint_dir = None
        self.resumed = None
        self.start_time = None
        self.stop_time = None
        self.current_stage = None

        self.wandb = False

        self.stages = []
        self.callbacks = CallbackList()

        self.add_callback(DiagnosticsCallback(), CbPriority.DIAGNOSTICS)
        self.add_callback(GitDiffCallback(), CbPriority.GIT)
        self.add_callback(_ForwardCallback(), CbPriority.OBJECT_METHODS)  # methods have priority 0

        if dist.is_gloo_available():
            self.gloo_group = dist.new_group(backend='gloo')
        else:
            warnings.warn('Gloo backend not available. Barriers will not use custom timeouts.')

    @property
    def checkpointing_enabled(self):
        return self.checkpoint_dir is not None

    def add_callback(self, callback: Callback, priority: int = 1):
        """
        Adds a callback to this pipeline.

        Callbacks added to the pipeline and not to individual stages are executed for all stages in the pipeline.
        Callbacks are executed based on their priority, with lower values being executed first.
        Callbacks with the same priority are executed in the order they were added.

        Methods of the stage and pipeline objects, e.g. pre_run(), have priority 0.

        Args:
            callback (StageCallback): The callback to add.
            priority (int, optional): The priority of the callback. Defaults to 1.
        """
        self.callbacks.append(callback, priority)

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

        if is_root():
            self.add_callback(CheckpointCallback(self.checkpoint_dir.path), CbPriority.CHECKPOINT)
            self.add_callback(CsvCallback(self.checkpoint_dir.path, append_stage_name=True), CbPriority.CSV)
            self.add_callback(TensorboardCallback(self.checkpoint_dir.path), CbPriority.TENSORBOARD)

        if self.device.type == 'cuda':
            self.add_callback(CudaCallback(), CbPriority.CUDA)

    def enable_wandb(
        self,
        project: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        tags: List[str] | None = None,
        startup_timeout: int = 360,
        **kwargs,
    ):
        if self.wandb:
            raise ValueError('Wandb already enabled')

        import wandb  # import now to avoid potential long import times later on  # noqa

        if is_root():
            project = project or self.name
            self.add_callback(WandbCallback(project, entity, group, tags, startup_timeout, **kwargs), CbPriority.WANDB)

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
        if len(self.stages) == 0:
            raise ValueError('No stages defined. Use append() to add stages to the pipeline.')

        # make sure everything is set up before starting the run
        # important to prevent checkpoint dir creation before all processes searched for it
        self.barrier(timeout=10 * 60)

        with _RunGuard(self):
            self._pre_run()
            for stage in self.stages:
                self.current_stage = stage
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
        self.start_time = datetime.now()

        if self.resumed:
            self._resume_run()

        for callback in self.callbacks:
            callback.pre_run(self)

    def _resume_run(self):
        dml_logging.info(f'Resuming training from checkpoint: {self.checkpoint_dir}')
        self.resume_run()

    def _post_run(self):
        self.stop_time = datetime.now()

        for callback in self.callbacks:
            callback.post_run(self)
