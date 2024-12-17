import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from progress_table import ProgressTable

from ..util.logging import DevNullIO
from . import logging as dml_logging
from .distributed import is_root
from .metrics import MetricTracker, Reduction

__all__ = [
    'Stage',
    'TrainValStage',
]


class Stage:
    """
    Hook Points:
        - pre_stage()
        - post_stage()
        - pre_epoch()
        - post_epoch()
    """

    def __init__(self):
        self.pipeline = None  # set by the pipeline
        self.max_epochs = None  # set by the pipeline
        self.name = None  # set by the pipeline

        self.start_time = None
        self.stop_time = None
        self.epoch_start_time = None
        self.epoch_stop_time = None
        self.current_epoch = 1
        self._stop_requested = False

        self.metric_prefix = None
        self.table = None
        self.barrier_timeout = None

    @property
    def tracker(self) -> MetricTracker:
        return self.pipeline.tracker

    @property
    def device(self):
        return self.pipeline.device

    @property
    def config(self):
        return self.pipeline.config

    def track_reduce(
        self,
        name: str,
        value: torch.Tensor,
        step: Optional[int] = None,
        reduction: Reduction = Reduction.MEAN,
        dim: Optional[List[int]] = None,
        reduce_globally: bool = True,
        prefixed: bool = True,
    ):
        if prefixed and self.metric_prefix:
            name = f'{self.metric_prefix}/{name}'
        self.pipeline.track_reduce(name, value, step, reduction, dim, reduce_globally)

    def track(self, name: str, value, step: Optional[int] = None, prefixed: bool = True):
        if prefixed and self.metric_prefix:
            name = f'{self.metric_prefix}/{name}'
        self.pipeline.track(name, value, step)

    def stop_stage(self):
        self._stop_requested = True

    def pre_stage(self):
        """
        Executed before the stage starts.
        Use this method to setup aby stage-specific data sets or models.
        """
        pass

    def post_stage(self):
        """
        Executed after the stage finishes.
        Use this method to clean up any stage-specific resources or to save any intermediate results/artifacts.
        """
        pass

    def pre_epoch(self):
        """
        Executed before each epoch.
        """
        pass

    def post_epoch(self):
        """
        Executed after each epoch and after the metrics have been reduced.
        """
        pass

    def run_epoch(self):
        """
        Train the model for one epoch. Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def table_columns(self) -> List[Union[str, Dict[str, Any]]]:
        """
        Override this method to customize the metrics displayed in the progress table.

        Should return a list containing either strings or dicts.
        If a string, it will be used as both the display name and the metric name.
        If a dict, it should contain a 'name' key and a 'metric' key.
        The 'name' key will be used as the display name, and the 'metric' key will be used as the metric name.
        Additional keys are forwarded to the ProgressTable.add_column method.
        If 'metric' is None, then the user is responsible for updating the column manually.
        """
        columns = [
            {'name': 'Epoch', 'metric': 'misc/epoch'},
            {'name': 'Time/Epoch', 'metric': None},
        ]
        if self.max_epochs is not None:
            columns.append({'name': 'ETA', 'metric': None})
        return columns

    def run(self):
        """
        Runs this stage. Either until max_epochs are reached, or until stop_stage() is called.
        """
        self._pre_stage()
        while self.max_epochs is None or self.current_epoch <= self.max_epochs:
            self._pre_epoch()
            self.run_epoch()
            self._post_epoch()
            if self._stop_requested:
                break
        self._post_stage()

    def _pre_stage(self):
        self.start_time = datetime.now()
        self.table = ProgressTable(file=sys.stdout if is_root() else DevNullIO())
        self._setup_table()
        if len(self.pipeline.stages) > 1:
            dml_logging.info(f'\n========== STAGE: {self.name} ==========')

        self.pre_stage()

        dml_logging.flush_logger()

        self.pipeline.barrier(self.barrier_timeout)

    def _post_stage(self):
        self.table.close()
        self.post_stage()
        self.pipeline.barrier(self.barrier_timeout)
        self.stop_time = datetime.now()
        if len(self.pipeline.stages) > 1:
            dml_logging.info(f'Finished stage in {self.stop_time - self.start_time}')

    def _pre_epoch(self):
        self.epoch_start_time = datetime.now()
        self.table['Epoch'] = self.current_epoch
        self.pre_epoch()
        self.pipeline._pre_epoch()

    def _post_epoch(self):
        self.epoch_stop_time = datetime.now()
        self._reduce_metrics()
        self.post_epoch()
        self.pipeline._post_epoch()
        self._update_table()
        self.current_epoch += 1

    def _reduce_metrics(self):
        self.track(name='misc/epoch', value=self.current_epoch, prefixed=False)
        self.track(
            name='misc/epoch_time', value=(self.epoch_stop_time - self.epoch_start_time).total_seconds(), prefixed=False
        )
        self.tracker.next_epoch()
        pass

    def _setup_table(self):
        for column_dct in self._metrics():
            display_name = column_dct.pop('name')
            column_dct.pop('metric')
            self.table.add_column(display_name, **column_dct)

    def _update_table(self):
        self.table.update('Epoch', self.current_epoch)
        self.table.update('Time/Epoch', (datetime.now() - self.start_time) / self.current_epoch)
        self.table.update(
            'ETA', (datetime.now() - self.start_time) / self.current_epoch * (self.max_epochs - self.current_epoch)
        )
        for column_dct in self._metrics():
            display_name = column_dct['name']
            metric_name = column_dct['metric']
            if metric_name is not None:
                self.table.update(display_name, self.tracker[metric_name][-1])
        self.table.next_row()

    def _metrics(self):
        metrics = []
        for column in self.table_columns():
            if isinstance(column, str):
                metrics.append({'name': column, 'metric': column})
            elif isinstance(column, dict):
                if 'name' not in column:
                    raise ValueError('Column dict must contain a "name" key')
                if 'metric' not in column:
                    raise ValueError('Column dict must contain a "metric" key')
                metrics.append(column)
            else:
                raise ValueError(f'Invalid column: {column}. Must be a string or a dict.')
        return metrics


class TrainValStage(Stage):
    def __init__(self):
        super().__init__()
        self.is_train = True

    def train_dataset(self):
        train_ds = self.pipeline.datasets.get('train')
        if train_ds is None:
            raise ValueError(
                'No "train" dataset found in pipeline. Use register_dataset("train", ...) to register a dataset.'
            )
        return train_ds

    def val_dataset(self):
        val_ds = self.pipeline.datasets.get('val')
        if val_ds is None:
            raise ValueError(
                'No "val" dataset found in pipeline. Use register_dataset("val", ...) to register a dataset.'
            )
        return val_ds

    def optimizers(self):
        return self.pipeline.optimizers.values()

    def loss_metric_name(self):
        return 'loss'

    def train_metric_prefix(self):
        return 'train'

    def val_metric_prefix(self):
        return 'val'

    def gradient_clip(self):
        return 0.0

    def run_epoch(self):
        self.train_epoch()
        self.val_epoch()

    def step(self, batch) -> torch.Tensor:
        raise NotImplementedError()

    def train_step(self, batch):
        return self.step(batch)

    def val_step(self, batch):
        return self.step(batch)

    def zero_grad(self):
        for optimizer in self.optimizers():
            optimizer.zero_grad()

    def clip_gradients(self):
        for optimizer in self.optimizers():
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.gradient_clip())

    def optimize(self, loss):
        loss.backward()

        if self.gradient_clip():
            self.clip_gradients()

        for optimizer in self.optimizers():
            optimizer.step()

    def train_epoch(self):
        self.is_train = True
        self.metric_prefix = self.train_metric_prefix()

        train_ds = self.train_dataset()
        if hasattr(train_ds, 'sampler') and hasattr(train_ds.sampler, 'set_epoch'):
            train_ds.sampler.set_epoch(self.current_epoch)

        for batch in train_ds:
            step_start_time = time.perf_counter_ns()
            self.zero_grad()
            loss = self.train_step(batch)
            self.optimize(loss)
            step_end_time = time.perf_counter_ns()

            self.track_reduce(self.loss_metric_name(), loss)
            self.track_reduce('misc/total_train_batches', torch.tensor(1), reduction=Reduction.SUM, prefixed=False)
            self.track_reduce(
                'misc/worker_train_batches',
                torch.tensor(1),
                reduction=Reduction.SUM,
                reduce_globally=False,
                prefixed=False,
            )
            self.track_reduce('misc/step_time_ms', torch.tensor(step_end_time - step_start_time) / 1e6, prefixed=False)

        for name, scheduler in self.pipeline.schedulers.items():
            self.track(f'misc/lr_{name}', scheduler.get_last_lr()[0], prefixed=False)
            scheduler.step()

    @torch.no_grad()
    def val_epoch(self):
        self.is_train = False
        self.metric_prefix = self.val_metric_prefix()

        for batch in self.val_dataset():
            loss = self.val_step(batch)
            self.track_reduce(self.loss_metric_name(), loss)
            self.track_reduce('misc/total_val_batches', torch.tensor(1), reduction=Reduction.SUM, prefixed=False)
            self.track_reduce(
                'misc/worker_val_batches',
                torch.tensor(1),
                reduction=Reduction.SUM,
                reduce_globally=False,
                prefixed=False,
            )

    def table_columns(self):
        columns = super().table_columns()
        columns.insert(1, {'name': '[Train] Loss', 'metric': f'{self.train_metric_prefix()}/{self.loss_metric_name()}'})
        columns.insert(2, {'name': '[Val] Loss', 'metric': f'{self.val_metric_prefix()}/{self.loss_metric_name()}'})
        return columns
