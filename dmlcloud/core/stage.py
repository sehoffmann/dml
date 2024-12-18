import sys
from datetime import datetime, timedelta
from typing import Any, Optional

from progress_table import ProgressTable

from ..util.logging import DevNullIO
from . import logging as dml_logging
from .distributed import is_root
from .metrics import Tracker, TrainingHistory

__all__ = [
    'Stage',
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

        self.history = TrainingHistory()
        self.tracker = Tracker()

        self.start_time = None
        self.stop_time = None
        self.epoch_start_time = None
        self.epoch_stop_time = None

        self.metric_prefix = None
        self.barrier_timeout = None

        self.table = None
        self.columns = {}

    @property
    def device(self):
        return self.pipeline.device

    @property
    def config(self):
        return self.pipeline.config

    @property
    def current_epoch(self):
        return self.history.num_steps

    def log(self, name: str, value: Any, reduction: str = 'mean', prefixed: bool = True):
        if prefixed:
            name = f'{self.metric_prefix}/{name}'
        self.tracker.log(name, value, reduction)

    def add_metric(self, name, metric):
        metric = metric.to(self.device)
        self.tracker.add_metric(name, metric)
        return metric

    def add_column(
        self,
        name: str,
        metric: Optional[str] = None,
        width: Optional[int] = None,
        color: Optional[str] = None,
        alignment: Optional[str] = None,
    ):
        self.columns[name] = metric
        self.table.add_column(name, width=width, color=color, alignment=alignment)

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

    def run(self):
        """
        Runs this stage. Either until max_epochs are reached, or until stop_stage() is called.
        """
        self._pre_stage()
        while self.max_epochs is None or self.current_epoch < self.max_epochs:
            self._pre_epoch()
            self.run_epoch()
            self._post_epoch()
        self._post_stage()

    def _pre_stage(self):
        self.start_time = datetime.now()
        if len(self.pipeline.stages) > 1:
            dml_logging.info(f'\n========== STAGE: {self.name} ==========')

        self.table = ProgressTable(file=sys.stdout if is_root() else DevNullIO())
        self.add_column('Epoch', None, color='bright', width=5)
        self.add_column('Took', None, width=7)
        self.add_column('ETA', None, width=7)

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

    def _post_epoch(self):
        self.epoch_stop_time = datetime.now()
        self._reduce_metrics()
        self.post_epoch()
        self._update_table()

    def _reduce_metrics(self):
        # self.log('misc/epoch', self.current_epoch, prefixed=False)
        # self.log('misc/epoch_time', (self.epoch_stop_time - self.epoch_start_time).total_seconds())
        metrics = self.tracker.reduce()
        self.history.append_metrics(**metrics)
        self.history.next_step()

    def _update_table(self):
        time = datetime.now() - self.epoch_start_time
        self.table.update('Took', str(time - timedelta(microseconds=time.microseconds)))

        per_epoch = (datetime.now() - self.start_time) / self.current_epoch
        eta = per_epoch * (self.max_epochs - self.current_epoch)
        self.table.update('ETA', str(eta - timedelta(microseconds=eta.microseconds)))

        last_metrics = self.history.last()
        for name, metric in self.columns.items():
            if metric is not None:
                self.table.update(name, last_metrics[metric])

        self.table.next_row()
