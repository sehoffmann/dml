import sys
from datetime import datetime, timedelta
from typing import Callable, Optional

import torch
from progress_table import ProgressTable

from ..util.logging import DevNullIO
from . import logging as dml_logging
from .distributed import is_root


__all__ = [
    'TimedeltaFormatter',
    'StageCallback',
    'TimreCallback',
    'TableCallback',
    'ReduceMetricsCallback',
]


class TimedeltaFormatter:
    """
    A formatter that converts a number of seconds to a human-readable string.
    """

    def __init__(self, microseconds=False):
        self.microseconds = microseconds

    def __call__(self, value: torch.Tensor) -> str:
        delta = timedelta(seconds=value.item())
        if not self.microseconds:
            delta -= timedelta(microseconds=delta.microseconds)
        return str(delta)


class StageCallback:
    """
    A callback that can be registered to a stage to receive updates on the training progress.
    """

    def pre_stage(self, stage: 'Stage'):
        """
        Executed before the stage starts.
        """
        pass

    def post_stage(self, stage: 'Stage'):
        """
        Executed after the stage finishes.
        """
        pass

    def pre_epoch(self, stage: 'Stage'):
        """
        Executed before each epoch.
        """
        pass

    def post_epoch(self, stage: 'Stage'):
        """
        Executed after each epoch.
        """
        pass


class TimerCallback(StageCallback):
    """
    A callback that logs the time taken for each epoch.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.epoch_start_time = None
        self.epoch_end_time = None

    def pre_stage(self, stage: 'Stage'):
        self.start_time = datetime.now()

    def post_stage(self, stage: 'Stage'):
        self.end_time = datetime.now()

    def pre_epoch(self, stage: 'Stage'):
        self.epoch_start_time = datetime.now()

    def post_epoch(self, stage: 'Stage'):
        self.epoch_end_time = datetime.now()

        stage.log('misc/epoch', stage.current_epoch, prefixed=False)
        stage.log('misc/epoch_time', (stage.epoch_end_time - self.epoch_start_time).total_seconds(), prefixed=False)
        stage.log('misc/total_time', (stage.epoch_end_time - self.start_time).total_seconds(), prefixed=False)

        eta = (
            (stage.epoch_end_time - self.start_time)
            / (stage.current_epoch + 1)
            * (stage.max_epochs - stage.current_epoch - 1)
        )
        stage.log('misc/eta', eta.total_seconds(), prefixed=False)

        if len(stage.pipe.stages) > 1:
            dml_logging.info(f'Finished stage in {stage.end_time - stage.start_time}')


class TableCallback(StageCallback):
    """
    A callback that updates a table with the latest metrics from a stage.
    """

    def __init__(self):
        self._table = None
        self.tracked_metrics = {}
        self.formatters = {}

    @property
    def table(self):
        if self._table is None:
            self.table = ProgressTable(file=sys.stdout if is_root() else DevNullIO())
            self.track_metric('Epoch', width=5)
            self.track_metric('Took', 'misc/epoch_time', formatter=TimedeltaFormatter(), width=7)
            self.track_metric('ETA', 'misc/eta', formatter=TimedeltaFormatter(), width=7)
        return self._table

    @table.setter
    def table(self, value):
        self._table = value

    def track_metric(
        self,
        name: str,
        metric: Optional[str] = None,
        formatter: Optional[Callable] = None,
        width: Optional[int] = None,
        color: Optional[str] = None,
        alignment: Optional[str] = None,
    ):
        """
        Track a metric in the table.

        If no metric name is provided, only a column is created and the caller must update the value manually.
        If a formatter is provided, the metric value will be passed through the formatter before being displayed.

        For a detailed description of width, color, and alignment, see `ProgressTable.add_column`.

        Args:
            name (str): The name of the column.
            metric (str, optional): The name of the metric to track. Defaults to None.
            formatter (Callable, optional): A function that takes the metric value and returns a string. Defaults to None.
            width (int, optional): The width of the column. Defaults to None.
            color (str, optional): The color of the column. Defaults to None.
            alignment (str, optional): The alignment of the column. Defaults to None.
        """
        if formatter and not metric:
            raise ValueError('Cannot provide a formatter without a metric name')

        self.table.add_column(name, width=width, color=color, alignment=alignment)

        if metric:
            self.tracked_metrics[name] = metric
            self.formatters[name] = formatter

    def pre_stage(self, stage: 'Stage'):
        _ = self.table  # Ensure the table has been created at this point

    def post_stage(self, stage: 'Stage'):
        self.table.close()

    def pre_epoch(self, stage: 'Stage'):
        if 'Epoch' in self.table.column_names:
            self.table['Epoch'] = stage.current_epoch

    def post_epoch(self, stage: 'Stage'):
        metrics = stage.history.last()

        for column_name, metric_name in self.tracked_metrics.items():
            if column_name not in self.table.column_names:
                continue

            value = metrics[metric_name]
            formatter = self.formatters[column_name]
            if formatter is not None:
                value = formatter(value)

            self.table.update(column_name, value)

        self.table.next_row()


class ReduceMetricsCallback(StageCallback):
    """
    A callback that reduces the metrics at the end of each epoch and appends them to the history.
    """

    def post_epoch(self, stage: 'Stage'):
        metrics = stage.tracker.reduce()
        stage.history.append_metrics(**metrics)
        stage.history.next_step()
