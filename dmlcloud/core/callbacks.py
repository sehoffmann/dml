import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING, Union

import torch
from progress_table import ProgressTable

from ..util.logging import DevNullIO, IORedirector
from . import logging as dml_logging
from .distributed import is_root

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .stage import Stage


__all__ = [
    'TimedeltaFormatter',
    'Callback',
    'TimerCallback',
    'TableCallback',
    'ReduceMetricsCallback',
    'CheckpointCallback',
    'CsvCallback',
    'WandbCallback',
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


class Callback:
    """
    A callback that can be registered to a stage or the whole pipeline to receive updates on the training progress.
    """

    def pre_run(self, pipe: 'Pipeline'):
        """
        Executed before the pipeline starts.
        """
        pass

    def post_run(self, pipe: 'Pipeline'):
        """
        Executed after the pipeline finishes.
        """
        pass

    def cleanup(self, pipe: 'Pipeline', exc_type, exc_value, traceback):
        """
        Executed after the pipeline finishes, even if an error occurred.
        E.g. to close file handles.

        Args:
            pipe (Pipeline): The pipeline that is being cleaned up.
            exc_type (type): The type of the exception that caused the cleanup or None if no exception occurred.
            exc_value (Exception): The exception that caused the cleanup or None if no exception occurred.
            traceback (Traceback): The traceback of the exception that caused the cleanup or None if no exception occurred.
        """
        pass

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


class TimerCallback(Callback):
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

        average_epoch_time = (stage.epoch_end_time - self.start_time) / (stage.current_epoch + 1)
        eta = average_epoch_time * (stage.max_epochs - stage.current_epoch - 1)
        stage.log('misc/eta', eta.total_seconds(), prefixed=False)

        if len(stage.pipe.stages) > 1:
            dml_logging.info(f'Finished stage in {stage.end_time - stage.start_time}')


class TableCallback(Callback):
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


class ReduceMetricsCallback(Callback):
    """
    A callback that reduces the metrics at the end of each epoch and appends them to the history.
    """

    def post_epoch(self, stage: 'Stage'):
        metrics = stage.tracker.reduce()
        stage.history.append_metrics(**metrics)
        stage.history.next_step()


class CheckpointCallback(Callback):
    """
    Creates the checkpoint directory and optionally setups io redirection.
    """

    def __init__(self, root_path: Union[str, Path], redirect_io: bool = True):
        """
        Initialize the callback with the given root path.

        Args:
            root_path (Union[str, Path]): The root path where the checkpoint directory will be created.
            redirect_io (bool, optional): Whether to redirect the IO to a file. Defaults to True.
        """
        self.root_path = Path(root_path)
        self.redirect_io = redirect_io
        self.io_redirector = None

    def pre_run(self, pipe: 'Pipeline'):
        if not pipe.checkpoint_dir.is_valid:
            pipe.checkpoint_dir.create()
            pipe.checkpoint_dir.save_config(pipe.config)

        self.io_redirector = IORedirector(pipe.checkpoint_dir.log_file)
        self.io_redirector.install()

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if self.io_redirector is not None:
            self.io_redirector.uninstall()


class CsvCallback(Callback):
    """
    Saves metrics to a CSV file at the end of each epoch.
    """

    def __init__(self, path: Union[str, Path], append_stage_name: bool = False):
        """
        Initialize the callback with the given path.

        Args:
            path (Union[str, Path]): The file path where the callback will operate.
            append_stage_name (bool, optional): Whether to append the stage name to the path. Defaults to False.
        """
        self.path = Path(path)
        self.append_stage_name = append_stage_name

    def csv_path(self, stage: 'Stage'):
        """
        Generate the CSV file path for the given stage.

        If `append_stage_name` is True, the method appends the stage name to the file name.
        Otherwise, it returns the base path.

        Args:
            stage (Stage): The stage object containing the name to be appended.

        Returns:
            Path: The complete path to the CSV file.
        """
        if self.append_stage_name:
            return self.path / f'metrics_{stage.name}.csv'
        else:
            return self.path

    def pre_stage(self, stage: 'Stage'):
        # If for some reason we can't write to the file or it exists already, its better to fail early
        with open(self.csv_path(stage), 'x'):
            pass

    def post_epoch(self, stage: 'Stage'):
        with open(self.csv_path(stage), 'a') as f:
            writer = csv.writer(f)

            metrics = stage.history.last()

            # Write the header if the file is empty
            if f.tell() == 0:
                writer.writerow(['Epoch'] + list(metrics))

            row = [stage.current_epoch - 1]  # epoch is already incremented
            for value in metrics.values():
                row.append(value.item())
            writer.writerow(row)


class WandbCallback(Callback):
    """
    A callback that logs metrics to Weights & Biases.
    """

    def __init__(self):
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb is required for the WandbCallback')

        self.wandb = wandb

    def pre_stage(self, stage: 'Stage'):
        self.wandb.init(project='dmlcloud', config=stage.config)

    def post_epoch(self, stage: 'Stage'):
        metrics = stage.history.last()
        self.wandb.log(metrics)
