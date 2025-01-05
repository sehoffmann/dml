import csv
import os
import sys
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING, Union

import torch
from omegaconf import OmegaConf
from progress_table import ProgressTable

from ..git import git_diff
from ..util.logging import DevNullIO, experiment_header, general_diagnostics, IORedirector
from ..util.wandb import wandb_is_initialized, wandb_set_startup_timeout
from . import logging as dml_logging
from .distributed import all_gather_object, is_root

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .stage import Stage


__all__ = [
    'TimedeltaFormatter',
    'CallbackList',
    'CbPriority',
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


class CallbackList:
    """
    A priority queue of callbacks.
    """

    def __init__(self):
        self.callbacks = []

    def append(self, callback: 'Callback', priority: int = 0):
        """
        Append a callback to the list with the given priority.

        Args:
            callback (Callback): The callback to append.
            priority (int, optional): The priority of the callback. Defaults to 0.
        """
        self.callbacks.append((priority, callback))

    def __iter__(self):
        for _, callback in sorted(self.callbacks, key=lambda x: x[0]):
            yield callback

    def __len__(self):
        return len(self.callbacks)

    def __add__(self, other: 'CallbackList'):
        result = CallbackList()
        result.callbacks = self.callbacks + other.callbacks
        return result


class CbPriority(IntEnum):
    """
    Default priorities for callbacks used by the pipeline and stage classes.
    """

    WANDB = -200
    CHECKPOINT = -190
    STAGE_TIMER = -180
    DIAGNOSTICS = -170
    GIT = -160
    METRIC_REDUCTION = -150

    OBJECT_METHODS = 0

    CSV = 110
    TABLE = 120


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

        with open(pipe.checkpoint_dir.path / "environment.txt", 'w') as f:
            for k, v in os.environ.items():
                f.write(f"{k}={v}\n")

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
            duplicate_stages = [s for s in stage.pipe.stages if s.name == stage.name]
            idx = duplicate_stages.index(stage)
            if len(duplicate_stages) > 1:
                return self.path / f'metrics_{stage.name}_{idx + 1}.csv'
            else:
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
                writer.writerow(['epoch'] + list(metrics))

            row = [stage.current_epoch - 1]  # epoch is already incremented
            for value in metrics.values():
                row.append(value.item())
            writer.writerow(row)


class WandbCallback(Callback):
    """
    A callback that logs metrics to Weights & Biases.
    """

    def __init__(self, entity, project, group, tags, startup_timeout, **kwargs):
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb is required for the WandbCallback')

        self.wandb = wandb
        self.entity = entity
        self.project = project
        self.group = group
        self.tags = tags
        self.startup_timeout = startup_timeout
        self.kwargs = kwargs

    def pre_run(self, pipe: 'Pipeline'):
        wandb_set_startup_timeout(self.startup_timeout)
        self.wandb.init(
            config=OmegaConf.to_container(pipe.config, resolve=True),
            name=pipe.name,
            entity=self.entity,
            project=self.project,
            group=self.group,
            tags=self.tags,
            **self.kwargs,
        )

    def post_epoch(self, stage: 'Stage'):
        metrics = stage.history.last()
        self.wandb.log(metrics)

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if wandb_is_initialized():
            self.wandb.finish(exit_code=0 if exc_type is None else 1)


class DiagnosticsCallback(Callback):
    """
    A callback that logs diagnostics information at the beginning of training.
    """

    def pre_run(self, pipe):
        header = '\n' + experiment_header(pipe.name, pipe.checkpoint_dir, pipe.start_time)
        dml_logging.info(header)

        diagnostics = general_diagnostics()

        diagnostics += '\n* DEVICES:\n'
        devices = all_gather_object(str(pipe.device))
        diagnostics += '\n'.join(f'    - [Rank {i}] {device}' for i, device in enumerate(devices))

        diagnostics += '\n* CONFIG:\n'
        diagnostics += '\n'.join(f'    {line}' for line in OmegaConf.to_yaml(pipe.config, resolve=True).splitlines())

        dml_logging.info(diagnostics)

    def post_stage(self, stage):
        if len(stage.pipe.stages) > 1:
            dml_logging.info(f'Finished stage in {stage.end_time - stage.start_time}')

    def post_run(self, pipe):
        dml_logging.info(f'Finished training in {pipe.stop_time - pipe.start_time} ({pipe.stop_time})')
        if pipe.checkpointing_enabled:
            dml_logging.info(f'Outputs have been saved to {pipe.checkpoint_dir}')


class GitDiffCallback(Callback):
    """
    A callback that prints a git diff and if checkpointing is enabled, saves it to the checkpoint directory.
    """

    def pre_run(self, pipe):
        diff = git_diff()

        if pipe.checkpointing_enabled:
            self._save(pipe.checkpoint_dir.path / 'git_diff.txt', diff)

        msg = '* GIT-DIFF:\n'
        msg += '\n'.join('\t' + line for line in diff.splitlines())
        dml_logging.info(msg)

    def _save(self, path, diff):
        with open(path, 'w') as f:
            f.write(diff)
