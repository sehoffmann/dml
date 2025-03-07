import csv
import json
import os
import sys
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING, Union

import pynvml
import torch
from omegaconf import OmegaConf
from progress_table import ProgressTable
from torch.profiler import profile, ProfilerActivity

from ..git import git_diff
from ..util.logging import DevNullIO, experiment_header, general_diagnostics, IORedirector
from ..util.wandb import wandb_is_initialized, wandb_set_startup_timeout
from . import checkpoint as dml_checkpoint, logging as dml_logging
from .distributed import all_gather_object, is_root


if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .stage import Stage


__all__ = [
    'TimedeltaFormatter',
    'CallbackList',
    'CbPriority',
    'Callback',
    'ProfilerCallback',
    'TimerCallback',
    'TableCallback',
    'ReduceMetricsCallback',
    'CheckpointCallback',
    'CsvCallback',
    'WandbInitCallback',
    'WandbLoggerCallback',
    'TensorboardCallback',
    'CudaCallback',
]


class TimedeltaFormatter:
    """
    A formatter that converts a number of seconds to a human-readable string.
    """

    def __init__(self, microseconds=False):
        self.microseconds = microseconds

    def __call__(self, seconds: float) -> str:
        delta = timedelta(seconds=seconds)
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

    WANDB_INIT = -200
    CHECKPOINT = -190
    STAGE_TIMER = -180
    DIAGNOSTICS = -170
    CUDA = -160
    GIT = -150
    METRIC_REDUCTION = -100

    OBJECT_METHODS = 0

    PROFILER = 100
    WANDB_LOGGER = 110
    CSV = 110
    TENSORBOARD = 110
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

    def post_step(self, stage: 'Stage'):
        """
        Executed after each step. Stage must call `finish_step` to trigger this callback.
        """
        pass


class ProfilerCallback(Callback):
    """
    A callback that profiles the training process and saves the results to a file.
    """

    def __init__(self, epochs=None, record_shapes=False, schedule=None):
        self.epochs = epochs
        self.record_shapes = record_shapes
        self.schedule = schedule

        self.profiler = None
        self._capturing = False

    def pre_epoch(self, stage: 'Stage'):
        if self.epochs and stage.current_epoch not in self.epochs:
            return

        self.profiler = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=self.record_shapes,
            schedule=self.schedule,
        )
        self.profiler.__enter__()
        self._capturing = True

    def post_epoch(self, stage):
        if self.epochs and (stage.current_epoch - 1) not in self.epochs:
            return

        self.profiler.__exit__(None, None, None)
        self._capturing = False

        if stage.run_dir:
            outfile = str(stage.run_dir / f'{stage.name}_epoch{stage.current_epoch - 1}_trace.json')
            self.profiler.export_chrome_trace(outfile)

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if self._capturing:
            self.profiler.__exit__(exc_type, exc_value, traceback)
            self._capturing = False


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

        epoch_time = (stage.epoch_end_time - self.epoch_start_time).total_seconds()
        total_time = (stage.epoch_end_time - self.start_time).total_seconds()
        stage.log('misc/epoch_time', epoch_time, prefixed=False, log_step=False)
        stage.log('misc/total_time', total_time, prefixed=False, log_step=False)

        if stage._run_epoch_overridden:
            average_epoch_time = (stage.epoch_end_time - self.start_time) / (stage.current_epoch + 1)
            eta = average_epoch_time * (stage.max_epochs - stage.current_epoch - 1)
            stage.log('misc/eta', eta.total_seconds(), prefixed=False, log_step=False)


class TableCallback(Callback):
    """
    A callback that updates a table with the latest metrics from a stage.
    """

    def __init__(self):
        self._table = None
        self.tracked_metrics = {}
        self.formatters = {}

    def get_table(self, stage: 'Stage'):
        if self._table is None:
            self._table = ProgressTable(file=sys.stdout if is_root() else DevNullIO(), interactive=0)
            self.track_metric(stage, 'Epoch', width=5)
            self.track_metric(stage, 'Took', 'misc/epoch_time', formatter=TimedeltaFormatter(), width=7)
            if stage._run_epoch_overridden:
                self.track_metric(stage, 'ETA', 'misc/eta', formatter=TimedeltaFormatter(), width=7)
        return self._table

    def set_table(self, value):
        self._table = value

    def track_metric(
        self,
        stage: 'Stage',
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

        table = self.get_table(stage)
        table.add_column(name, width=width, color=color, alignment=alignment)

        if metric:
            self.tracked_metrics[name] = metric
            self.formatters[name] = formatter

    def pre_stage(self, stage: 'Stage'):
        self.get_table(stage)  # Ensure the table has been created at this point

    def post_stage(self, stage: 'Stage'):
        table = self.get_table(stage)
        table.close()

    def pre_epoch(self, stage: 'Stage'):
        table = self.get_table(stage)
        if 'Epoch' in self.get_table(stage).column_names:
            table['Epoch'] = stage.current_epoch

    def post_epoch(self, stage: 'Stage'):
        table = self.get_table(stage)
        metrics = stage.history.last()

        for column_name, metric_name in self.tracked_metrics.items():
            if column_name not in table.column_names:  # When does this happen?
                continue

            if metric_name in metrics:
                value = metrics[metric_name]
                formatter = self.formatters[column_name]
                if formatter is not None:
                    value = formatter(value)
                table.update(column_name, value)
            else:
                pass  # don't update -> empty cell

        table.next_row()


class ReduceMetricsCallback(Callback):
    """
    A callback that reduces the metrics at the end of each epoch and appends them to the history.
    """

    def __init__(self, log_every_n_steps=50):
        self.log_every_n_steps = log_every_n_steps

    def _reduce_epoch_metrics(self, stage):
        metrics = stage.metrics.reduce()
        stage.history.append_metrics(**metrics)

    def _reduce_step_metrics(self, stage):
        metrics = stage.step_metrics.reduce()
        stage.step_history.append_metrics(**metrics)

    def post_epoch(self, stage: 'Stage'):
        stage.log('misc/epoch', stage.current_epoch, prefixed=False, reduction='max')
        self._reduce_epoch_metrics(stage)
        stage.step = 0  # Reset the step counter

    def post_step(self, stage: 'Stage'):
        stage.log('misc/step', stage.global_step, prefixed=False, reduction='max')

        if stage.global_step % self.log_every_n_steps == 0:
            self._reduce_step_metrics(stage)

        stage.step += 1
        stage.global_step += 1

    def post_stage(self, stage):
        has_unreduced_metrics = False
        for metric in stage.step_metrics.metrics.values():
            if metric.update_called:
                has_unreduced_metrics = True
                break

        # need to check global_step > 0 to avoid reducing when finish_step() was never called once
        if has_unreduced_metrics and stage.global_step > 0:
            self._reduce_step_metrics(stage)


class CheckpointCallback(Callback):
    """
    Creates the checkpoint directory and optionally setups io redirection.
    """

    def __init__(self, run_dir: Union[str, Path], redirect_io: bool = True):
        """
        Initialize the callback with the given path.

        Args:
            run_dir: The path to the checkpoint directory.
            redirect_io: Whether to redirect the IO to a file. Defaults to True.
        """
        self.run_dir = Path(run_dir)
        self.redirect_io = redirect_io
        self.io_redirector = None

    def pre_run(self, pipe: 'Pipeline'):
        if not dml_checkpoint.is_valid_checkpoint_dir(self.run_dir):
            dml_checkpoint.create_checkpoint_dir(self.run_dir)
            dml_checkpoint.save_config(pipe.config, self.run_dir)

        self.io_redirector = IORedirector(pipe.run_dir / 'log.txt')
        self.io_redirector.install()

        with open(pipe.run_dir / "environment.txt", 'w') as f:
            for k, v in os.environ.items():
                f.write(f"{k}={v}\n")

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if self.io_redirector is not None:
            self.io_redirector.uninstall()


class CsvCallback(Callback):
    """
    Saves metrics to a CSV file at the end of each epoch.
    """

    def __init__(self, directory: Union[str, Path]):
        """
        Initialize the callback with the given path.

        Args:
            directory (Union[str, Path]): The path to the directory where the CSV files will be saved.
        """
        self.directory = Path(directory)
        self.last_steps = {}

    def _build_name(self, stage: 'Stage', prefix: str):
        duplicate_stages = [s for s in stage.pipe.stages if s.name == stage.name]
        idx = duplicate_stages.index(stage)
        if len(duplicate_stages) > 1:
            return self.directory / f'{prefix}_{stage.name}_{idx + 1}.csv'
        else:
            return self.directory / f'{prefix}_{stage.name}.csv'

    def epoch_path(self, stage: 'Stage'):
        return self._build_name(stage, 'epoch_metrics')

    def step_path(self, stage: 'Stage'):
        return self._build_name(stage, 'step_metrics')

    def pre_stage(self, stage: 'Stage'):
        # If for some reason we can't write to the file or it exists already, its better to fail early
        with open(self.epoch_path(stage), 'x'):
            pass

    def _write_history(self, file, history, step_metric, step_name):
        writer = csv.writer(file)

        metric_names = list(history.keys())
        metric_names.remove(step_metric)

        writer.writerow([step_name] + metric_names)  # Header
        for row in history.rows():
            csv_row = [row[step_metric]] + [row[name] for name in metric_names]
            writer.writerow(csv_row)

    def _maybe_write_step_metrics(self, stage: 'Stage'):
        if stage.step_history.num_steps > self.last_steps.get(stage, 0):
            self.last_steps[stage] = stage.step_history.num_steps
            with open(self.step_path(stage), 'w') as f:
                self._write_history(f, stage.step_history, 'misc/step', 'step')

    def post_epoch(self, stage: 'Stage'):
        with open(self.epoch_path(stage), 'w') as f:
            self._write_history(f, stage.history, 'misc/epoch', 'epoch')

    def post_step(self, stage: 'Stage'):
        self._maybe_write_step_metrics(stage)

    def post_stage(self, stage):
        self._maybe_write_step_metrics(stage)  # edge case: last steps of training


class WandbInitCallback(Callback):
    """
    A callback that initializes Weights & Biases and closes it at the end.
    This is separated from the WandbLoggerCallback to ensure it is called right at the beginning of training.
    """

    def __init__(self, project, entity, group, tags, startup_timeout, **kwargs):
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb is required for the WandbInitCallback')

        self.wandb = wandb
        self.project = project
        self.entity = entity
        self.group = group
        self.tags = tags
        self.startup_timeout = startup_timeout
        self.kwargs = kwargs

    def pre_run(self, pipe: 'Pipeline'):
        wandb_set_startup_timeout(self.startup_timeout)
        self.wandb.init(
            config=OmegaConf.to_container(pipe.config, resolve=True),
            name=pipe.name,
            project=self.project,
            entity=self.entity,
            group=self.group,
            tags=self.tags,
            **self.kwargs,
        )

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if wandb_is_initialized():
            self.wandb.finish(exit_code=0 if exc_type is None else 1)


class WandbLoggerCallback(Callback):
    """
    A callback that logs metrics to Weights & Biases.
    """

    def __init__(self):
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb is required for the WandbLoggerCallback')

        self.wandb = wandb

    def post_epoch(self, stage: 'Stage'):
        metrics = stage.history.last()
        self.wandb.log(metrics, commit=True, step=stage.current_epoch)


class TensorboardCallback(Callback):
    """
    A callback that logs metrics to Tensorboard.
    """

    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError:
            raise ImportError('tensorflow is required for the TensorboardCallback')

    def pre_run(self, pipe):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def post_epoch(self, stage: 'Stage'):
        metrics = stage.history.last()
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, stage.current_epoch)

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if self.writer is not None:
            self.writer.close()


class DiagnosticsCallback(Callback):
    """
    A callback that logs diagnostics information at the beginning of training.
    """

    def pre_run(self, pipe):
        header = '\n' + experiment_header(pipe.name, pipe.run_dir, pipe.start_time)
        dml_logging.info(header)

        diagnostics = general_diagnostics()

        diagnostics += '\n* CONFIG:\n'
        diagnostics += '\n'.join(f'    {line}' for line in OmegaConf.to_yaml(pipe.config, resolve=True).splitlines())

        dml_logging.info(diagnostics)

    def post_stage(self, stage):
        if len(stage.pipe.stages) > 1:
            dml_logging.info(f'Finished stage in {stage.end_time - stage.start_time}')

    def post_run(self, pipe):
        dml_logging.info(f'Finished training in {pipe.stop_time - pipe.start_time} ({pipe.stop_time})')
        if pipe.has_checkpointing:
            dml_logging.info(f'Outputs have been saved to {pipe.run_dir}')


class GitDiffCallback(Callback):
    """
    A callback that prints a git diff and if checkpointing is enabled, saves it to the checkpoint directory.
    """

    def pre_run(self, pipe):
        diff = git_diff()
        if diff is None:
            return

        if pipe.has_checkpointing and is_root():
            self._save(pipe.run_dir / 'git_diff.txt', diff)

        msg = '* GIT-DIFF:\n'
        msg += '\n'.join('    ' + line for line in diff.splitlines())
        dml_logging.info(msg)

    def _save(self, path, diff):
        with open(path, 'w') as f:
            f.write(diff)


class CudaCallback(Callback):
    """
    Logs various properties pertaining to CUDA devices.
    """

    @staticmethod
    def _call_pynvml(method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except pynvml.NVMLError:
            return None

    def pre_run(self, pipe):
        handle = torch.cuda._get_pynvml_handler(pipe.device)

        info = {
            'name': self._call_pynvml(pynvml.nvmlDeviceGetName, handle),
            'uuid': self._call_pynvml(pynvml.nvmlDeviceGetUUID, handle),
            'serial': self._call_pynvml(pynvml.nvmlDeviceGetSerial, handle),
            'torch_device': str(pipe.device),
            'minor_number': self._call_pynvml(pynvml.nvmlDeviceGetMinorNumber, handle),
            'architecture': self._call_pynvml(pynvml.nvmlDeviceGetArchitecture, handle),
            'brand': self._call_pynvml(pynvml.nvmlDeviceGetBrand, handle),
            'vbios_version': self._call_pynvml(pynvml.nvmlDeviceGetVbiosVersion, handle),
            'driver_version': self._call_pynvml(pynvml.nvmlSystemGetDriverVersion),
            'cuda_driver_version': self._call_pynvml(pynvml.nvmlSystemGetCudaDriverVersion_v2),
            'nvml_version': self._call_pynvml(pynvml.nvmlSystemGetNVMLVersion),
            'total_memory': self._call_pynvml(pynvml.nvmlDeviceGetMemoryInfo, handle, pynvml.nvmlMemory_v2).total,
            'reserved_memory': self._call_pynvml(pynvml.nvmlDeviceGetMemoryInfo, handle, pynvml.nvmlMemory_v2).reserved,
            'num_gpu_cores': self._call_pynvml(pynvml.nvmlDeviceGetNumGpuCores, handle),
            'power_managment_limit': self._call_pynvml(pynvml.nvmlDeviceGetPowerManagementLimit, handle),
            'power_managment_default_limit': self._call_pynvml(pynvml.nvmlDeviceGetPowerManagementDefaultLimit, handle),
            'cuda_compute_capability': self._call_pynvml(pynvml.nvmlDeviceGetCudaComputeCapability, handle),
        }
        all_devices = all_gather_object(info)

        msg = '* CUDA-DEVICES:\n'
        info_strings = [
            f'{info["torch_device"]} -> /dev/nvidia{info["minor_number"]} -> {info["name"]} (UUID: {info["uuid"]}) (VRAM: {info["total_memory"] / 1000 ** 2:.0f} MB)'
            for info in all_devices
        ]
        msg += '\n'.join(f'    - [{i}] {info_str}' for i, info_str in enumerate(info_strings))
        dml_logging.info(msg)

        if pipe.has_checkpointing and is_root():
            self._save(pipe.run_dir / 'cuda_devices.json', all_devices)

    def _save(self, path, all_devices):
        with open(path, 'w') as f:
            devices = {f'rank_{i}': device for i, device in enumerate(all_devices)}
            obj = {'devices': devices}
            json.dump(obj, f, indent=4)
