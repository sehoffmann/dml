from typing import Any, Callable

import torch

from . import logging as dml_logging
from .callbacks import (
    Callback,
    CallbackList,
    CbPriority,
    ProfilerCallback,
    ReduceMetricsCallback,
    TableCallback,
    TimerCallback,
)
from .distributed import is_root
from .metrics import Tracker, TrainingHistory


__all__ = [
    'Stage',
]


class _ForwardCallback(Callback):
    """
    Invokes the pre_stage, post_stage, pre_epoch, and post_epoch methods of the Stage.
    """

    def pre_stage(self, stage):
        stage.pre_stage()

    def post_stage(self, stage):
        stage.post_stage()

    def pre_epoch(self, stage):
        stage.pre_epoch()

    def post_epoch(self, stage):
        stage.post_epoch()

    def post_step(self, stage):
        stage.post_step()


class Stage:
    """
    Hook Points:
        - pre_stage()
        - post_stage()
        - pre_epoch()
        - post_epoch()
    """

    def __init__(self, name: str = None, epochs: int | None = 1):
        self.name = name or self.__class__.__name__
        self.max_epochs = epochs

        self.callbacks = CallbackList()

        self.pipe = None  # set by the pipeline

        self.history = TrainingHistory()
        self.metrics = Tracker()

        self.step = 0
        self.global_step = 0

        self.metric_prefix = None
        self.barrier_timeout = None

        self._timer = TimerCallback()
        self._table_callback = TableCallback()
        self._reduce_metrics_callback = ReduceMetricsCallback()
        self._forward_callback = _ForwardCallback()
        self._profiler_callback = None
        self.add_callback(self._timer, CbPriority.STAGE_TIMER)
        self.add_callback(self._reduce_metrics_callback, CbPriority.METRIC_REDUCTION)
        self.add_callback(self._table_callback, CbPriority.TABLE)
        self.add_callback(self._forward_callback, CbPriority.OBJECT_METHODS)  # methods have priority 0

    @property
    def device(self):
        """
        Same as :attr:`Pipeline.device`.
        """
        return self.pipe.device

    @property
    def config(self):
        """
        Same as :attr:`Pipeline.config`.
        """
        return self.pipe.config

    @property
    def run_dir(self):
        """
        Same as :attr:`Pipeline.run_dir`.
        """
        return self.pipe.run_dir

    @property
    def current_epoch(self):
        return self.history.num_steps

    @property
    def start_time(self):
        return self._timer.start_time

    @property
    def end_time(self):
        return self._timer.end_time

    @property
    def epoch_start_time(self):
        return self._timer.epoch_start_time

    @property
    def epoch_end_time(self):
        return self._timer.epoch_end_time

    @property
    def table(self):
        return self._table_callback.get_table(self)

    @property
    def has_profiler(self) -> bool:
        """
        Returns True if the profiler is enabled for this stage, otherwise False.
        """
        return self._profiler_callback is not None

    @property
    def profiler(self) -> torch.profiler.profile | None:
        """
        If enabled, returns the profiler object associated with this stage, otherwise None.

        Returns:
            torch.profiler.profile or None: The profiler object.
        """
        if not self.has_profiler:
            return None
        return self._profiler_callback.profiler

    @property
    def _run_overridden(self):
        return type(self).run != Stage.run

    @property
    def _run_epoch_overridden(self):
        return type(self).run_epoch != Stage.run_epoch

    def add_callback(self, callback: 'Callback', priority: int = 1):
        """
        Adds a callback to this stage.

        Callbacks are executed based on their priority, with lower values being executed first.
        Callbacks with the same priority are executed in the order they were added.

        The pre_stage, post_stage, pre_epoch, and post_epoch methods are treated as callbacks with priority 0.

        Args:
            callback (StageCallback): The callback to add.
            priority (int, optional): The priority of the callback. Defaults to 1.
        """
        self.callbacks.append(callback, priority)

    def log(self, name: str, value: Any, reduction: str = 'mean', prefixed: bool = True):
        if prefixed and self.metric_prefix:
            name = f'{self.metric_prefix}/{name}'
        self.metrics.log(name, value, reduction)

    def add_metric(self, name, metric):
        metric = metric.to(self.device)
        self.metrics.add_metric(name, metric)
        return metric

    def add_column(
        self,
        name: str,
        metric: str | None = None,
        formatter: Callable | None = None,
        width: int | None = None,
        color: str | None = None,
        alignment: str | None = None,
    ):
        """
        Adds a column to the table.

        If metric is provided, the column will be updated with the latest value of the metric.
        Otherwise,the caller must update the value manually using `table.update`.

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
        self._table_callback.track_metric(
            self, name, metric=metric, formatter=formatter, width=width, color=color, alignment=alignment
        )

    def enable_profiler(self, epochs: list | None = [0], schedule=None):
        """
        Enables the profiler for this stage.

        If the `schedule` argument is not provided, the following default schedule is used:
        ```
        schedule = torch.profiler.schedule(
            wait=5,
            warmup=10,
            active=5,
            repeat=1,
        )
        ```

        The user must call `self.profiler.step()` on the root rank at the end of each iteration to advance the profiler.

        Args:
            epochs (list, optional): The epochs to profile. Defaults to [0]. If None, the profiler is enabled for all epochs.
            schedule: The schedule for the profiler, i.e. the object returned by torch.profiler.schedule(). If None, a default schedule is used. Defaults to None.
        """
        if not is_root():
            return

        if self.has_profiler:
            raise ValueError('Profiler is already enabled for this stage.')

        if schedule is None:
            schedule = torch.profiler.schedule(
                wait=10,
                warmup=10,
                active=5,
                repeat=1,
            )

        self._profiler_callback = ProfilerCallback(epochs=epochs, schedule=schedule)
        self.add_callback(self._profiler_callback, CbPriority.PROFILER)

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
        Executed after each epoch.
        """
        pass

    def post_step(self):
        """
        Executed after each step. Stage must call `finish_step()` at the end of each step.
        """
        pass

    def run():
        """
        Override this method to implement the main logic of the stage and do manual epoch management.

        Either this method or :meth:`run_epoch` must be implemented by subclasses.
        Unlike :meth:`run_epoch`, this method is called only once per stage, and the implementation is responsible for
        managing the epochs and calling :meth:`next_epoch` when appropriate.
        """
        raise NotImplementedError()

    def next_epoch(self):
        """
        Advances the stage to the next epoch.

        This method must only be called by the implementation of :meth:`run` when the stage finishes an epoch.
        """
        if self._run_epoch_overridden:
            raise ValueError('next_epoch() must not be called when run_epoch() is implemented.')

        self._post_epoch()
        self._pre_epoch()

    def finish_step(self):
        self._post_step()
        self.step += 1
        self.global_step += 1

    def run_epoch(self):
        """
        Override this method to implement the main logic of the stage for a single epoch.

        Either this method or :meth:`run` must be implemented by subclasses.
        Unlike :meth:`run`, this method is called automatically by the stage and does not need to manage the epochs.
        """
        raise NotImplementedError()

    def _run(self):
        """
        Runs this stage. Either until max_epochs are reached, or until stop_stage() is called.
        """
        if self._run_overridden and self._run_epoch_overridden:
            raise ValueError('Only one of run() or run_epoch() must be implemented.')
        elif not self._run_overridden and not self._run_epoch_overridden:
            raise ValueError('Either run() or run_epoch() must be implemented.')
        elif self._run_epoch_overridden:
            self._pre_stage()
            while self.max_epochs is None or self.current_epoch < self.max_epochs:
                self._pre_epoch()
                self.run_epoch()
                self._post_epoch()
            self._post_stage()
        else:
            self._pre_stage()
            self._pre_epoch()
            self.run()
            self._post_epoch()
            self._post_stage()

    def _pre_stage(self):
        if len(self.pipe.stages) > 1:
            dml_logging.info(f'\n========== STAGE: {self.name} ==========')

        callbacks = self.callbacks + self.pipe.callbacks
        for callback in callbacks:
            callback.pre_stage(self)

        dml_logging.flush_logger()
        self.pipe.barrier(self.barrier_timeout)

    def _post_stage(self):
        callbacks = self.callbacks + self.pipe.callbacks
        for callback in callbacks:
            callback.post_stage(self)
        self.pipe.barrier(self.barrier_timeout)

    def _pre_epoch(self):
        callbacks = self.callbacks + self.pipe.callbacks
        for callback in callbacks:
            callback.pre_epoch(self)

    def _post_epoch(self):
        callbacks = self.callbacks + self.pipe.callbacks
        for callback in callbacks:
            callback.post_epoch(self)

    def _post_step(self):
        callbacks = self.callbacks + self.pipe.callbacks
        for callback in callbacks:
            callback.post_step(self)
