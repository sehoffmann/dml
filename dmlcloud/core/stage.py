from typing import Any, Callable, TYPE_CHECKING

from . import logging as dml_logging
from .callbacks import ReduceMetricsCallback, TableCallback, TimerCallback
from .metrics import Tracker, TrainingHistory

if TYPE_CHECKING:
    from .callbacks import Callback

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

    def __init__(self, name: str = None, epochs: int = 1):
        self.name = name or self.__class__.__name__
        self.max_epochs = epochs

        self.callbacks: list[Callback] = []

        self.pipe = None  # set by the pipeline

        self.history = TrainingHistory()
        self.tracker = Tracker()

        self._timer = TimerCallback()
        self.add_callback(self._timer)

        self.add_callback(ReduceMetricsCallback())

        self._table_callback = TableCallback()
        self.add_callback(self._table_callback)

        self.metric_prefix = None
        self.barrier_timeout = None

    @property
    def device(self):
        return self.pipe.device

    @property
    def config(self):
        return self.pipe.config

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
        return self._table_callback.table

    def add_callback(self, callback: 'Callback'):
        """
        Adds a callback to this stage.

        Callbacks are executed in the order they are added and after the stage-specific hooks.

        Args:
            callback (StageCallback): The callback to add.
        """
        self.callbacks.append(callback)

    def log(self, name: str, value: Any, reduction: str = 'mean', prefixed: bool = True):
        if prefixed and self.metric_prefix:
            name = f'{self.metric_prefix}/{name}'
        self.tracker.log(name, value, reduction)

    def add_metric(self, name, metric):
        metric = metric.to(self.device)
        self.tracker.add_metric(name, metric)
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
            name, metric=metric, formatter=formatter, width=width, color=color, alignment=alignment
        )

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
        if len(self.pipe.stages) > 1:
            dml_logging.info(f'\n========== STAGE: {self.name} ==========')

        self.pre_stage()

        for callback in self.callbacks:
            callback.pre_stage(self)

        dml_logging.flush_logger()

        self.pipe.barrier(self.barrier_timeout)

    def _post_stage(self):
        self.post_stage()
        for callback in self.callbacks:
            callback.post_stage(self)

        self.pipe.barrier(self.barrier_timeout)

    def _pre_epoch(self):
        self.pre_epoch()
        for callback in self.callbacks:
            callback.pre_epoch(self)

    def _post_epoch(self):
        self.post_epoch()
        for callback in self.callbacks:
            callback.post_epoch(self)
