from collections import namedtuple
from typing import Any, Union

import numpy as np
import torch
import torchmetrics
from numpy.typing import ArrayLike


__all__ = [
    'TrainingHistory',
    'Tracker',
]


class TrainingHistory:
    """
    Stores the training history of a model.

    Metrics can either be ArrayLike objects or any pickleable object.

    Usage:
        history = TrainingHistory()
        history.append_metric('loss', 0.5)
        history.append_metric('accuracy', 0.99)
        history.next_step()

        for metric in history:
            print(f'{metric}': history[metric])
    """

    max_return_type = namedtuple('Max', ['value', 'step'])
    min_return_type = namedtuple('Min', ['value', 'step'])

    def __init__(self):
        self.num_steps = 0
        self._metrics = {}
        self._dtypes = {}

    def __getitem__(self, name: str):
        if name not in self._metrics:
            raise KeyError(f'Metric {name} does not exist')

        return np.stack(self._metrics[name], axis=0, dtype=self._dtypes[name])

    def __delattr__(self, name):
        del self._metrics[name]

    def __contains__(self, name: str):
        return name in self._metrics

    def __len__(self):
        return len(self._metrics)

    def __iter__(self):
        return iter(self._metrics)

    def keys(self):
        return self._metrics.keys()

    def values(self):
        return [self[name] for name in self._metrics]

    def items(self):
        return [(name, self[name]) for name in self._metrics]

    def rows(self):
        for i in range(self.num_steps):
            yield {name: self._metrics[name][i] for name in self._metrics}

    def append_metric(self, name: str, value: Union[ArrayLike, Any]):
        """
        Adds a value for a metric at the current step.

        Args:
            name (str): The name of the metric.
            value (ArrayLike, Any): The value of the metric. Must be a ArrayLike or pickleable object.
        """
        if name in self._current_values:
            raise ValueError(f'Metric {name} already has a value for step {self.num_steps}')

    def append_metrics(self, **metrics):
        """
        Adds multiple metrics at the current step.

        Args:
            **metrics: The metrics to add.
        """
        for name, value in metrics.items():
            dtype = value.dtype if type(value) == ArrayLike else object  # noqa
            if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                value = value.item()

            if name not in self._metrics:
                self._metrics[name] = ([None] * self.num_steps) + [value]
                self._dtypes[name] = dtype
            else:
                self._metrics[name].append(value)

        self.num_steps += 1

    def last(self) -> dict[str, Any]:
        """
        Returns the last value for each metric.

        Returns:
            dict[str, Any]: The last value for each metric.
        """

        return {name: values[-1] for name, values in self._metrics.items()}

    def min(self) -> dict[str, min_return_type]:
        """
        Returns a namedtuple (value, step) containing the minimum value and the corresponding step for each metric across all steps.

        Returns:
            dict[str, namedtuple]: The minimum value and the corresponding step for each metric.
        """
        argmin = {name: np.argmin(values, axis=0) for name, values in self._metrics.items()}
        return {name: self.min_return_type(self._metrics[name][idx], idx) for name, idx in argmin.items()}

    def max(self) -> dict[str, max_return_type]:
        """
        Returns a namedtuple (value, step) containing the maximum value and the corresponding step for each metric across all steps.

        Returns:
            dict[str, namedtuple]: The maximum value and the corresponding step for each metric.
        """
        argmax = {name: np.argmax(values, axis=0) for name, values in self._metrics.items()}
        return {name: self.max_return_type(self._metrics[name][idx], idx) for name, idx in argmax.items()}


class Tracker(torch.nn.Module):
    """
    Keeps track of multiple metrics and reduces them at the end of each epoch.
    """

    def __init__(self):
        super().__init__()

        self.metrics = torch.nn.ModuleDict()

    def add_metric(self, name: str, metric: torchmetrics.Metric):
        if name in self.metrics:
            raise ValueError(f'Metric {name} already exists')

        self.metrics[name] = metric

    def log(self, name: str, value: Any, reduction: str = 'mean', **kwargs):
        if reduction not in ['mean', 'sum', 'min', 'max', 'cat']:
            raise ValueError(f'Invalid reduction {reduction}. Must be one of mean, sum, min, max, cat')

        if not torch.is_tensor(value):
            value = torch.tensor(value)
        value = value.cpu()
        dtype = value.dtype

        if name not in self.metrics:
            if reduction == 'mean':
                metric = torchmetrics.MeanMetric(**kwargs)
                dtype = torch.float32
            elif reduction == 'sum':
                metric = torchmetrics.SumMetric(**kwargs)
            elif reduction == 'min':
                metric = torchmetrics.MinMetric(**kwargs)
            elif reduction == 'max':
                metric = torchmetrics.MaxMetric(**kwargs)
            elif reduction == 'cat':
                metric = torchmetrics.CatMetric(**kwargs)
            metric = metric.cpu().set_dtype(dtype)
            self.add_metric(name, metric)

        self.metrics[name].update(value)

    def reduce(self, reset: bool = True):
        values = {}
        for name, metric in self.metrics.items():
            if metric.update_called:
                values[name] = metric.compute()
                if reset:
                    metric.reset()
            else:
                values[name] = None
        return values

    def clear(self):
        for metric in self.metrics.values():
            metric.reset()
        self.metrics.clear()

    def __getitem__(self, name: str):
        return self.metrics[name]

    def __setitem__(self, name: str, metric: torchmetrics.Metric):
        self.add_metric(name, metric)
