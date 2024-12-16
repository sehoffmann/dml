from enum import Enum

import torch
import torch.distributed as dist


__all__ = [
    'Reduction',
    'reduce_tensor',
    'MetricReducer',
    'MetricTracker',
]


class Reduction(Enum):
    """
    Reduction operation
    """

    MEAN = 'MEAN'
    SUM = 'SUM'
    MIN = 'MIN'
    MAX = 'MAX'

    def as_torch(self):
        """
        Returns the corresponding torch.distribution.ReduceOp
        """

        if self == Reduction.SUM:
            return dist.ReduceOp.SUM
        elif self == Reduction.MIN:
            return dist.ReduceOp.MIN
        elif self == Reduction.MAX:
            return dist.ReduceOp.MAX
        else:
            raise ValueError(f'Reduction {self} is not supported by torch')


def reduce_tensor(tensor: torch.Tensor, reduction: Reduction, dim=None):
    """
    Reduces tensor along dim with the given reduction.
    """

    if not isinstance(tensor, torch.Tensor):
        raise ValueError('tensor must be a torch.Tensor')

    # required because dim=None is not supported by torch
    if dim is None:
        dim = list(range(tensor.dim()))

    if reduction is Reduction.MEAN:
        return tensor.mean(dim)
    elif reduction is Reduction.SUM:
        return tensor.sum(dim)
    elif reduction is Reduction.MIN:
        return tensor.amin(dim)
    elif reduction is Reduction.MAX:
        return tensor.amax(dim)
    else:
        raise ValueError(f'Unknown reduction {reduction}')


class MetricReducer:
    """
    Stores a list of tensors and reduces them at the end of an epoch.
    The dim argument specifies the dimensions to reduce over. If None, every dimension is completely reduced.
    Notice that the list of individual tensors stored in this obcect, is ALWAYS reduced, both locally and distributed.
    Hence, dimension 0 refers to the first dimension of individual tensors, which is usually the batch dimension.
    """

    def __init__(self, reduction=Reduction.MEAN, dim=None, globally=True):
        if reduction not in [Reduction.MEAN, Reduction.SUM, Reduction.MIN, Reduction.MAX]:
            raise ValueError(f'Unknown reduction {self.reduction}')

        self.values = []
        self.reduction = reduction
        self.globally = globally
        if isinstance(dim, int):
            self.dim = [dim]
        elif dim is not None:
            self.dim = list(dim)
        else:
            self.dim = None

    def append(self, value):
        """
        Appends a value to the list of values.
        If the value is a tensor, it is detached and moved to the cpu to avoid growing memory consumption.
        """
        value = torch.as_tensor(value)
        value = value.detach().cpu()
        self.values.append(value)

    def extend(self, values):
        for value in values:
            self.append(value)

    def __iadd__(self, value):
        self.append(value)
        return self

    def __setitem__(self, idx, value):
        value = torch.as_tensor(value)
        value = value.detach().cpu()
        self.values[idx] = value

    def __getitem__(self, idx):
        return self.values[idx]

    def __delitem__(self, idx):
        del self.values[idx]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def clear(self):
        self.values.clear()

    def reduce_and_append(self, value):
        value = reduce_tensor(value, self.reduction, dim=self.dim)
        self.values.append(value)

    def reduce_locally(self):
        if len(self.values) == 0:
            return None

        if isinstance(self.dim, list):
            dim = [0] + [d + 1 for d in self.dim]
        elif isinstance(self.dim, int):
            dim = [0, self.dim + 1]
        else:
            dim = None
        tensor = torch.stack(self.values)
        tensor = reduce_tensor(tensor, reduction=self.reduction, dim=dim)
        return tensor

    def reduce_globally(self, group=None):
        # if the list of values is empty, the result is None
        if self.globally:
            empty_workers = [None] * dist.get_world_size(group)
            dist.all_gather_object(empty_workers, len(self.values) == 0, group=group)
            if any(empty_workers):
                if len(empty_workers) > 1 and not all(empty_workers):
                    raise ValueError('Some workers tracked values this epoch and some did not. This is likely a bug.')
                else:
                    return None
        elif len(self.values) == 0:
            return None

        tensor = self.reduce_locally()
        if self.globally:
            if self.reduction == Reduction.MEAN:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
                tensor /= dist.get_world_size(group)
            else:
                dist.all_reduce(tensor, op=self.reduction.as_torch(), group=group)
        return tensor

    def state_dict(self):
        return {
            'reduction': self.reduction,
            'dim': self.dim,
            'globally': self.globally,
            'values': self.values,
        }

    def load_state_dict(self, state):
        self.reduction = state['reduction']
        self.dim = state['dim']
        self.globally = state['globally']
        self.values = state['values']


class MetricTracker:
    """
    This class keeps track of multiple metrics and their history.

    Usage:
        tracker = MetricTracker()
        tracker.register_metric('loss', reduction=Reduction.MEAN)
        tracker.track('loss', torch.randn(10, 1))
        tracker.next_epoch()

        print(tracker['loss'].last())
    """

    def __init__(self):
        self.histories = {}
        self.reducers = {}
        self.epoch = 1

    def __getitem__(self, name):
        """
        Returns the history of a metric up to the current epoch.
        Values for the current epoch that have been reduced already are not included.
        """
        if name not in self:
            raise ValueError(f'Metric {name} does not exist')
        return list(self.histories[name])[: self.epoch - 1]

    def __contains__(self, name):
        return name in self.histories

    def __len__(self):
        return len(self.histories)

    def __iter__(self):
        return iter(self.histories)

    def current_value(self, name):
        """
        If the metric already has an reduced value for the current epoch, it is returned. Otherwise, None is returned.
        """
        if name not in self:
            raise ValueError(f'Metric {name} does not exist')
        if self.has_value(name):
            return self.histories[name][-1]
        else:
            return None

    def is_reduced_metric(self, name):
        """
        Returns True if the metric gets (all)reduced at the end of each epoch.
        """
        if name not in self:
            raise ValueError(f'Metric {name} does not exist')
        return name in self.reducers

    def has_value(self, name):
        """
        Returns True if the metric has a final value for the current epoch.
        """
        if name not in self:
            raise ValueError(f'Metric {name} does not exist')
        return len(self.histories[name]) >= self.epoch

    def register_metric(self, name, reduction=None, dim=None, globally=True):
        if name in self:
            raise ValueError(f'Metric {name} already exists')

        if dim is not None and reduction is None:
            raise ValueError('If dim is specified, reduction must be specified as well')

        self.histories[name] = [] + [None] * (self.epoch - 1)
        if reduction is not None:
            self.reducers[name] = MetricReducer(reduction=reduction, dim=dim, globally=globally)

    def track(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().to('cpu', non_blocking=True)

        if name not in self:
            raise ValueError(f'Metric {name} does not exist')

        if self.has_value(name):
            raise ValueError(f'History for {name} already has a value for epoch {self.epoch}')

        history = self.histories[name]
        reducer = self.reducers.get(name)
        if reducer is not None:
            reducer.append(value)
        else:
            history.append(value)

    def reduce_all(self, prefix=None, strict=True):
        """
        Reduces all metrics and appends their reduced values to the history.
        If prefix is specified, only metrics with the specified prefix are reduced.
        If strict is True, an error is raised if a metric has already been reduced for the current epoch.

        After this method has been called, no more values for the reduced metrics can be tracked for the current epoch,
        and next_epoch() must be called to be able to track new values.
        """
        for name, history in self.histories.items():
            if prefix is not None and not name.startswith(prefix):
                continue

            if self.has_value(name):
                if strict:
                    raise ValueError(f'History for {name} has already been reduced for epoch {self.epoch}')
                else:
                    continue

            reducer = self.reducers.get(name)
            if reducer is not None:
                history.append(reducer.reduce_globally())
                reducer.clear()
            else:
                history.append(None)

    def next_epoch(self):
        """
        Reduces all metrics (if not already reduced) and advances the epoch counter.
        """
        self.reduce_all(strict=False)
        self.epoch += 1

    def state_dict(self):
        state = {
            'epoch': self.epoch,
            'histories': dict(self.histories),
            'reducers': {name: reducer.state_dict() for name, reducer in self.reducers.items()},
        }
        return state

    def load_state_dict(self, state):
        self.epoch = state['epoch']
        self.histories = state['histories']
        self.reducers = {}
        for name, reducer_state in state['reducers'].items():
            self.reducers[name] = MetricReducer()
            self.reducers[name].load_state_dict(reducer_state)

    def __str__(self):
        s = 'MetricTracker('
        for name, history in self.histories.items():
            s += f'\n  {name}: {history}'
        if len(self.histories) > 0:
            s += '\n)'
        else:
            s += ')'
        return s
