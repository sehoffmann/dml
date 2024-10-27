import sys

import pytest
import torch
from dmlcloud.core.metrics import MetricReducer, MetricTracker, Reduction


class TestMetricReducer:
    def test_local_reduction(self):
        reducer = MetricReducer(reduction=Reduction.MIN, globally=False)
        reducer.append(torch.tensor([1, 2, 3], dtype=torch.float))
        reducer.append(torch.tensor([-1, -2, -3], dtype=torch.float))
        reducer.append(torch.tensor([1, 7, 10], dtype=torch.float))

        assert reducer.reduce_locally().item() == -3
        assert reducer.reduce_globally().item() == -3

        reducer.reduction = Reduction.MAX
        assert reducer.reduce_locally().item() == 10
        assert reducer.reduce_globally().item() == 10

        reducer.reduction = Reduction.SUM
        assert reducer.reduce_locally().item() == 18
        assert reducer.reduce_globally().item() == 18

        reducer.reduction = Reduction.MEAN
        assert reducer.reduce_locally().item() == 2
        assert reducer.reduce_globally().item() == 2

    def test_global_reduction(self, torch_distributed):
        reducer = MetricReducer(reduction=Reduction.MIN, globally=True)
        reducer.append(torch.tensor([1, 2, 3], dtype=torch.float))
        reducer.append(torch.tensor([-1, -2, -3], dtype=torch.float))
        reducer.append(torch.tensor([1, 7, 10], dtype=torch.float))

        assert reducer.reduce_locally().item() == -3
        assert reducer.reduce_globally().item() == -3

        reducer.reduction = Reduction.MAX
        assert reducer.reduce_locally().item() == 10
        assert reducer.reduce_globally().item() == 10

        reducer.reduction = Reduction.SUM
        assert reducer.reduce_locally().item() == 18
        assert reducer.reduce_globally().item() == 18

        reducer.reduction = Reduction.MEAN
        assert reducer.reduce_locally().item() == 2
        assert reducer.reduce_globally().item() == 2

    def test_partial_reduction(self):
        tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=torch.float)  # shape: 2x2x3
        print(tensor.shape)

        reducer = MetricReducer(reduction=Reduction.MIN, globally=False, dim=[1, 2])
        reducer.append(tensor)
        result = reducer.reduce_locally()
        assert result.shape == (2,)
        assert result[0].item() == 1
        assert result[1].item() == 1

        reducer = MetricReducer(reduction=Reduction.SUM, globally=False, dim=2)
        reducer.append(tensor)
        result = reducer.reduce_locally()
        assert result.shape == (2, 2)
        assert result[0, 0].item() == 6
        assert result[0, 1].item() == 15
        assert result[1, 0].item() == 6
        assert result[1, 1].item() == 15

    def test_serialization(self):
        reducer = MetricReducer(reduction=Reduction.MIN, dim=(1, 2, 3))
        reducer.append(torch.tensor([1, 2, 3]))
        state_dict = reducer.state_dict()

        new_reducer = MetricReducer()
        new_reducer.load_state_dict(state_dict)
        assert new_reducer.reduction == Reduction.MIN
        assert new_reducer.dim == [1, 2, 3]
        assert new_reducer.values == reducer.values

    def test_empty_reduction(self, torch_distributed):
        reducer = MetricReducer(reduction=Reduction.MIN, globally=True)
        result = reducer.reduce_locally()
        assert result is None

        result = reducer.reduce_globally()
        assert result is None


class TestMetricTracker:
    def test_dictionary(self):
        tracker = MetricTracker()
        assert len(tracker) == 0

        tracker.register_metric('A')
        tracker.register_metric('B', reduction=Reduction.MEAN, globally=False)
        assert len(tracker) == 2

        assert 'A' in tracker
        assert 'B' in tracker
        assert 'C' not in tracker

        assert isinstance(tracker['A'], list)
        assert len(tracker['A']) == 0

    def test_is_reduced_metric(self):
        tracker = MetricTracker()
        tracker.register_metric('A')
        tracker.register_metric('B', reduction=Reduction.MEAN, globally=False)

        assert not tracker.is_reduced_metric('A')
        assert tracker.is_reduced_metric('B')

    def test_epoch_filling(self):
        tracker = MetricTracker()
        tracker.register_metric('A')

        tracker.next_epoch()
        assert len(tracker['A']) == 1 and tracker['A'][0] is None
        assert tracker.epoch == 2

        tracker.next_epoch()
        assert len(tracker['A']) == 2 and tracker['A'][1] is None
        assert tracker.epoch == 3

        tracker.register_metric('B', reduction=Reduction.MEAN, globally=False)
        assert len(tracker['B']) == 2 and tracker['B'][1] is None

    def test_track(self):
        tracker = MetricTracker()
        tracker.register_metric('A')

        tracker.track('A', 1)
        with pytest.raises(ValueError):  # haven't progressed the epoch yet
            tracker.track('A', 42)
        tracker.next_epoch()

        tracker.track('A', 42)

        tracker.register_metric('B', reduction=Reduction.MEAN, globally=False)
        tracker.track('B', 2.0)
        tracker.track('B', 4.0)
        tracker.track('B', 1.0)
        tracker.track('B', 1.0)

        tracker.next_epoch()
        assert tracker['A'] == [1, 42]
        assert tracker['B'] == [None, torch.tensor(2.0)]

    def test_str(self):
        tracker = MetricTracker()
        tracker.register_metric('A')
        tracker.register_metric('B', reduction=Reduction.MEAN, globally=False)
        tracker.track('A', 1)
        print(str(tracker))

    def test_manual_reduction(self):
        tracker = MetricTracker()
        tracker.register_metric('A')
        tracker.register_metric('B', reduction=Reduction.SUM, globally=False)
        tracker.track('B', 1.0)
        tracker.track('B', 2.0)
        tracker.track('B', 3.0)
        tracker.reduce_all(prefix='B')

        assert tracker.has_value('B')
        assert not tracker.has_value('A')
        assert tracker.current_value('B').item() == 6.0
        assert tracker.current_value('A') is None
        assert tracker['B'] == []

        with pytest.raises(ValueError):
            tracker.reduce_all(prefix='B')

        # does not throw, nor modify value
        tracker.reduce_all(prefix='B', strict=False)
        assert tracker.current_value('B').item() == 6.0
        assert tracker['B'] == []

        # advances epoch
        tracker.next_epoch()
        assert tracker['B'] == [torch.tensor(6.0)]
        assert tracker['A'] == [None]
        assert tracker.current_value('B') is None

    def test_serialization(self):
        tracker1 = MetricTracker()
        tracker1.register_metric('A')
        tracker1.register_metric('B', reduction=Reduction.MEAN, globally=False)

        tracker1.track('A', 1)
        tracker1.track('B', torch.randn(3, 2))
        tracker1.next_epoch()
        tracker1.track('A', 2)
        tracker1.track('B', torch.randn(3, 2))

        state_dict = tracker1.state_dict()
        tracker2 = MetricTracker()
        tracker2.load_state_dict(state_dict)
        assert tracker2.epoch == tracker1.epoch
        assert 'A' in tracker2 and 'B' in tracker2
        assert tracker2['A'] == tracker1['A']
        assert tracker2['B'] == tracker1['B']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
