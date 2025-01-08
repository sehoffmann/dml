import random
import sys

import dmlcloud as dml
import numpy as np
import pytest
import torch


def seed(seed=None):
    seed = dml.seed(seed)
    state = dict(
        seed=seed,
        torch_state=np.array(torch.get_rng_state()),
        numpy_state=np.random.get_state()[1],
        random_state=np.array(random.getstate()[1]),
    )
    return state


class TestSeed:
    def test_single_worker_deterministic(self, torch_distributed):
        prev_torch_state = np.array(torch.get_rng_state())
        prev_numpy_state = np.random.get_state()[1]
        prev_random_state = np.array(random.getstate()[1])

        states = seed(42)
        assert states['seed'] == 42
        assert (states['torch_state'] != prev_torch_state).any()
        assert (states['numpy_state'] != prev_numpy_state).any()
        assert (states['random_state'] != prev_random_state).any()

        # advance the RNG
        torch.randint(0, 10, (1,))
        np.random.randint(0, 10)

        # reseeding should reset the RNG
        new_states = seed(42)
        assert new_states['seed'] == 42
        assert (new_states['torch_state'] == states['torch_state']).all()
        assert (new_states['numpy_state'] == states['numpy_state']).all()
        assert (new_states['random_state'] == states['random_state']).all()

    def test_input_validation(self, torch_distributed):
        with pytest.raises(RuntimeError):
            dml.seed(2**80)
        assert dml.seed(2**64 - 1) == 2**64 - 1

    def test_single_worker_random(self, torch_distributed):
        prev_torch_state = np.array(torch.get_rng_state())
        prev_numpy_state = np.random.get_state()[1]
        prev_random_state = np.array(random.getstate()[1])

        states = seed()
        assert type(states['seed']) is int
        assert (states['torch_state'] != prev_torch_state).any()
        assert (states['numpy_state'] != prev_numpy_state).any()
        assert (states['random_state'] != prev_random_state).any()

        # reseeding should yield different states
        new_states = seed()
        assert new_states['seed'] != states['seed']
        assert (new_states['torch_state'] != states['torch_state']).any()
        assert (new_states['numpy_state'] != states['numpy_state']).any()
        assert (new_states['random_state'] != states['random_state']).any()

    def test_multi_worker_deterministic(self, distributed_environment):
        states = distributed_environment(4).start(seed, 42)
        assert [s['seed'] for s in states] == [42, 42, 42, 42]

        # workers should have different states
        assert all((s['torch_state'] != states[0]['torch_state']).any() for s in states[1:])
        assert all((s['numpy_state'] != states[0]['numpy_state']).any() for s in states[1:])
        assert all((s['random_state'] != states[0]['random_state']).any() for s in states[1:])

        # same seed should yield same states
        new_states = distributed_environment(4).start(seed, 42)
        assert [s['seed'] for s in new_states] == [42, 42, 42, 42]
        assert all((s1['torch_state'] == s2['torch_state']).all() for s1, s2 in zip(states, new_states))
        assert all((s1['numpy_state'] == s2['numpy_state']).all() for s1, s2 in zip(states, new_states))
        assert all((s1['random_state'] == s2['random_state']).all() for s1, s2 in zip(states, new_states))

        # different seed should yield different states
        new_states = distributed_environment(4).start(seed, 11)
        assert [s['seed'] for s in new_states] == [11, 11, 11, 11]
        assert all((s1['torch_state'] != s2['torch_state']).any() for s1, s2 in zip(states, new_states))
        assert all((s1['numpy_state'] != s2['numpy_state']).any() for s1, s2 in zip(states, new_states))
        assert all((s1['random_state'] != s2['random_state']).any() for s1, s2 in zip(states, new_states))

    def test_multi_worker_random(self, distributed_environment):
        # all workers should have same seeds
        states = distributed_environment(4).start(seed)
        assert [s['seed'] for s in states] == [states[0]['seed']] * 4

        # workers should have different states
        assert all((s['torch_state'] != states[0]['torch_state']).any() for s in states[1:])
        assert all((s['numpy_state'] != states[0]['numpy_state']).any() for s in states[1:])
        assert all((s['random_state'] != states[0]['random_state']).any() for s in states[1:])

        # reseeding should yield different states and seeds
        new_states = distributed_environment(4).start(seed)
        assert [s['seed'] for s in new_states] != [s['seed'] for s in states]
        assert all((s1['torch_state'] != s2['torch_state']).any() for s1, s2 in zip(states, new_states))
        assert all((s1['numpy_state'] != s2['numpy_state']).any() for s1, s2 in zip(states, new_states))
        assert all((s1['random_state'] != s2['random_state']).any() for s1, s2 in zip(states, new_states))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
