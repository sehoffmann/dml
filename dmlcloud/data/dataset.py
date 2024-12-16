from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

import torch.distributed as dist
from torch.utils.data import get_worker_info, IterableDataset

from .sharding import shard_sequence


__all__ = [
    'ShardedSequenceDataset',
    'DownstreamDataset',
    'PrefetchDataset',
    'BatchDataset',
]


class ShardedSequenceDataset(IterableDataset):
    def __init__(
        self,
        sequence: Sequence,
        shuffle: bool = False,
        even_shards: bool = True,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        self.sequence = sequence
        self.shuffle = shuffle
        self.even_shards = even_shards
        self.seed = seed
        self.rank = rank if rank is not None else dist.get_rank()
        self.world_size = world_size if world_size is not None else dist.get_world_size()
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            rank = self.rank
            world_size = self.world_size
        else:
            rank = self.rank * worker_info.num_workers + worker_info.id
            world_size = self.world_size * worker_info.num_workers
        shards = shard_sequence(
            self.sequence,
            rank,
            world_size,
            shuffle=self.shuffle,
            even_shards=self.even_shards,
            seed=self.seed + self.epoch,
        )
        return iter(shards)


class DownstreamDataset(IterableDataset):
    def __init__(self, source_ds: Iterable):
        self.source_ds = source_ds

    def set_epoch(self, epoch: int):
        if hasattr(self.source_ds, 'set_epoch'):
            self.source_ds.set_epoch(epoch)

    def __len__(self):
        return len(self.source_ds)


class PrefetchDataset(DownstreamDataset):
    def __init__(self, source_ds: Iterable, num_elements: int):
        super().__init__(source_ds)
        self.num_elements = num_elements

    def __iter__(self):
        pool = ThreadPoolExecutor(max_workers=1)
        iter_ = iter(self.source_ds)

        with pool:
            futures = [pool.submit(next, iter_) for _ in range(self.num_elements)]
            while True:
                future = futures.pop(0)
                try:
                    element = future.result()
                except StopIteration:
                    return
                futures += [pool.submit(next, iter_)]
                yield element


class BatchDataset(DownstreamDataset):
    def __init__(self, source_ds: Iterable, batch_size: int, drop_remainder: bool = False):
        super().__init__(source_ds)
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder

    def __len__(self):
        if self.drop_remainder:
            return len(self.source_ds) // self.batch_size
        else:
            return (len(self.source_ds) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch = []
        for element in self.source_ds:
            batch.append(element)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_remainder:
            yield batch
