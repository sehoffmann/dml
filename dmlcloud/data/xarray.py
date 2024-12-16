from typing import Iterable

import torch.distributed as dist
import xarray as xr
from torch.utils.data import get_worker_info, IterableDataset

from .sharding import chunk_and_shard_indices


__all__ = [
    'sharded_xr_dataset',
    'ShardedXrDataset',
]


def sharded_xr_dataset(
    ds: xr.Dataset | xr.DataArray,
    dim: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    even_shards: bool = True,
    equal_chunks: bool = True,
    shuffle: bool = False,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    process_group: dist.ProcessGroup | None = None,
    load: bool = False,
    load_kwargs: dict | None = None,
) -> Iterable[xr.Dataset | xr.DataArray]:
    if rank is None:
        rank = dist.get_rank(process_group)
    if world_size is None:
        world_size = dist.get_world_size(process_group)

    num_elements = len(ds[dim])
    chunks = chunk_and_shard_indices(
        num_elements,
        chunk_size,
        rank,
        world_size,
        chunk_overlap=chunk_overlap,
        even_shards=even_shards,
        equal_chunks=equal_chunks,
        shuffle=shuffle,
        seed=seed,
    )
    for start, end in chunks:
        chunk = ds.isel({dim: slice(start, end)})
        if load:
            kwargs = load_kwargs or {}
            chunk.load(**kwargs)
        yield chunk


class ShardedXrDataset(IterableDataset):
    def __init__(
        self,
        ds: xr.Dataset | xr.DataArray,
        dim: str,
        chunk_size: int,
        chunk_overlap: int = 0,
        even_shards: bool = True,
        equal_chunks: bool = True,
        shuffle: bool = False,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        process_group: dist.ProcessGroup | None = None,
        load: bool = False,
        load_kwargs: dict | None = None,
    ):
        self.ds = ds
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.even_shards = even_shards
        self.equal_chunks = equal_chunks
        self.shuffle = shuffle
        self.seed = seed
        self.load = load
        self.load_kwargs = load_kwargs

        self.rank = rank if rank is not None else dist.get_rank(process_group)
        self.world_size = world_size if world_size is not None else dist.get_world_size(process_group)
        self._num_iters = 0

    def set_epoch(self, epoch: int):
        self._num_iters = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            rank = self.rank
            world_size = self.world_size
        else:
            rank = self.rank * worker_info.num_workers + worker_info.id
            world_size = self.world_size * worker_info.num_workers

        return sharded_xr_dataset(
            self.ds,
            self.dim,
            self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            even_shards=self.even_shards,
            equal_chunks=self.equal_chunks,
            shuffle=self.shuffle,
            seed=self.seed + self._num_iters,
            rank=rank,
            world_size=world_size,
            load=self.load,
            load_kwargs=self.load_kwargs,
        )
