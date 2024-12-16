"""Utilities for sharding data across multiple workers."""


from typing import Sequence

import numpy as np

__all__ = [
    'shard_indices',
    'shard_sequence',
    'chunk_and_shard_indices',
]


def shard_indices(
    num_elements: int,
    rank: int,
    world_size: int,
    shuffle: bool = False,
    even_shards: bool = True,
    seed: int = 0,
) -> list[int]:
    """
    even_shards: If True, every worker receives the same number of shards, and the last shards are dropped.
    """
    indices = np.arange(num_elements)

    if shuffle:
        np.random.Generator(np.random.MT19937(seed)).shuffle(indices)

    if even_shards:
        indices = indices[: num_elements - num_elements % world_size]

    return indices[rank::world_size].tolist()  # this also converts np.int64 to python's int


def shard_sequence(
    sequence: Sequence,
    rank: int,
    world_size: int,
    shuffle: bool = False,
    even_shards: bool = True,
    seed: int = 0,
):
    indices = shard_indices(len(sequence), rank, world_size, shuffle=shuffle, even_shards=even_shards, seed=seed)
    return [sequence[i] for i in indices]


def chunk_and_shard_indices(
    num_elements: int,
    chunk_size: int,
    rank: int,
    world_size: int,
    chunk_overlap: int = 0,
    even_shards: bool = True,
    equal_chunks: bool = True,
    shuffle: bool = False,
    seed: int = 0,
):
    if equal_chunks:
        num_chunks = num_elements // chunk_size
    else:
        num_chunks = (num_elements + chunk_size - 1) // chunk_size

    chunk_indices = shard_indices(num_chunks, rank, world_size, shuffle=shuffle, even_shards=even_shards, seed=seed)
    chunks = []
    for chunk_idx in chunk_indices:
        start = chunk_idx * chunk_size
        end = start + chunk_size + chunk_overlap
        chunks.append((start, end))
    return chunks
