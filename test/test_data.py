import sys
from functools import partial

import numpy as np
import pytest
import torch
import xarray as xr
from dmlcloud.data import interleave_batches, shard_indices, sharded_xr_dataset, ShardedXrDataset
from numpy.testing import assert_array_equal
from torch.utils.data import DataLoader, IterableDataset


class _Unzip(IterableDataset):
    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        for chunk in self.ds:
            arr = chunk.to_array().values[0]
            yield from arr


class TestSharding:
    def test_types(self):
        indices = shard_indices(10, 0, 2, shuffle=False, even_shards=False)
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)

    def test_even(self):
        assert shard_indices(10, 0, 2, shuffle=False, even_shards=False) == [0, 2, 4, 6, 8]
        assert shard_indices(10, 1, 2, shuffle=False, even_shards=False) == [1, 3, 5, 7, 9]

    def test_uneven(self):
        assert shard_indices(10, 0, 3, shuffle=False, even_shards=False) == [0, 3, 6, 9]
        assert shard_indices(10, 1, 3, shuffle=False, even_shards=False) == [1, 4, 7]
        assert shard_indices(10, 2, 3, shuffle=False, even_shards=False) == [2, 5, 8]

        assert shard_indices(11, 0, 2, shuffle=False, even_shards=False) == [0, 2, 4, 6, 8, 10]
        assert shard_indices(11, 1, 2, shuffle=False, even_shards=False) == [1, 3, 5, 7, 9]

    def test_dropping(self):
        assert shard_indices(10, 0, 3, shuffle=False, even_shards=True) == [0, 3, 6]
        assert shard_indices(10, 1, 3, shuffle=False, even_shards=True) == [1, 4, 7]
        assert shard_indices(10, 2, 3, shuffle=False, even_shards=True) == [2, 5, 8]

        assert shard_indices(11, 0, 2, shuffle=False, even_shards=True) == [0, 2, 4, 6, 8]
        assert shard_indices(11, 1, 2, shuffle=False, even_shards=True) == [1, 3, 5, 7, 9]

    def test_shuffling(self):
        indices = shard_indices(10, 0, 2, shuffle=True, even_shards=False, seed=0)
        assert len(indices) == 5
        assert len(np.unique(indices)) == 5
        assert indices != list(sorted(indices))
        assert (np.array(indices) >= 0).all() and (np.array(indices) <= 9).all()


class TestShardedXr:
    def test_basic(self):
        ds = xr.DataArray(np.arange(100), dims=['x'], name='var').to_dataset()
        world_size = 3
        chunk_size = 15

        shard = partial(sharded_xr_dataset, ds, 'x', chunk_size, world_size=world_size, shuffle=False)
        chunks_1 = list(shard(rank=0))
        chunks_2 = list(shard(rank=1))
        chunks_3 = list(shard(rank=2))

        assert len(chunks_1) == 2
        assert len(chunks_2) == 2
        assert len(chunks_3) == 2

        assert isinstance(chunks_1[0], xr.Dataset)

        assert chunks_1[0].x.size == 15
        assert chunks_1[1].x.size == 15
        assert chunks_2[0].x.size == 15
        assert chunks_2[1].x.size == 15
        assert chunks_3[0].x.size == 15
        assert chunks_3[1].x.size == 15

        assert_array_equal(chunks_1[0]['var'], np.arange(0, 15))
        assert_array_equal(chunks_2[0]['var'], np.arange(15, 30))
        assert_array_equal(chunks_3[0]['var'], np.arange(30, 45))
        assert_array_equal(chunks_1[1]['var'], np.arange(45, 60))
        assert_array_equal(chunks_2[1]['var'], np.arange(60, 75))
        assert_array_equal(chunks_3[1]['var'], np.arange(75, 90))

    def test_uneven(self):
        ds = xr.DataArray(np.arange(100), dims=['x'], name='var').to_dataset()
        world_size = 3
        chunk_size = 20

        shard = partial(
            sharded_xr_dataset, ds, 'x', chunk_size, even_shards=False, world_size=world_size, shuffle=False
        )
        chunks_1 = list(shard(rank=0))
        chunks_2 = list(shard(rank=1))
        chunks_3 = list(shard(rank=2))

        assert len(chunks_1) == 2
        assert len(chunks_2) == 2
        assert len(chunks_3) == 1

        assert isinstance(chunks_1[0], xr.Dataset)

        assert chunks_1[0].x.size == 20
        assert chunks_1[1].x.size == 20
        assert chunks_2[0].x.size == 20
        assert chunks_2[1].x.size == 20
        assert chunks_3[0].x.size == 20

        assert_array_equal(chunks_1[0]['var'], np.arange(0, 20))
        assert_array_equal(chunks_2[0]['var'], np.arange(20, 40))
        assert_array_equal(chunks_3[0]['var'], np.arange(40, 60))
        assert_array_equal(chunks_1[1]['var'], np.arange(60, 80))
        assert_array_equal(chunks_2[1]['var'], np.arange(80, 100))

    def test_unequal(self):
        ds = xr.DataArray(np.arange(110), dims=['x'], name='var').to_dataset()
        world_size = 3
        chunk_size = 20

        shard = partial(
            sharded_xr_dataset, ds, 'x', chunk_size, equal_chunks=False, world_size=world_size, shuffle=False
        )
        chunks_1 = list(shard(rank=0))
        chunks_2 = list(shard(rank=1))
        chunks_3 = list(shard(rank=2))

        assert len(chunks_1) == 2
        assert len(chunks_2) == 2
        assert len(chunks_3) == 2

        assert isinstance(chunks_1[0], xr.Dataset)

        assert chunks_1[0].x.size == 20
        assert chunks_1[1].x.size == 20
        assert chunks_2[0].x.size == 20
        assert chunks_2[1].x.size == 20
        assert chunks_3[0].x.size == 20
        assert chunks_3[1].x.size == 10

        assert_array_equal(chunks_1[0]['var'], np.arange(0, 20))
        assert_array_equal(chunks_2[0]['var'], np.arange(20, 40))
        assert_array_equal(chunks_3[0]['var'], np.arange(40, 60))
        assert_array_equal(chunks_1[1]['var'], np.arange(60, 80))
        assert_array_equal(chunks_2[1]['var'], np.arange(80, 100))
        assert_array_equal(chunks_3[1]['var'], np.arange(100, 110))

    def test_shuffled(self):
        ds = xr.DataArray(np.arange(100), dims=['x'], name='var').to_dataset()
        world_size = 3
        chunk_size = 15

        shard = partial(sharded_xr_dataset, ds, 'x', chunk_size, world_size=world_size, shuffle=True, seed=0)
        chunks_1 = list(shard(rank=0))
        chunks_2 = list(shard(rank=1))
        chunks_3 = list(shard(rank=2))

        assert len(chunks_1) == 2
        assert len(chunks_2) == 2
        assert len(chunks_3) == 2

        catted = xr.concat(chunks_1 + chunks_2 + chunks_3, dim='x')['var'].values
        assert catted.tolist() != list(range(90))
        assert list(sorted(catted.tolist())) == list(range(90))

        chunk = chunks_1[0]['var'].values
        assert chunk.tolist() == list(range(chunk[0], chunk[-1] + 1))

    def test_XrShardedDataset_multiprocessing(self):
        xr_ds = xr.DataArray(np.arange(100), dims=['x'], name='var').to_dataset()

        # Simple case: 2 workers, world_size=1
        # Workers act as additional processes and we expect interleaved chunks
        torch_ds = ShardedXrDataset(xr_ds, chunk_size=15, dim='x', world_size=1, rank=0, shuffle=False)
        torch_ds = _Unzip(torch_ds)
        dataloader = DataLoader(
            torch_ds,
            num_workers=2,
            batch_size=1,
            prefetch_factor=1,
        )
        results = list(batch.item() for batch in dataloader)
        assert results == [
            0,
            15,
            1,
            16,
            2,
            17,
            3,
            18,
            4,
            19,
            5,
            20,
            6,
            21,
            7,
            22,
            8,
            23,
            9,
            24,
            10,
            25,
            11,
            26,
            12,
            27,
            13,
            28,
            14,
            29,
            30,
            45,
            31,
            46,
            32,
            47,
            33,
            48,
            34,
            49,
            35,
            50,
            36,
            51,
            37,
            52,
            38,
            53,
            39,
            54,
            40,
            55,
            41,
            56,
            42,
            57,
            43,
            58,
            44,
            59,
            60,
            75,
            61,
            76,
            62,
            77,
            63,
            78,
            64,
            79,
            65,
            80,
            66,
            81,
            67,
            82,
            68,
            83,
            69,
            84,
            70,
            85,
            71,
            86,
            72,
            87,
            73,
            88,
            74,
            89,
        ]

        # Advanced case: 2 workers, world_size=2
        # Each rank gets consecutive chunks and splits them between workers (which interleave again)
        # Since the effective world size is now 4, and the dataset has 6 chunks in total, we will only get 4 chunks (up to 60)
        torch_ds = ShardedXrDataset(xr_ds, chunk_size=15, dim='x', world_size=2, rank=0, shuffle=False)
        torch_ds = _Unzip(torch_ds)
        dataloader = DataLoader(
            torch_ds,
            num_workers=2,
            batch_size=1,
            prefetch_factor=1,
        )
        results = list(batch.item() for batch in dataloader)
        assert results == [
            0,
            15,
            1,
            16,
            2,
            17,
            3,
            18,
            4,
            19,
            5,
            20,
            6,
            21,
            7,
            22,
            8,
            23,
            9,
            24,
            10,
            25,
            11,
            26,
            12,
            27,
            13,
            28,
            14,
            29,
        ]

        torch_ds = ShardedXrDataset(xr_ds, chunk_size=15, dim='x', world_size=2, rank=1, shuffle=False)
        torch_ds = _Unzip(torch_ds)
        dataloader = DataLoader(
            torch_ds,
            num_workers=2,
            batch_size=1,
            prefetch_factor=1,
        )
        results = list(batch.item() for batch in dataloader)
        assert results == [
            30,
            45,
            31,
            46,
            32,
            47,
            33,
            48,
            34,
            49,
            35,
            50,
            36,
            51,
            37,
            52,
            38,
            53,
            39,
            54,
            40,
            55,
            41,
            56,
            42,
            57,
            43,
            58,
            44,
            59,
        ]

    def test_overlap(self):
        ds = xr.DataArray(np.arange(100), dims=['x'], name='var').to_dataset()
        world_size = 3
        chunk_size = 15
        overlap = 5

        shard = partial(
            sharded_xr_dataset, ds, 'x', chunk_size, chunk_overlap=overlap, world_size=world_size, shuffle=False
        )

        chunks_1 = list(shard(rank=0))
        chunks_2 = list(shard(rank=1))
        chunks_3 = list(shard(rank=2))

        assert len(chunks_1) == 2
        assert len(chunks_2) == 2
        assert len(chunks_3) == 2

        assert isinstance(chunks_1[0], xr.Dataset)

        assert chunks_1[0].x.size == 20
        assert chunks_1[1].x.size == 20
        assert chunks_2[0].x.size == 20
        assert chunks_2[1].x.size == 20
        assert chunks_3[0].x.size == 20
        assert chunks_3[1].x.size == 20

        assert_array_equal(chunks_1[0]['var'], np.arange(0, 20))
        assert_array_equal(chunks_2[0]['var'], np.arange(15, 35))
        assert_array_equal(chunks_3[0]['var'], np.arange(30, 50))
        assert_array_equal(chunks_1[1]['var'], np.arange(45, 65))
        assert_array_equal(chunks_2[1]['var'], np.arange(60, 80))
        assert_array_equal(chunks_3[1]['var'], np.arange(75, 95))

    def test_overlap_unequal_uneven(self):
        ds = xr.DataArray(np.arange(100), dims=['x'], name='var').to_dataset()
        world_size = 3
        chunk_size = 15
        overlap = 5

        shard = partial(
            sharded_xr_dataset,
            ds,
            'x',
            chunk_size,
            chunk_overlap=overlap,
            even_shards=False,
            equal_chunks=False,
            world_size=world_size,
            shuffle=False,
        )

        chunks_1 = list(shard(rank=0))
        chunks_2 = list(shard(rank=1))
        chunks_3 = list(shard(rank=2))

        assert len(chunks_1) == 3
        assert len(chunks_2) == 2
        assert len(chunks_3) == 2

        assert isinstance(chunks_1[0], xr.Dataset)

        assert chunks_1[0].x.size == 20
        assert chunks_1[1].x.size == 20
        assert chunks_2[0].x.size == 20
        assert chunks_2[1].x.size == 20
        assert chunks_3[0].x.size == 20
        assert chunks_3[1].x.size == 20
        assert chunks_1[2].x.size == 10

        assert_array_equal(chunks_1[0]['var'], np.arange(0, 20))
        assert_array_equal(chunks_2[0]['var'], np.arange(15, 35))
        assert_array_equal(chunks_3[0]['var'], np.arange(30, 50))
        assert_array_equal(chunks_1[1]['var'], np.arange(45, 65))
        assert_array_equal(chunks_2[1]['var'], np.arange(60, 80))
        assert_array_equal(chunks_3[1]['var'], np.arange(75, 95))
        assert_array_equal(chunks_1[2]['var'], np.arange(90, 100))


class TestInterleaveBatches:
    def test_basic(self):
        batches = [
            torch.arange(0, 8),
            torch.arange(8, 16),
            torch.arange(16, 24),
            torch.arange(24, 32),
        ]
        interleaved_batches = list(t.clone() for t in interleave_batches(batches, num_batches=2))
        assert len(interleaved_batches) == 4
        assert {t.item() for t in interleaved_batches[0]} == {0, 1, 2, 3, 8, 9, 10, 11}
        assert {t.item() for t in interleaved_batches[1]} == {4, 5, 6, 7, 12, 13, 14, 15}
        assert {t.item() for t in interleaved_batches[2]} == {16, 17, 18, 19, 24, 25, 26, 27}
        assert {t.item() for t in interleaved_batches[3]} == {20, 21, 22, 23, 28, 29, 30, 31}


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
