from typing import Iterable

import torch

__all__ = [
    'interleave_batches',
    'interleave_dict_batches',
]


def interleave_batches(
    iterable: Iterable[torch.Tensor], num_batches: int, pin_memory: bool = False
) -> Iterable[torch.Tensor]:
    """
    Interleaves batches from an iterable of batches.
    Important: Returned batches must be used immediately or copied to avoid overwriting.
    """
    if num_batches < 1:
        raise ValueError('num_batches must be greater than 0')

    if num_batches == 1:
        yield from iterable

    batches = []
    memory = None
    batch_size = None
    slice_size = None
    for batch in iterable:
        if memory is None:
            batch_size = batch.shape[0]
            slice_size = batch_size // num_batches
            if batch_size % num_batches != 0:
                raise ValueError(f'Batch dimension ({batch_size}) must be divisible by num_batches={num_batches}')
            memory = torch.empty(
                (num_batches, *batch.shape), dtype=batch.dtype, device=batch.device, pin_memory=pin_memory
            )

        batches.append(batch)

        if len(batches) == num_batches:
            for i in range(num_batches):
                for j in range(num_batches):
                    memory[i, j * slice_size : (j + 1) * slice_size] = batches[j][i * slice_size : (i + 1) * slice_size]
            batches = []
            for i in range(num_batches):
                yield memory[i]


def interleave_dict_batches(
    iterable: Iterable[torch.Tensor], num_batches: int, pin_memory: bool = False
) -> Iterable[torch.Tensor]:
    """
    Interleaves batches from an iterable of batches.
    Important: Returned batches must be used immediately or copied to avoid overwriting.
    """
    if num_batches < 1:
        raise ValueError('num_batches must be greater than 0')

    if num_batches == 1:
        yield from iterable

    batches = []
    memory = {}
    slice_size = {}
    for batch in iterable:
        if not memory:
            for k, tensor in batch.items():
                batch_size = tensor.shape[0]
                if batch_size % num_batches != 0:
                    raise ValueError(f'Batch dimension ({batch_size}) must be divisible by num_batches={num_batches}')
                slice_size[k] = batch_size // num_batches
                memory[k] = torch.empty(
                    (num_batches, *tensor.shape), dtype=tensor.dtype, device=tensor.device, pin_memory=pin_memory
                )

        batches.append(batch)

        if len(batches) == num_batches:
            for k in memory:
                for i in range(num_batches):
                    for j in range(num_batches):
                        source = batches[j][k][i * slice_size[k] : (i + 1) * slice_size[k]]
                        memory[k][i, j * slice_size[k] : (j + 1) * slice_size[k]] = source
            batches = []
            for i in range(num_batches):
                yield {k: memory[k][i] for k in memory.keys()}
