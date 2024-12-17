import torch
from torch import nn

from . import logging as dml_logging


__all__ = [
    'count_parameters',
    'wrap_ddp',
]


def count_parameters(module: nn.Module) -> int:
    """
    Returns the number of trainable parameters in a module.

    Args:
        module (nn.Module): The module to count the parameters of.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def wrap_ddp(module: nn.Module, device: torch.device, sync_bn: bool = False, verbose: bool = True) -> nn.Module:
    """
    Wraps a module with DistributedDataParallel.

    Args:
        module (nn.Module): The module to wrap.
        device (torch.device): The device to use.
        sync_bn (bool, optional): If True, uses SyncBatchNorm. Default is False.
        verbose (bool, optional): If True, prints information about the model. Default is True.

    Returns:
        nn.Module: The wrapped module.
    """

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError('DistributedDataParallel requires torch.distributed to be initialized.')
    
    module = module.to(device)  # Doing it in this order is important for SyncBN
    if sync_bn:
        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)

    device_ids = [device] if device.type == 'cuda' else None  # Must be None for cpu devices
    ddp = nn.parallel.DistributedDataParallel(module, broadcast_buffers=False, device_ids=device_ids)
    if verbose:
        msg = f'* MODEL:\n'
        msg += f'    - Parameters: {count_parameters(module) / 1e6:.1f} kk\n'
        msg += f'    - {module}'
        dml_logging.info(msg)

    return ddp