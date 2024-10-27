import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from ..util.tcp import find_free_port, get_local_ips

__all__ = [
    'has_slurm',
    'has_environment',
    'has_mpi',
    'is_root',
    'root_only',
    'root_first',
    'rank',
    'world_size',
    'local_rank',
    'local_world_size',
    'local_node',
    'print_worker',
    'print_root',
    'all_gather_object',
    'gather_object',
    'broadcast_object',
    'init_process_group_dummy',
    'init_process_group_slurm',
    'init_process_group_MPI',
    'init_process_group_auto',
    'deinitialize_torch_distributed',   
]


DEFAULT_PORT = os.environ.get('DMLCLOUD_PORT', 41312)  # dml


class _WorkerInfo:
    INIT_METHOD = None
    RANK = None
    WORLD_SIZE = None
    LOCAL_RANK = None
    LOCAL_WORLD_SIZE = None
    NODE_ID = None


def has_slurm():
    return 'SLURM_PROCID' in os.environ


def has_environment():
    return 'MASTER_PORT' in os.environ


def has_mpi():
    try:
        from mpi4py import MPI  # noqa: F401

        return True
    except ImportError:
        return False


def is_root():
    return dist.get_rank() == 0


def root_only(fn):
    """
    Decorator for methods that should only be called on the root rank.
    """

    def wrapper(*args, **kwargs):
        if is_root():
            return fn(*args, **kwargs)

    return wrapper


@contextmanager
def root_first():
    """
    Context manager that ensures that the root rank executes the code first before all other ranks
    """
    if is_root():
        try:
            yield
        finally:
            dist.barrier()
    else:
        dist.barrier()
        try:
            yield
        finally:
            pass


def mpi_local_comm():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
        return local_comm
    except ImportError:
        return None


def rank():
    return _WorkerInfo.RANK


def world_size():
    return _WorkerInfo.WORLD_SIZE


def local_rank():
    return _WorkerInfo.LOCAL_RANK


def local_world_size():
    return _WorkerInfo.LOCAL_WORLD_SIZE


def local_node():
    return _WorkerInfo.NODE_ID


def print_worker(msg, barrier=True, flush=True):
    if barrier:
        dist.barrier()
    s = f'Worker {rank()}'
    if local_node() is not None:
        s += f'({local_node()}.{local_rank()})'
    s += f':{msg}'
    print(s, flush=flush)
    if barrier:
        dist.barrier()


@root_only
def print_root(msg, flush=True):
    print(msg, flush=flush)


def all_gather_object(obj, group=None):
    outlist = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(outlist, obj, group=group)
    return outlist


def gather_object(obj, dst=0, group=None):
    if dist.get_rank() == dst:
        outlist = [None for _ in range(dist.get_world_size(group))]
    else:
        outlist = None
    dist.gather_object(obj, outlist, dst=dst, group=group)
    return outlist


def broadcast_object(obj, src=0, group=None, device=None):
    objlist = [obj]
    dist.broadcast_object_list(objlist, src=src, group=group, device=None)
    return objlist[0]


def init_process_group_dummy(**kwargs):
    """
    Initializes the process group with a single process.
    Uses HashStore under the hood. Useful for applications that
    only run on a single gpu.
    """
    _WorkerInfo.INIT_METHOD = 'dummy'
    _WorkerInfo.RANK = 0
    _WorkerInfo.WORLD_SIZE = 1
    _WorkerInfo.LOCAL_RANK = 0
    _WorkerInfo.LOCAL_WORLD_SIZE = 1
    _WorkerInfo.NODE_ID = 0

    backend = kwargs.get('backend', None)
    if backend is None:
        backend = 'cpu:gloo,cuda:nccl' if dist.is_nccl_available() and torch.cuda.is_available() else 'gloo'
    store = dist.HashStore()
    dist.init_process_group(store=store, rank=0, world_size=1, backend=backend, **kwargs)


def init_process_group_slurm(port=DEFAULT_PORT, **kwargs):
    _WorkerInfo.INIT_METHOD = 'slurm'
    _WorkerInfo.RANK = int(os.environ['SLURM_PROCID'])
    _WorkerInfo.WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
    _WorkerInfo.LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
    _WorkerInfo.LOCAL_WORLD_SIZE = int(os.environ['SLURM_STEP_TASKS_PER_NODE'])
    _WorkerInfo.NODE_ID = int(os.environ['SLURM_NODEID'])

    ip = os.environ['SLURM_SRUN_COMM_HOST']

    dist.init_process_group(
        init_method=f'tcp://{ip}:{port}',
        world_size=_WorkerInfo.WORLD_SIZE,
        rank=_WorkerInfo.RANK,
        **kwargs,
    )


def init_process_group_MPI(ip_idx=0, port=DEFAULT_PORT, **kwargs):
    """
    This method setups up the distributed backend using MPI, even
    if torch was not built with MPI support. For this to work, you
    need to have mpi4py installed and the root rank must be reachable
    via TCP.

    If port is None, we will automatically try to find a free port.

    ip_idx can be used to specify which IP address to use if the root
    has multiple IP addresses. The default is 0, which means the first.

    kwargs are passed to torch.distributed.init_process_group.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    local_comm = mpi_local_comm()

    _WorkerInfo.INIT_METHOD = 'mpi'
    _WorkerInfo.RANK = comm.Get_rank()
    _WorkerInfo.WORLD_SIZE = comm.Get_size()
    _WorkerInfo.LOCAL_RANK = local_comm.Get_rank()
    _WorkerInfo.LOCAL_WORLD_SIZE = local_comm.Get_size()

    if port is None:
        port = find_free_port()

    if _WorkerInfo.RANK == 0:
        ip = get_local_ips()[ip_idx]
    else:
        ip = None

    ip = comm.bcast(ip, root=0)
    port = comm.bcast(port, root=0)
    url = f'tcp://{ip}:{port}'

    comm.Barrier()

    dist.init_process_group(
        init_method=url,
        world_size=_WorkerInfo.WORLD_SIZE,
        rank=_WorkerInfo.RANK,
        **kwargs,
    )


def init_process_group_auto(verbose=True, **kwargs):
    """
    Tries to initialize torch.distributed in the following order:
    1. If the MASTER_PORT environment variable is set, use environment variable initialization
    2. If srun (slurm) was used to launch this program, use slurms environment variables
    2. If MPI is available, use MPI to exchange ip addresses (see init_process_group_MPI)
    3. Otherwise, use a single process group (see init_process_group_dummy)
    """

    # determine init method
    if has_environment():
        dist.init_process_group(init_method='env://', **kwargs)
    elif has_slurm():
        init_process_group_slurm(**kwargs)
    elif has_mpi():
        init_process_group_MPI(**kwargs)
    else:
        init_process_group_dummy()


def deinitialize_torch_distributed():
    """
    Deinitializes the torch distributed framework.
    At the time of writing, `dist.destroy_process_group()` is not well documented.
    Hence, this function.
    """
    _WorkerInfo.INIT_METHOD = None
    _WorkerInfo.RANK = None
    _WorkerInfo.WORLD_SIZE = None
    _WorkerInfo.LOCAL_RANK = None
    _WorkerInfo.LOCAL_WORLD_SIZE = None
    _WorkerInfo.NODE_ID = None
    dist.destroy_process_group()
