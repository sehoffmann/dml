import os
from contextlib import contextmanager
from functools import wraps

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
    'init',
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
    """
    Check if the program was started using srun (SLURM).

    This is determined by checking if the SLURM_PROCID environment variable is set.
    """

    return 'SLURM_PROCID' in os.environ


def has_environment():
    """
    Check if the environment variables used by the "env://" initialization method are set.

    This is determined by checking if the MASTER_PORT environment variable is set.
    """

    return 'MASTER_PORT' in os.environ


def has_mpi():
    """
    Check if MPI is available.

    Requires the mpi4py package.
    """

    try:
        from mpi4py import MPI  # noqa: F401

        return True
    except ImportError:
        return False


def is_root():
    """
    Check if the current rank is the root rank (rank 0).
    """
    return dist.get_rank() == 0


def root_only(fn):
    """
    Decorator for methods that should only be called on the root rank.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if is_root():
            return fn(*args, **kwargs)

    return wrapper


@contextmanager
def root_first():
    """
    Context manager that ensures that the root rank executes the code first before all other ranks.

    This is realized by inserting a barrier before or after the code block depending on the rank.
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
    """
    Returns the rank of the current process.
    """

    return _WorkerInfo.RANK


def world_size():
    """
    Returns the total number of processes.
    """

    return _WorkerInfo.WORLD_SIZE


def local_rank():
    """
    Returns the local rank of the current process.

    Returns:
        int: The local rank of the current process if available, otherwise None.
    """

    return _WorkerInfo.LOCAL_RANK


def local_world_size():
    """
    Returns the local world size.

    Returns:
        int: The local world size if available, otherwise None.
    """

    return _WorkerInfo.LOCAL_WORLD_SIZE


def local_node():
    """
    Returns the node id of the current process.

    Returns:
        int: The node id of the current process if available, otherwise None.
    """

    return _WorkerInfo.NODE_ID


def print_worker(*values, sep=' ', end="\n", file=None, flush=True, barrier=False):
    """
    Print the values to a stream, default sys.stdout, with additional information about the worker.

    Args:
        values (Any): The values to print.
        sep (str, optional): The separator between arguments. Default is a space.
        end (str, optional): The string to append at the end of the message. Default is a newline.
        file (file, optional): The file to write the message to. Default is None.
        flush (bool, optional): If True, the output buffer is flushed. Default is True.
        barrier (bool, optional): If True, a barrier is inserted before and after printing. Default is False.
    """

    if barrier:
        dist.barrier()
    modified_values = [f'Worker {rank()}']
    if local_node() is not None:
        modified_values += [f'({local_node()}.{local_rank()})']
    modified_values.extend(values)
    print(*modified_values, sep=sep, end=end, file=file, flush=flush)
    if barrier:
        dist.barrier()


@root_only
def print_root(*values, sep=' ', end="\n", file=None, flush=True):
    """
    Print the values to a stream if the current rank is the root rank.

    Default is to print to the standard output stream.

    Args:
        msg (str): The message to print.
        sep (str, optional): The separator between arguments. Default is a space.
        end (str, optional): The string to append at the end of the message. Default is a newline.
        file (file, optional): The file to write the message to. Default is None.
        flush (bool, optional): If True, the output buffer is flushed. Default is True.
    """

    print(*values, sep=sep, end=end, file=file, flush=flush)


def all_gather_object(obj, group=None):
    """
    Gather objects from all ranks in the group.

    This is a convenience wrapper around `torch.distributed.all_gather_object`.

    Args:
        obj (object): The object to gather. Must be pickable.
        group (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Default is None.

    Returns:
        List[object]: A list of objects gathered from all ranks.
    """

    outlist = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(outlist, obj, group=group)
    return outlist


def gather_object(obj, dst=0, group=None):
    """
    Gathers objects from all ranks in the group to the destination rank.

    This is a convenience wrapper around `torch.distributed.gather_object.

    Args:
        obj (object): The object to gather. Must be pickable. Only used on ranks that are not the destination rank.
        dst (int): Destination rank to gather the object to. Default is 0.
        group (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Default is None.

    Returns:
        List[object]: A list of objects gathered from all ranks. None for ranks that are not the destination rank.
    """

    if dist.get_rank() == dst:
        outlist = [None for _ in range(dist.get_world_size(group))]
    else:
        outlist = None
    dist.gather_object(obj, outlist, dst=dst, group=group)
    return outlist


def broadcast_object(obj=None, src=0, group=None, device=None):
    """
    Broadcasts an object from the source rank to all other ranks in the group.

    This is a wrapper around `torch.distributed.broadcast_object_list` that broadcasts a single object.

    Args:
        obj (Any, optional): The object to broadcast. Must be pickable. Only used on the source rank. Default is None.
        src (int): Source rank from which to broadcast the object. Source rank is based on global process group (regardless of group argument)
        group (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Default is None.
        device (torch.device, optional): If not None, the object is serialized and converted to as tensor which is moved to the device before broadcasting. Default is None.

    Returns:
        The broadcasted object.
    """
    objlist = [obj]
    dist.broadcast_object_list(objlist, src=src, group=group, device=device)
    return objlist[0]


def _init_process_group_dummy(**kwargs):
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


def _init_process_group_slurm(port=DEFAULT_PORT, **kwargs):
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


def _init_process_group_MPI(ip_idx=0, port=DEFAULT_PORT, **kwargs):
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


def _init_process_group_auto(verbose=True, **kwargs):
    """
    Tries to initialize torch.distributed in the following order:
    1. If the MASTER_PORT environment variable is set, use environment variable initialization
    2. If srun (slurm) was used to launch this program, use slurms environment variables
    2. If MPI is available, use MPI to exchange ip addresses (see init_process_group_MPI)
    3. Otherwise, a dummy process group with a single process is used (no distributed training)
    """

    # determine init method
    if has_environment():
        dist.init_process_group(init_method='env://', **kwargs)
    elif has_slurm():
        _init_process_group_slurm(**kwargs)
    elif has_mpi():
        _init_process_group_MPI(**kwargs)
    else:
        _init_process_group_dummy()


def init(kind='auto'):
    """
    Initializes the torch.distributed framework.

    For most use cases, kind='auto' (the default) should be sufficient.

    - If kind is 'env', the "env://" initialization method is used. See torch.distributed.init_process_group.
    - If kind is 'slurm', SLURM environment variables are used to find the ip address of the root rank.
    - If kind is 'mpi', MPI is used to exchange ip addresses.
    - If kind is 'dummy', a dummy process group with a single process is used (no distributed training). This is useful for debugging and testing.


    The 'auto' kind tries to initialize the process group in the following order:

    1. If the MASTER_PORT environment variable is set, use environment variable initialization
    2. If srun (slurm) was used to launch this program, use slurms environment variables
    3. If MPI is available, use MPI to exchange ip addresses
    4. Otherwise, a dummy process group with a single process is used (no distributed training)

    Args:
        kind (str): The kind of initialization to use. Can be one of 'auto', 'dummy', 'slurm', 'mpi', or 'env'.
    """

    if kind not in ['auto', 'dummy', 'slurm', 'mpi', 'env']:
        raise ValueError(f"Invalid kind: {kind}. Must be one of 'auto', 'dummy', 'slurm', 'mpi', 'env'")

    if kind == 'auto':
        _init_process_group_auto()
    elif kind == 'dummy':
        _init_process_group_dummy()
    elif kind == 'slurm':
        _init_process_group_slurm()
    elif kind == 'mpi':
        _init_process_group_MPI()
    elif kind == 'env':
        dist.init_process_group(init_method='env://')


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
