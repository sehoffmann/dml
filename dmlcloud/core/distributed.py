import inspect
import os
import sys
from contextlib import contextmanager
from datetime import timedelta
from functools import wraps
from typing import Callable, TYPE_CHECKING

import torch
import torch.distributed
import torch.distributed as dist

from ..util.tcp import find_free_port, get_local_ips


if TYPE_CHECKING:
    from dmlcloud import Pipeline, Stage  # noqa: F401


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
    'all_gather_object',
    'gather_object',
    'broadcast_object',
    'init',
    'deinitialize_torch_distributed',
]


DEFAULT_PORT = os.environ.get('DMLCLOUD_PORT', 41312)  # dml
LONG_TIMEOUT = 24 * 60 * 60  # timeout for long running barriers, default is 24 hours


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


def is_root(group: dist.ProcessGroup = None):
    """
    Check if the current rank is the root rank (rank 0).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None (default), the default process group will be used.
    """
    return dist.get_rank(group) == 0


def root_only(
    fn: Callable | type,
    group: torch.distributed.ProcessGroup = None,
    synchronize: bool = True,
    timeout: int = LONG_TIMEOUT,
) -> Callable | type:
    """
    Decorator for methods that should only be called on the root rank.

    Can also be applied to individual callback methods of :class:`Pipeline` and :class:`Stage`, or to the whole class.
    In that case, :attr:`Pipeline.gloo_group` is used as process group.

    If ``synchronize=True``, a monitored_barrier before or after the function call depending on the rank.
    This can be important to prevent timeouts from future all_reduce operations if non-root ranks move on before the root rank has finished.

    Args:
        fn: The function to decorate or a subclass of :class:`Pipeline` or :class:`Stage`.
        group: The process group to work on. If None (default), the default process group will be used.
        synchronize: If True, a barrier is inserted before or after the function call depending on the rank. Default is True.
        timeout: Timeout in seconds for the monitored_barrier. Default is 24 hours.

    Returns:
        The decorated function or class.

    Examples:

        Annotating an individual function:

        >>> @root_only
        >>> def my_function():
        >>>     print('Only the root rank prints this.')

        Annotating a whole :class:`Stage` subclass:

        >>> @root_only
        >>> class MyStage(Stage):
        >>>     def pre_stage(self):
        >>>         print('Only the root rank prints this.')
        >>>
        >>>     def run_epoch(self):
        >>>         print('Only the root rank prints this.')
        >>>
        >>>     def post_stage(self):
        >>>         print('Only the root rank prints this.')

        Annotating individual methods of :class:`Stage`:

        >>> class MyStage(Stage):
        >>>     def pre_stage(self):
        >>>         print('All ranks print this.')
        >>>
        >>>     @root_only
        >>>     def post_stage(self):
        >>>         print('Only the root rank prints this.')
    """

    if not inspect.isclass(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if is_root(group):
                ret = fn(*args, **kwargs)
                if synchronize:
                    dist.monitored_barrier(group, timeout=timedelta(seconds=timeout), wait_all_ranks=True)
                return ret
            elif synchronize:
                dist.monitored_barrier(group, timeout=timedelta(seconds=timeout), wait_all_ranks=True)

        return wrapper

    elif 'dmlcloud.core.pipeline' in sys.modules and issubclass(
        fn, sys.modules['dmlcloud.core.pipeline'].Pipeline
    ):  # avoids circular imports
        pipeline_cls = fn

        def make_wrapper(method):
            @wraps(method)
            def pipeline_wrapper(self, *args, **kwargs):
                if is_root(group):
                    ret = method(self, *args, **kwargs)
                    if synchronize:
                        dist.monitored_barrier(self.gloo_group, timeout=timedelta(seconds=timeout), wait_all_ranks=True)
                    return ret
                elif synchronize:
                    dist.monitored_barrier(self.gloo_group, timeout=timedelta(seconds=timeout), wait_all_ranks=True)

            return pipeline_wrapper

        pipeline_cls.pre_run = make_wrapper(pipeline_cls.pre_run)
        pipeline_cls.post_run = make_wrapper(pipeline_cls.post_run)

        return pipeline_cls

    elif 'dmlcloud.core.stage' in sys.modules and issubclass(
        fn, sys.modules['dmlcloud.core.stage'].Stage
    ):  # avoids circular imports
        stage_cls = fn

        def make_wrapper(method):
            @wraps(method)
            def stage_wrapper(self, *args, **kwargs):
                if is_root(group):
                    ret = method(self, *args, **kwargs)
                    if synchronize:
                        dist.monitored_barrier(
                            self.pipe.gloo_group, timeout=timedelta(seconds=timeout), wait_all_ranks=True
                        )
                    return ret
                elif synchronize:
                    dist.monitored_barrier(
                        self.pipe.gloo_group, timeout=timedelta(seconds=timeout), wait_all_ranks=True
                    )

            return stage_wrapper

        stage_cls.pre_stage = make_wrapper(stage_cls.pre_stage)
        stage_cls.post_stage = make_wrapper(stage_cls.post_stage)
        stage_cls.pre_epoch = make_wrapper(stage_cls.pre_epoch)
        stage_cls.post_epoch = make_wrapper(stage_cls.post_epoch)
        stage_cls.run_epoch = make_wrapper(stage_cls.run_epoch)

        return stage_cls

    else:
        raise ValueError('root_only can only be applied to functions, Pipeline, or Stage subclasses.')


@contextmanager
def root_first(group: dist.ProcessGroup = None):
    """
    Context manager that ensures that the root rank executes the code first before all other ranks.

    This is realized by inserting a barrier before or after the code block depending on the rank.
    Notice, that only a regular barrier is used, and, hence, the default timeout of 1800000 seconds applies for nccl.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None (default), the default process group will be used.
    """
    if is_root():
        try:
            yield
        finally:
            dist.barrier(group)
    else:
        dist.barrier(group)
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

    if _WorkerInfo.RANK is None:
        return dist.get_rank()
    else:
        return _WorkerInfo.RANK


def world_size():
    """
    Returns the total number of processes.
    """

    if _WorkerInfo.WORLD_SIZE is None:
        return dist.get_world_size()
    else:
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


def _init_process_group_env(**kwargs):
    """
    Intialize using "env://" method.

    Reads out helper environment variables to determine local rank and local world size.

    """
    _WorkerInfo.INIT_METHOD = 'env'
    _WorkerInfo.RANK = int(os.environ['RANK'])
    _WorkerInfo.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    _WorkerInfo.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    _WorkerInfo.LOCAL_WORLD_SIZE = int(os.environ['LOCAL_WORLD_SIZE'])
    _WorkerInfo.NODE_ID = int(os.environ['GROUP_RANK'])

    dist.init_process_group(init_method='env://', **kwargs)


def _init_process_group_dummy(**kwargs):
    """
    Initializes the process group with a single process.

    Uses HashStore under the hood. Useful for applications that only run on a single GPU.
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
    _WorkerInfo.NODE_ID = int(os.environ['SLURM_NODEID'])

    # Determine local world size via SLURM_TASKS_PER_NODE
    # Format example: 2(x3),4,1
    tasks_per_node_raw = os.environ['SLURM_TASKS_PER_NODE'].split(',')
    tasks_per_node = []
    for t in tasks_per_node_raw:
        if '(x' in t:
            ntasks, nnodes = t.split('(x')
            tasks_per_node.extend([int(ntasks)] * int(nnodes[:-1]))
        else:
            tasks_per_node.append(int(t))
    _WorkerInfo.LOCAL_WORLD_SIZE = tasks_per_node[_WorkerInfo.NODE_ID]

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

    if has_environment():
        _init_process_group_env(**kwargs)
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
        _init_process_group_env()


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
