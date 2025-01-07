from multiprocessing import Pipe, Process

import pytest
import torch
import torch.distributed
from dmlcloud.core.distributed import init


@pytest.fixture
def torch_distributed():
    init(kind='dummy')
    yield
    torch.distributed.destroy_process_group()


class DistributedEnvironment:
    @staticmethod
    def bind(tmpdir):
        def init(world_size, timeout=5 * 60, daemon=True):
            return DistributedEnvironment(world_size, timeout, daemon, str(tmpdir / 'filestore'))

        return init

    def __init__(self, world_size: int, timeout: int = 5 * 60, daemon: bool = True, file: str = None):
        self.world_size = world_size
        self.timeout = timeout
        self.daemon = daemon
        self.file = str(file)

    def _run(self, rank, conn, func, *args, **kwargs):
        store = torch.distributed.FileStore(self.file, self.world_size)
        torch.distributed.init_process_group(backend='gloo', world_size=self.world_size, rank=rank, store=store)

        torch.distributed.barrier()
        ret = func(*args, **kwargs)  # TODO: need to handle exceptions
        torch.distributed.barrier()

        conn.send(ret)

        torch.distributed.destroy_process_group()

    def start(self, func, *args, **kwargs):
        self.processes = []
        self.conns = []
        for rank in range(self.world_size):
            recv_conn, send_conn = Pipe()
            process_args = (rank, send_conn, func) + args
            process_kwargs = dict(kwargs)
            process = Process(target=self._run, args=process_args, kwargs=process_kwargs, daemon=self.daemon)
            self.conns.append(recv_conn)
            self.processes.append(process)

        for process in self.processes:
            process.start()

        return_values = []
        for process, conn in zip(self.processes, self.conns):  # TODO: should probably be a context manager
            ret = conn.recv()
            return_values.append(ret)
            process.join(self.timeout)

        return return_values


@pytest.fixture
def distributed_environment(tmp_path):
    return DistributedEnvironment.bind(tmp_path)
