import pytest
import torch
from dmlcloud.core.distributed import init


@pytest.fixture
def torch_distributed():
    init(kind='dummy')
    yield
    torch.distributed.destroy_process_group()
