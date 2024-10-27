import pytest
from dmlcloud.core.distributed import deinitialize_torch_distributed, init_process_group_dummy


@pytest.fixture
def torch_distributed():
    init_process_group_dummy()
    yield
    deinitialize_torch_distributed()
