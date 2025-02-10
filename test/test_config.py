import sys
from datetime import date

import dmlcloud as dml
import pytest


class TestConfig:
    def test_import_object(self):
        obj = dml.import_object('datetime.date')
        assert obj is date

    def test_factory_from_cfg(self):
        factory = dml.factory_from_cfg('datetime.date')
        assert factory(2025, 1, day=1) == date(2025, 1, 1)

        factory2 = dml.factory_from_cfg('datetime.date', 2025, 1, 1)
        assert factory2() == date(2025, 1, 1)

        factory3 = dml.factory_from_cfg('datetime.date', 2025, day=31)
        assert factory3(12) == date(2025, 12, 31)
        assert factory3(month=12) == date(2025, 12, 31)

    def test_factory_from_cfg_mapping(self):
        config = {'factory': 'datetime.date', 'year': 2025, 'month': 1, 'day': 1}
        factory = dml.factory_from_cfg(config)
        assert factory() == date(2025, 1, 1)

        config = {'factory': 'datetime.date', 'month': 1, 'day': 1}
        factory = dml.factory_from_cfg(config)
        assert factory(1990) == date(1990, 1, 1)

    def test_obj_from_cfg(self):
        assert dml.obj_from_cfg('datetime.date', 2025, 1, 1) == date(2025, 1, 1)
        assert dml.obj_from_cfg('datetime.date', year=2025, month=1, day=1) == date(2025, 1, 1)
        assert dml.obj_from_cfg('datetime.date', 2025, month=1, day=1) == date(2025, 1, 1)

    def test_obj_from_cfg_mapping(self):
        config = {'factory': 'datetime.date', 'year': 2025, 'month': 1, 'day': 1}
        assert dml.obj_from_cfg(config) == date(2025, 1, 1)

        config = {'factory': 'datetime.date', 'month': 1, 'day': 1}
        assert dml.obj_from_cfg(config, 1990) == date(1990, 1, 1)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
