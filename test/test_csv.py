import csv
import sys

import dmlcloud as dml
import pytest
from dmlcloud.core.callbacks import CsvCallback


class DummyStage(dml.Stage):
    def run_epoch(self):
        self.log('train/loss', 10 - self.current_epoch)
        self.log('train/acc', 90 + self.current_epoch)


class TestCsvCallback:
    def test_basic_metrics(self, torch_distributed, tmp_path):
        metrics_file = tmp_path / 'epoch_metrics_DummyStage.csv'

        pipe = dml.Pipeline()
        pipe.append(DummyStage(epochs=5))
        pipe.add_callback(CsvCallback(tmp_path))
        pipe.run()

        assert metrics_file.exists()

        with open(metrics_file) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 6
        assert rows[0][:3] == ['epoch', 'train/loss', 'train/acc']
        assert rows[1][:3] == ['0', '10.0', '90.0']
        assert rows[2][:3] == ['1', '9.0', '91.0']
        assert rows[3][:3] == ['2', '8.0', '92.0']
        assert rows[4][:3] == ['3', '7.0', '93.0']
        assert rows[5][:3] == ['4', '6.0', '94.0']

        # misc metrics
        assert 'misc/epoch_time' in rows[0]
        assert 'misc/total_time' in rows[0]
        assert 'misc/eta' in rows[0]

    def test_stage_name(self, torch_distributed, tmp_path):
        pipe = dml.Pipeline()
        pipe.append(DummyStage(epochs=5))
        pipe.add_callback(CsvCallback(tmp_path))
        pipe.run()

        assert (tmp_path / 'epoch_metrics_DummyStage.csv').exists()

    def test_duplicate_names(self, torch_distributed, tmp_path):
        pipe = dml.Pipeline()
        pipe.append(DummyStage(epochs=5))
        pipe.append(DummyStage(epochs=5))
        pipe.add_callback(CsvCallback(tmp_path))
        pipe.run()

        assert (tmp_path / 'epoch_metrics_DummyStage_1.csv').exists()
        assert (tmp_path / 'epoch_metrics_DummyStage_2.csv').exists()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
