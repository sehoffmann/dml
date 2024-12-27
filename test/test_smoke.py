import sys

import dmlcloud as dml
import pytest
import torch


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 256

    def __getitem__(self, idx):
        x = torch.randn(10)
        y = x.sum() * 0.1
        return x, y


class DummyStage(dml.Stage):
    def pre_stage(self):
        self.train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=32)

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.Linear(32, 1),
        )
        self.model = dml.wrap_ddp(model, self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=dml.scale_lr(1e-2))
        self.loss = torch.nn.L1Loss()

    def run_epoch(self):
        for x, y in self.train_dl:
            self.optim.zero_grad()

            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.loss(output[:, 0], y)
            loss.backward()

            self.optim.step()

            self.log('train/loss', loss)


class TestSmoke:
    def test_smoke(self, torch_distributed):
        pipe = dml.Pipeline()
        stage = DummyStage(epochs=3)
        pipe.append(stage)
        pipe.run()

        assert stage.current_epoch == 3
        assert 'train/loss' in stage.history
        assert stage.history.last()['train/loss'] < 0.1


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
