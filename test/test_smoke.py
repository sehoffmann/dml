import sys

import pytest
import torch
from dmlcloud.core.pipeline import TrainingPipeline
from dmlcloud.core.stage import TrainValStage


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return torch.randn(10), torch.randint(0, 10, size=(1,)).item()


class DummyStage(TrainValStage):
    def pre_stage(self):
        self.model = torch.nn.Linear(10, 10)
        self.pipeline.register_model('linear', self.model)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.pipeline.register_optimizer('sgd', self.optimizer)

        self.pipeline.register_dataset('train', torch.utils.data.DataLoader(DummyDataset(), batch_size=4))
        self.pipeline.register_dataset('val', torch.utils.data.DataLoader(DummyDataset(), batch_size=4))

        self.loss = torch.nn.CrossEntropyLoss()

    def step(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        output = self.model(x)
        loss = self.loss(output, y)
        return loss


class TestSmoke:
    def test_smoke(self, torch_distributed):
        pipeline = TrainingPipeline()
        pipeline.append_stage(DummyStage(), max_epochs=1)
        pipeline.run()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
