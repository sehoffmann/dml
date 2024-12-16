import sys

sys.path.insert(0, './')

import dmlcloud as dml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTStage(dml.TrainValStage):
    def pre_stage(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        with dml.root_first():
            train_dataset = datasets.MNIST(root='data', train=True, download=dml.is_root(), transform=transform)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.pipeline.register_dataset('train', DataLoader(train_dataset, batch_size=32, sampler=train_sampler))

            val_dataset = datasets.MNIST(root='data', train=False, download=dml.is_root(), transform=transform)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            self.pipeline.register_dataset('val', DataLoader(val_dataset, batch_size=32, sampler=val_sampler))

        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(784, 10),
        )
        self.pipeline.register_model('cnn', model)

        self.pipeline.register_optimizer('adam', torch.optim.Adam(model.parameters(), lr=1e-3))

        self.loss = nn.CrossEntropyLoss()

    def step(self, batch) -> torch.Tensor:
        img, target = batch
        img, target = img.to(self.device), target.to(self.device)

        output = self.pipeline.models['cnn'](img)
        loss = self.loss(output, target)

        self.track_reduce('accuracy', (output.argmax(1) == target).float().mean())
        return loss

    def table_columns(self):
        columns = super().table_columns()
        columns.insert(-2, {'name': '[Val] Acc.', 'metric': 'val/accuracy'})
        columns.insert(-2, {'name': '[Train] Acc.', 'metric': 'train/accuracy'})
        return columns


def main():
    dml.init_process_group_auto()
    pipeline = dml.TrainingPipeline(name='mnist')
    pipeline.enable_checkpointing('checkpoints', resume=False)
    pipeline.enable_wandb()
    pipeline.append_stage(MNISTStage(), max_epochs=3)
    pipeline.run()


if __name__ == '__main__':
    main()
