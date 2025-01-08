import argparse

import dmlcloud as dml
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTStage(dml.Stage):
    # The pre_stage method is called before the first epoch
    # It's a good place to load the dataset, create the model, and set up the optimizer
    def pre_stage(self):
        # Load the MNIST dataset
        # The root_first context manager ensures the root process downloads the dataset before the other processes
        with dml.root_first():
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.MNIST(root='data', train=True, download=dml.is_root(), transform=transform)
            val_dataset = datasets.MNIST(root='data', train=False, download=dml.is_root(), transform=transform)

        # For distributed training, we need to shard our dataset across all processes
        # Here we use the DistributedSampler to do this
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=32, sampler=self.train_sampler)

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)

        # We create our model regularly...
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

        # ...and then wrap it with dml.wrap_ddp to enable distributed training
        self.model = dml.wrap_ddp(model, self.device)

        # It's also important to scale the learning rate based on the number of GPUs, dml.scale_lr does this for us
        # Otherwise, we wouldn't profit from the increased batch size
        # In practice, you would likely want to combine this with a linear lr rampup during the very first steps as well
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=dml.scale_lr(1e-3))

        self.loss = nn.CrossEntropyLoss()

        # Finally, we add columns to the table to track the loss and accuracy
        self.add_column('[Train] Loss', 'train/loss', color='green')
        self.add_column('[Train] Acc.', 'train/accuracy', formatter=lambda acc: f'{100*acc:.2f}%', color='green')
        self.add_column('[Val] Loss', 'val/loss', color='cyan')
        self.add_column('[Val] Acc.', 'val/accuracy', formatter=lambda acc: f'{100*acc:.2f}%', color='cyan')

        self.train_acc = self.add_metric('train/accuracy', torchmetrics.Accuracy('multiclass', num_classes=10))
        self.val_acc = self.add_metric('val/accuracy', torchmetrics.Accuracy('multiclass', num_classes=10))

    # The run_epoch method is called once per epoch
    def run_epoch(self):
        self._train_epoch()
        self._val_epoch()

    def _train_epoch(self):
        self.model.train()
        self.metric_prefix = 'train'  # This is used to prefix the metrics in the table
        self.train_sampler.set_epoch(self.current_epoch)

        for img, target in self.train_loader:
            img, target = img.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(img)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.log('loss', loss)
            # self.log('accuracy', (output.argmax(1) == target).float().mean())
            self.train_acc(output, target)

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        self.metric_prefix = 'val'

        for img, target in self.val_loader:
            img, target = img.to(self.device), target.to(self.device)

            output = self.model(img)
            loss = self.loss(output, target)

            self.log('loss', loss)
            # self.log('accuracy', (output.argmax(1) == target).float().mean())
            self.val_acc(output, target)


def main():
    dml.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    seed = dml.seed(args.seed)  # This is a helper function to set the seed for all devices
    config = {
        'seed': seed,
        'epochs': args.epochs,
    }

    pipe = dml.Pipeline(config, name='MNIST')
    pipe.append(MNISTStage(epochs=args.epochs))
    pipe.enable_checkpointing('checkpoints')
    pipe.enable_wandb()
    pipe.run()


if __name__ == '__main__':
    main()
