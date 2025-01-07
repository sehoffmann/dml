import dmlcloud as dml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CustomEpochStage(dml.Stage):
    def pre_stage(self):
        with dml.root_first():
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.MNIST(root='data', train=True, download=dml.is_root(), transform=transform)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=32, sampler=self.train_sampler)

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
        self.model = dml.wrap_ddp(model, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=dml.scale_lr(1e-3))

        self.loss = nn.CrossEntropyLoss()

        # Finally, we add columns to the table to track the loss and accuracy
        self.add_column('# Steps', 'misc/steps')
        self.add_column('# Samples', 'misc/total_samples')

        self.add_column('Loss', 'train/loss', color='green')

    def run(self):
        MAX_STEPS = 5000
        LOG_PERIOD = 250

        num_steps = 0
        total_samples = 0
        while num_steps < MAX_STEPS:
            self.train_sampler.set_epoch(self.current_epoch)

            for img, target in self.train_loader:
                img, target = img.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(img)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

                self.log('train/loss', loss)
                self.log('misc/samples', len(img), reduction='sum')

                num_steps += 1
                if num_steps % LOG_PERIOD == 0:
                    total_samples += self.metrics['misc/samples'].compute()
                    self.log('misc/total_samples', total_samples)
                    self.log('misc/steps', num_steps)
                    if num_steps < MAX_STEPS:
                        self.next_epoch()
                        self.train_sampler.set_epoch(self.current_epoch)
                    else:
                        break


def main():
    pipe = dml.Pipeline(name='custom-epochs')
    pipe.append(CustomEpochStage())
    pipe.enable_checkpointing('checkpoints')
    pipe.enable_wandb()
    pipe.run()


if __name__ == '__main__':
    main()
