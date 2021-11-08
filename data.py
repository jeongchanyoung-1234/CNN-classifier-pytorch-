import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import FashionMNIST


def get_dataloader(config):
    train_dataset = \
        FashionMNIST(root='./data',
                     train=True,
                     transform=torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize(0, 1)
                     ]),
                     download=True)
    valid_dataset = \
        FashionMNIST(root='./data',
                     train=False,
                     transform=torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize(0, 1)
                     ]),
                     download=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_dataloader, valid_dataloader

if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.batch_size = 4
    config = Config()
    train_dataloader, _ = get_dataloader(config)
    print(next(iter(train_dataloader))[0].size())