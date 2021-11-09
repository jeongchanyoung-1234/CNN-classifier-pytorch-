import argparse

import torch.nn as nn
import torch.optim as optim

from model import FashionMNISTClassifier
from trainer import FashionMNISTTrainer
from data import get_dataloader

def define_argparse():
    p = argparse.ArgumentParser()

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--n_kernels', type=int, default=16)
    p.add_argument('--kernel_size', type=int, default=5)
    p.add_argument('--pool_size', type=int, default=2)
    p.add_argument('--dropout', type=float, default=.5)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config

def main(config):
    optimizer_map = {'adam': optim.Adam, 'sgd': optim.SGD, 'rmsprop': optim.RMSprop}

    model = FashionMNISTClassifier(config)
    optimizer = optimizer_map[config.optimizer.lower()](model.parameters(), lr=config.lr)
    criterion = nn.NLLLoss()
    train_dataloader, valid_dataloader = get_dataloader(config)
    print('|train X|:', next(iter(train_dataloader))[0].size())
    print('|valid X|:', next(iter(valid_dataloader))[0].size())
    print()

    trainer = FashionMNISTTrainer(
        model,
        optimizer,
        criterion,
        train_dataloader,
        valid_dataloader,
        config
    )

    trainer.train()

if __name__ == '__main__':
    config = define_argparse()
    main(config)