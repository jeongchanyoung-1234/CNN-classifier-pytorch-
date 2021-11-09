import torch.nn as nn
import torch.nn.functional as F

class FashionMNISTClassifier(nn.Module):
    def __init__(self, config):
        super(FashionMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=config.n_kernels,
            kernel_size=config.kernel_size,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=config.n_kernels,
            out_channels=config.n_kernels * 2,
            kernel_size=config.kernel_size,
            stride=1,
            padding=1
        )
        self.dropout1 = nn.Dropout2d(config.dropout)
        self.dropout2 = nn.Dropout2d(config.dropout)

        self.pool = nn.MaxPool2d(kernel_size=config.pool_size)
        self.flat = nn.Flatten()

        self.out_layers = nn.Sequential(
            nn.Linear(4608, config.hidden_size),
            nn.Dropout2d(config.dropout),
            nn.Linear(config.hidden_size, 10),
            nn.LogSoftmax(-1)
        )

    def forward(self, x):
        x = self.dropout1(F.relu(self.conv1(x)))
        x = self.dropout2(F.relu(self.conv2(x)))
        x = self.pool(x)
        z = self.flat(x)

        y = self.out_layers(z)
        return y
