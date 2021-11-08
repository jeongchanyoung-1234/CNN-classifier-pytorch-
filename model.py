import torch.nn as nn
import torch.nn.functional as F

class FashionMNISTClassifier(nn.Module):
    def __init__(self, config):
        super(FashionMNISTClassifier, self).__init__()
        self.conv = nn.Conv2d(
                        in_channels=1,
                        out_channels=config.n_kernels,
                        kernel_size=config.kernel_size,
                        stride=1,
                        padding=1,
                    )

        self.pool = nn.MaxPool2d(kernel_size=config.kernel_size)
        self.flat = nn.Flatten()

        self.out_layers = nn.Sequential(
            nn.Linear(400, 10),
            nn.LogSoftmax(-1)
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        z = self.flat(x)

        y = self.out_layers(z)
        return y
