import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(1,20,3),
                nn.Tanh(),
                nn.Conv2d(20,10,3),
                nn.Tanh(),
                nn.Flatten(),
                nn.Linear(24*24*10, 100),
                nn.Tanh(),
                nn.Linear(100, 10)
                )
    def forward(self, x):
        return self.model(x)