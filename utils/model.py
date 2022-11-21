import torch
import torch.nn as nn
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_shape, n_output):
        super(SimpleNN, self).__init__()

        self.ffl = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, n_output),
        )

    def forward(self, x):
        return self.ffl(x)

