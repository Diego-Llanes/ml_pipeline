import torch.nn as nn
import torch.nn.functional as F
import torch


class ELU6(nn.Module):

    def __init__(self, alpha=1, window=6, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.window = window
        self.inplace = inplace

    def forward(self, x):
        cond1 = x < 0
        cond3 = x > self.window

        x = torch.where(cond1, F.elu(x, self.alpha, self.inplace), x)
        x = torch.where(cond3, self.window - F.elu(self.window - x, self.alpha, self.inplace), x)
        return x


activations = {
    'ReLU': nn.ReLU(),
    'ReLU6': nn.ReLU6(),
    'LeakyReLU': nn.LeakyReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ELU': nn.ELU(),
    'ELU6': ELU6(),
    'PReLU': nn.PReLU(),
    'SELU': nn.SELU()
}
