import torch.nn as nn

activations = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ELU': nn.ELU(),
    'PReLU': nn.PReLU(),
    'SELU': nn.SELU()
}


class DNN(nn.Module):
    def __init__(self, in_size, hsize, out_size, nlayers, activation='ReLU'):
        super(DNN, self).__init__()

        self.act = activations[activation]
        self.in_linear = nn.Linear(in_size, hsize)
        self.hidden_layers = nn.ModuleList()
        for _ in range(nlayers):
            self.hidden_layers.append(nn.Linear(hsize, hsize))
        self.out_linear = nn.Linear(hsize, out_size)

    def forward(self, x):
        x = self.act(self.in_linear(x))
        for hidden_layer in self.hidden_layers:
            x = self.act(hidden_layer(x))
        return self.out_linear(x)
