import torch.nn as nn
import torch
from model import activations


class ClassificationDNN(nn.Module):

    def __init__(self, in_size, hsize, out_size, nlayers, activation='ReLU'):
        super(ClassificationDNN, self).__init__()

        self.act = activations[activation]
        self.in_linear = nn.Linear(in_size, hsize)
        self.hidden_layers = nn.ModuleList()
        for _ in range(nlayers):
            self.hidden_layers.append(nn.Linear(hsize, hsize))
        self.out_linear = nn.Linear(hsize, out_size)
        # Softmax across the last dimension (The inferenece not the batch dim)
        self.probs = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.act(self.in_linear(x))
        for hidden_layer in self.hidden_layers:
            x = self.act(hidden_layer(x))
        return self.probs(self.out_linear(x))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    line = torch.linspace(-5, 10, 200)
    fig, ax = plt.subplots(1, 1)
    elu6 = ELU6()
    plt.plot(elu6(line))
    plt.show()
