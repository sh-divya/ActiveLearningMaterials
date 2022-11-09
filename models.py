import torch
import torch.nn as nn

class ProxyMLP(nn.Module):
    
    def __init__(self, in_feat, hidden_layers):
        self.nn_layers = []

        for i in range(len(hidden_layers) - 1):
            if i == 0:
                self.nn_layers.append(nn.Linear(in_feat, hidden_layers[i]))
            else:
                self.nn_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        self.nn_layers.append(nn.Linear(hidden_layers[-1], 1))
        self.nn_layers = nn.ModuleList(self.nn_layers)
        self.hidden_act = nn.ReLU()
        self.final_act = nn.Sigmoid()

        def forward(self, x):
            act = self.hidden_act
            for l, layer in enumerate(self.nn_layers):
                if l == len(self.nn_layers) - 1:
                    act = self.final_act
                x = act(layer(x))

            return x
