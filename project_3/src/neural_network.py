import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, layers, activations, init):
        super().__init__()
        self.flatten = nn.Flatten()

        # Create layers from input list
        layers = []
        for i in range(len(layers) - 1):
            layers.append(nn.Linear(layers[i], layers[i + 1]))

            if i < len(activations):
                match activations[i]:
                    case "ReLU":
                        layers.append(nn.ReLU())
                    case "sigmoid":
                        layers.append(nn.Sigmoid())
                    case "tanh":
                        layers.append(nn.Tanh())

        self.nn_model = nn.Sequential(*layers)

        def init_weights(m):
            if init == "xavier" or init == None:
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform(m.weight)
            elif init == "he":
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_normal(m.weight)
            else:
                raise ValueError(f"{init} is not a valid initialization")

        self.nn_model.apply(init_weights)

    def forward(self, x, t):
        x = self.flatten(x)
        t = self.flatten(t)
        data = torch.cat((x, t), dim=1)
        logits = self.nn_model(data)
        return logits
