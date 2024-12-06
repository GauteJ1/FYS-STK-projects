import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, layers, activations, init):
        super().__init__()
        self.flatten = nn.Flatten()

        layers_list = []
        for i in range(len(layers) - 1):

            layers_list.append(nn.Linear(layers[i], layers[i + 1]))

            if i < len(activations):
                match activations[i]:
                    case "ReLU":
                        layers_list.append(nn.ReLU())
                    case "sigmoid":
                        layers_list.append(nn.Sigmoid())
                    case "tanh":
                        layers_list.append(nn.Tanh())
                    case "leakyReLU":
                        layers_list.append(nn.LeakyReLU())

        self.nn_model = nn.Sequential(*layers_list)

        def init_weights(m):
            if init == "xavier" or init == None:
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
            elif init == "he":
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight)
            else:
                raise ValueError(f"{init} is not a valid initialization")

        self.nn_model.apply(init_weights)

    def forward(self, x, t):

        X = torch.cat((x,t),axis = 1)
        logits = self.nn_model(X)
        
        return logits
