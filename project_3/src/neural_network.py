import torch
from torch import nn

class NeuralNetwork(nn.Module):

    def __init__(self, layers, activations, init):

        """ 
        Neural network model 

        Parameters
        ----------
        layers : list[int]
            List of layer sizes, including input and output layer

        activations : list[str]
            List of activation functions as str, one for each layer

        init : str
            Initialization method for the weights, "xavier" or "he"
        """

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
                    case "identity": 
                        layers_list.append(nn.Identity())

        self.nn_model = nn.Sequential(*layers_list)

        def init_weights(m):
            """
            Initialize the weights of the neural network

            Parameters
            ----------
            m : torch.nn.Module
                Module of the neural network

            Raises
            ------
            ValueError
                If the initialization method is not valid
            """
            if init == "xavier" or init == None:
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
            elif init == "he":
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight)
            else:
                raise ValueError(f"{init} is not a valid initialization")

        self.nn_model.apply(init_weights)

    def forward(self, x, t):

        """
        Forward pass of the neural network

        Parameters
        ----------
        x : torch.Tensor
            Position
        t : torch.Tensor
            Time

        Returns
        -------
        logits : torch.Tensor
            Output of the neural network
        """
        
        X = torch.cat((x,t),axis = 1)
        logits = self.nn_model(X)
        
        return logits
