import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_network import NeuralNetwork
from cost_pinn import total_cost

from data_gen import RodDataGen

learning_rate = 1e-3
batch_size = 64
epochs = 10

model_nn = NeuralNetwork([2, 20, 20, 1], ["ReLU", "ReLU", "ReLU"], "xavier")

optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)

training_data = (RodDataGen.x, RodDataGen.t)
train_dataloader = DataLoader(training_data, batch_size=64)

def train_loop(dataloader, model_nn, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model_nn to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model_nn.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model_nn(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")