import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from cost_pinn import total_cost
from data_gen import RodDataGen

learning_rate = 1e-3 
epochs = 500
batch_size = 1000

model_nn = NeuralNetwork([2, 20, 20, 1], ["ReLU", "ReLU", "ReLU"], "xavier")

optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)

# data 
data = RodDataGen(Nx = 100, Nt = 100)
train_dataset = TensorDataset(data.x, data.t)  
test_dataset = TensorDataset(data.x, data.t)   
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train_loop(data, model_nn, loss_fn, optimizer):

    x, t = data.x, data.t

    model_nn.train()
    # Backpropagation
    optimizer.zero_grad()
    loss = loss_fn(x, t, model_nn)
    loss.backward()
    optimizer.step()

    return loss.item()

def plot_heatmap(final_preds):
        
        N = 100 
        
        X = torch.linspace(0, 1, N + 1)
        Y = data["time_steps"]
        X, Y = torch.meshgrid(X, Y)
        Z = data["values"]

        plt.contourf(X, Y, Z, cmap="hot", levels=500, vmin=0, vmax=1)
        
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    loss = []
    for t in tqdm(range(epochs)):
        loss_loop = train_loop(data, model_nn, total_cost, optimizer)
        loss.append(loss_loop)

    final_preds = model_nn(data.x, data.t)

    plt.plot(loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.imshow(final_preds.detach().reshape(100, 100))
    plt.show()
