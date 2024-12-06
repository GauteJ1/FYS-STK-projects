import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from neural_network import NeuralNetwork
from cost_pinn import total_cost
from data_gen import RodDataGen

torch.manual_seed(123)

learning_rate = 1e-3
epochs = 1000
batch_size = 3000
Nx = 100
Nt = 100

model_nn = NeuralNetwork([2, 50, 50, 50, 1], ["tanh", "tanh", "sigmoid"], "he")

optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)

# data 
data = RodDataGen(Nx=Nx, Nt=Nt, T = 0.5)
test_x, train_x, test_t, train_t = train_test_split(data.x, data.t, test_size=0.4)
train_dataset = TensorDataset(train_x, train_t)
test_dataset = TensorDataset(test_x, test_t)

train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train_loop(dataloader, model_nn, loss_fn, optimizer):
    
    model_nn.train()  
    epoch_loss_train = 0

    for x_batch, t_batch in dataloader:
        
        optimizer.zero_grad()
        loss = loss_fn(x_batch, t_batch, model_nn)
        loss.backward()
        optimizer.step()

        epoch_loss_train += loss.item()

    return epoch_loss_train / len(dataloader)  


def test_loop(dataloader, model_nn, loss_fn):
    
    model_nn.eval()
    epoch_loss_test = 0

    for x_batch, t_batch in dataloader:

        loss = loss_fn(x_batch, t_batch, model_nn)
        epoch_loss_test += loss.item()

    return epoch_loss_test / len(dataloader)  


def plot_heatmap(nn_model):

    Nx = 100
    Nt = 100

    X = torch.linspace(0, 1, Nx + 1)
    T = torch.linspace(0, 0.5, Nt + 1)

    X, T = torch.meshgrid(X, T)
    X_ = X.flatten().reshape([(Nx + 1) * (Nt + 1), 1])
    T_ = T.flatten().reshape([(Nx + 1) * (Nt + 1), 1])

    Z = nn_model(X_, T_).detach().reshape([(Nx + 1), (Nt + 1)])

    plt.contourf(X, T, Z, cmap="hot", levels=500, vmin=0, vmax=1)

    plt.colorbar()
    plt.savefig("../plots/heat_map.png")
    plt.show()

if __name__ == "__main__":
    
    loss_history_train = []
    loss_history_test = []

    for epoch in tqdm(range(epochs)):
        train_loss = train_loop(train_data,  model_nn, total_cost, optimizer)
        loss_history_train.append(train_loss)

        test_loss = test_loop(test_data, model_nn, total_cost)
        loss_history_test.append(test_loss)

        # check if test loss is increasing the 10 last epochs
        if epoch > 10: 
            if (loss_history_test[-1] > loss_history_test[-11]):
                print("Early stopping")
                break

    start_idx = 100
    epochs_list = list(range(start_idx, len(loss_history_train)))
    plt.plot(epochs_list, loss_history_train[start_idx:], label="Training loss")
    plt.plot(epochs_list, loss_history_test[start_idx:], label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss over epochs")
    plt.legend()
    plt.show()

    model_nn.eval()
    plot_heatmap(model_nn)