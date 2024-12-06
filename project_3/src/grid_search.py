import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json 
import sys

from neural_network import NeuralNetwork
from cost_pinn import total_cost
from data_gen import RodDataGen

#torch.manual_seed(123)

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


def train(seed, n_layers, value_layers, activation):

    print(f"Seed: {seed}, Layers: {n_layers}, Value layers: {value_layers}, Activation: {activation}")

    layers = [initial_layer_size] + [value_layers] * (n_layers - 2) + [final_layer_size]
    activations = [activation] * (n_layers - 1)

    model_nn = NeuralNetwork(layers, activations, initialization)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)

    # data 
    data = RodDataGen(Nx=Nx, Nt=Nt, T = 0.5)
    test_x, train_x, test_t, train_t = train_test_split(data.x, data.t, test_size=0.4)
    train_dataset = TensorDataset(train_x, train_t)
    test_dataset = TensorDataset(test_x, test_t)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
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

    model_nn.eval()

    final_loss = loss_history_test[-1]

    return final_loss

def plot_heatmap_nn(nn_model):

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

def grid_search(n_layers, value_layers, activation):
    grid_search_results = []

    for n_layers in layers_num:
        for value_layers in layer_sizes:
            for activation in activation_funcs:

                loss_seeds = []

                for seed in seeds:
                    torch.manual_seed(seed)

                    final_loss = train(seed, n_layers, value_layers, activation)

                    loss_seeds.append(final_loss)

                final_loss = sum(loss_seeds) / len(loss_seeds)
                grid_search_results.append({
                    "n_layers": n_layers,
                    "value_layers": value_layers,
                    "activation": activation,
                    "final_loss": final_loss
                })

    # save grid search results as json
    with open("../results/grid_search.json", "w") as f:
        json.dump(grid_search_results, f, indent=4)

def activations_search(best_n_layers, best_value_layers):
    
    # keeping n_layers and value_layers constant
    new_search_activations_results = []
    
    for activation in activation_funcs:
        print(f"Activation: {activation}")

        loss_seeds_act = []

        for seed in seeds: 
            torch.manual_seed(seed)

            final_loss = train(seed, best_n_layers, best_value_layers, activation)
            loss_seeds_act.append(final_loss)
            
        new_search_activations_results.append({
            "activation": activation,
            "final_loss": loss_seeds_act
            })
            
    with open("../results/activation_search.json", "w") as f:
        json.dump(new_search_activations_results, f, indent=4)

def layers_search(best_value_layers, best_activation):

    # keeping n_layers and activation constant
    new_search_value_layers_results = []

    for value_layers in layer_sizes:
        print(f"Value layers: {value_layers}")

        loss_seeds_val = []

        for seed in seeds: 
            torch.manual_seed(seed)

            final_loss = train(seed, best_value_layers, value_layers, best_activation)
            loss_seeds_val.append(final_loss)
            
        new_search_value_layers_results.append({
            "value_layers": value_layers,
            "final_loss": loss_seeds_val
            })

def n_layers_search(best_value_layers, best_activation):

    # keeping value_layers and activation constant
    new_search_n_layers_results = []

    for n_layers in layers_num:
        print(f"Number of layers: {n_layers}")

        loss_seeds_n = []

        for seed in seeds: 
            torch.manual_seed(seed)

            final_loss = train(seed, n_layers, best_value_layers, best_activation)
            loss_seeds_n.append(final_loss)
            
        new_search_n_layers_results.append({
            "n_layers": n_layers,
            "final_loss": loss_seeds_n
            })
        
    with open("../results/n_layers_search.json", "w") as f:
        json.dump(new_search_n_layers_results, f, indent=4)


if __name__ == "__main__":

    seeds = [123, 426, 47]
    layers_num = [2, 3, 4]
    layer_sizes = [10, 25, 50, 100]
    activation_funcs = ["tanh", "ReLU", "sigmoid", "leakyReLU"]

    initial_layer_size = 2
    final_layer_size = 1

    learning_rate = 1e-3
    epochs = 1000
    batch_size = 3000
    Nx = 100
    Nt = 100

    initialization = "he"

    if "grid" in sys.argv:
        grid_search(layers_num, layer_sizes, activation_funcs)

    else: 
        with open("../results/grid_search.json", "r") as f:
            grid_search_results = json.load(f)

        best_result = min(grid_search_results, key=lambda x: x['final_loss'])
        best_n_layers = best_result['n_layers']
        best_value_layers = best_result['value_layers']
        best_activation = best_result['activation']

    seeds = [981, 123, 42, 7, 81, 23, 1, 261, 928, 77]

    if "activations" in sys.argv:
        activations_search(best_n_layers, best_value_layers)

    if "value_layers" in sys.argv:
        layers_search(best_value_layers, best_activation)

    if "n_layers" in sys.argv:
        n_layers_search(best_value_layers, best_activation)

    if len(sys.argv) == 1: 

        grid_search(layers_num, layer_sizes, activation_funcs)
        activations_search(best_n_layers, best_value_layers)
        layers_search(best_value_layers, best_activation)
        n_layers_search(best_value_layers, best_activation)
