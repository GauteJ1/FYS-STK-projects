import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json 
import sys
import numpy as np

from neural_network import NeuralNetwork
from cost_pinn import total_cost
from data_gen import RodDataGen
from analytic import exact_sol


### CODE FOR TRAINING THE NEURAL NETWORK MODEL ###


def train_loop(dataloader, model_nn, loss_fn, optimizer):

    """
    Training loop for the neural network model

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing the training data
    model_nn : NeuralNetwork
        Neural network model
    loss_fn : function
        Loss function for the neural network model
    optimizer : torch.optim
        Optimizer for the neural network model
    """
    
    model_nn.train()  
    epoch_loss_train = 0

    for x_batch, t_batch in dataloader:
        
        optimizer.zero_grad()
        loss = loss_fn(x_batch, t_batch, model_nn)
        loss.backward()
        optimizer.step()

        epoch_loss_train += loss.item()


def test_loop(dataloader, model_nn, loss_fn):

    """
    Testing loop for the neural network model

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing the testing data
    model_nn : NeuralNetwork
        Neural network model
    loss_fn : function
        Loss function for the neural network model
    """
    
    model_nn.eval()
    epoch_loss_test = 0

    for x_batch, t_batch in dataloader:

        loss = loss_fn(x_batch, t_batch, model_nn)
        epoch_loss_test += loss.item()

def mse_against_analytic(model_nn, Nx=100, Nt=100):

    """
    Calculate the mean squared error (MSE) between the neural network model and the analytical solution

    Parameters
    ----------
    model_nn : NeuralNetwork
        Neural network model
    Nx : int
        Number of spatial points, default is 100
    Nt : int
        Number of temporal points, default is 100

    Returns
    -------
    mse : torch.Tensor
        Mean squared error between the neural network model and the analytical solution 

    """

    X = torch.linspace(0, 1, Nx + 1)
    T = torch.linspace(0, 0.5, Nt + 1)

    X, T = torch.meshgrid(X, T)
    X_ = X.flatten().reshape([(Nx + 1) * (Nt + 1), 1])
    T_ = T.flatten().reshape([(Nx + 1) * (Nt + 1), 1])

    Z = model_nn(X_, T_).detach().reshape([(Nx + 1), (Nt + 1)])
    
    Z_analytic = exact_sol(X, T)

    mse = torch.mean((Z - Z_analytic)**2)

    return mse

def train(seed, n_layers, value_layers, activation, return_model=False):

    """
    Train the neural network model
    Starts by printing the hyperparameters used for the training

    Parameters
    ----------
    seed : int
        Seed for the random number generator
    n_layers : int
        Number of hidden layers
    value_layers : int
        Size of the hidden layers
    activation : str
        Activation function for the hidden layers
    
    return_model : bool
        If True, return the trained model, default is False

    Returns
    -------
    mse_final : torch.Tensor
        Mean squared error between the neural network model and the analytical solution
    """

    print(f"Seed: {seed}, Hidden layers: {n_layers}, Value layers: {value_layers}, Activation: {activation}")

    layers = [initial_layer_size] + [value_layers] * (n_layers) + [final_layer_size]
    activations = [activation] * n_layers + ["identity"]

    model_nn = NeuralNetwork(layers, activations, initialization)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)

    # data 
    data = RodDataGen(Nx=Nx, Nt=Nt, T = 0.5)
    test_x, train_x, test_t, train_t = train_test_split(data.x, data.t, test_size=0.4)
    train_dataset = TensorDataset(train_x, train_t)
    test_dataset = TensorDataset(test_x, test_t)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    for epoch in tqdm(range(epochs)):
        train_loop(train_data, model_nn, total_cost, optimizer) 
        test_loop(test_data, model_nn, total_cost) 

    if return_model:
        return model_nn
    
    else:
        model_nn.eval()
        mse_final = mse_against_analytic(model_nn)
        return mse_final
    

### CODE FOR THE 3D GRID SEARCH ###

def grid_search(n_layers, value_layers, activation):

    """
    Perform a 3d grid search over three of the hyperparameters of the neural network model
    Saves the resulting mse and hyperparameters in a json file for each combination of hyperparameters

    Parameters
    ----------
    n_layers : list[int]
        List of number of hidden layers
    value_layers : list[int]
        List of size of hidden layers
    activation : list[str]
        List of activation functions for the hidden layers
    """

    grid_search_results = []

    for n_layers in layers_num:
        for value_layers in layer_sizes:
            for activation in activation_funcs:

                list_seeds = []

                for seed in seeds:
                    torch.manual_seed(seed)
                    
                    mse_final = train(seed, n_layers, value_layers, activation)

                    list_seeds.append(mse_final)

                mse_final = float(np.mean(list_seeds))
                grid_search_results.append({
                    "n_layers": n_layers,
                    "value_layers": value_layers,
                    "activation": activation,
                    "final_mse": mse_final
                })

    # save grid search results as json
    with open("../results/grid_search.json", "w") as f:
        json.dump(grid_search_results, f, indent=4)

### CODE FOR THE 1D SEARCHES ###

def activations_search(best_n_layers, best_value_layers, activation_funcs):

    """
    Using the best width and depth from the grid search, tests 10 models for each of the activation functions in the activation_funcs list
    Saving the final mse for each model in a json file

    Parameters
    ----------
    best_n_layers : int
        Best number of hidden layers from the 3d grid search
    best_value_layers : int
        Best size of hidden layers from the 3d grid search
    activation_funcs : list[str]
        List of activation functions for the hidden layers
    """
    
    # keeping n_layers and value_layers constant
    new_search_activations_results = []
    
    for activation in activation_funcs:
        print(f"Activation: {activation}")

        list_seeds_act = []

        for seed in seeds: 
            torch.manual_seed(seed)

            final_mse = train(seed, best_n_layers, best_value_layers, activation)
            list_seeds_act.append(float(final_mse))
            
        new_search_activations_results.append({
            "activation": activation,
            "final_mse": list_seeds_act
            })
            
    with open("../results/activation_search.json", "w") as f:
        json.dump(new_search_activations_results, f, indent=4)

def layers_search(best_n_layers, best_activation, layer_sizes):

    """
    Using the best depth and activation function from the grid search, tests 10 models for each of the hidden layer sizes (width) in the layer_sizes list
    Saving the final mse for each model in a json file

    Parameters
    ----------
    best_n_layers : int
        Best number of hidden layers from the 3d grid search
    best_activation : str
        Best activation function from the 3d grid search
    layer_sizes : list[int]
        List of size of hidden layers
    """

    # keeping n_layers and activation constant
    new_search_value_layers_results = []

    for value_layers in layer_sizes:
        print(f"Value layers: {value_layers}")

        list_seeds_val = []

        for seed in seeds: 
            torch.manual_seed(seed)

            final_mse = train(seed, best_n_layers, value_layers, best_activation)
            list_seeds_val.append(float(final_mse))
            
        new_search_value_layers_results.append({
            "value_layers": value_layers,
            "final_mse": list_seeds_val
            })
        
    with open("../results/value_layers_search.json", "w") as f:
        json.dump(new_search_value_layers_results, f, indent=4)

def n_layers_search(best_value_layers, best_activation, layers_num):

    """
    Using the best width and activation function from the grid search, tests 10 models for each of the number of hidden layers (depth) in the layers_num list
    Saving the final mse for each model in a json file

    Parameters
    ----------
    best_value_layers : int
        Best size of hidden layers from the 3d grid search
    best_activation : str
        Best activation function from the 3d grid search
    layers_num : list[int]
        List of number of hidden layers
    """

    # keeping value_layers and activation constant
    new_search_n_layers_results = []

    for n_layers in layers_num:
        print(f"Number of layers: {n_layers}")

        list_seeds_n = []

        for seed in seeds: 
            torch.manual_seed(seed)

            final_mse = train(seed, n_layers, best_value_layers, best_activation)
            list_seeds_n.append(float(final_mse))
            
        new_search_n_layers_results.append({
            "n_layers": n_layers,
            "final_mse": list_seeds_n
            })
        
    with open("../results/n_layers_search.json", "w") as f:
        json.dump(new_search_n_layers_results, f, indent=4)


if __name__ == "__main__":

    seeds = [123] # for the 3d grid search
    layers_num = [1, 2, 3] # number of hidden layers 
    layer_sizes = [10, 25, 50, 100] # size of hidden layers
    activation_funcs = ["leakyReLU", "ReLU", "tanh", "sigmoid"]

    initial_layer_size = 2
    final_layer_size = 1

    learning_rate = 1e-3
    epochs = 1
    batch_size = 3000
    Nx = 100
    Nt = 100

    initialization = "xavier"

    if len(sys.argv) > 1:

        if "grid" in sys.argv:
            print("Running 3d grid search")
            grid_search(layers_num, layer_sizes, activation_funcs)

        print("Loading best hyperparameters from 3d grid search")
        with open("../results/grid_search.json", "r") as f:
            grid_search_results = json.load(f)

        best_result = min(grid_search_results, key=lambda x: x['final_mse'])
        best_n_layers = best_result['n_layers']
        best_value_layers = best_result['value_layers']
        best_activation = best_result['activation']

        seeds = [981, 123, 42, 7, 81, 23, 1, 261, 928, 77]

        if "activations" in sys.argv:
            print("Running 1d search for activation functions")
            activations_search(best_n_layers, best_value_layers, activation_funcs)

        if "value_layers" in sys.argv:
            print("Running 1d search for hidden layer sizes")
            layers_search(best_n_layers, best_activation, layer_sizes)

        if "n_layers" in sys.argv:
            print("Running 1d search for number of hidden layers")
            n_layers_search(best_value_layers, best_activation, layers_num)

    elif len(sys.argv) == 1: 

        print("No command line arguments given")
        print(f"Running 3d grid search and three 1d searches with the best parameters from the grid search")
        print(f"To only run some of the searches, use one or more of the following command line arguments: 'grid', 'activations', 'value_layers', 'n_layers'")

        print("3d grid search")
        grid_search(layers_num, layer_sizes, activation_funcs)

        print("Loading best hyperparameters from 3d grid search")
        with open("../results/grid_search.json", "r") as f:
            grid_search_results = json.load(f)

            best_result = min(grid_search_results, key=lambda x: x['final_mse'])
            best_n_layers = best_result['n_layers']
            best_value_layers = best_result['value_layers']
            best_activation = best_result['activation']

        seeds = [981, 123, 42, 7, 81, 23, 1, 261, 928, 77]
        
        print("Running 1d searches")
        activations_search(best_n_layers, best_value_layers, activation_funcs)
        layers_search(best_value_layers, best_activation, layer_sizes)
        n_layers_search(best_value_layers, best_activation, layers_num)
