import matplotlib.pyplot as plt
import numpy as np
import json
import torch 
import sys
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from analytic import exact_sol
from data_gen import RodDataGen
from neural_network import NeuralNetwork
from finite_difference import ForwardEuler
from cost_pinn import total_cost
from data_gen import RodDataGen
from grid_search import train_loop, test_loop, mse_against_analytic


### Functions to be able to train the best model and save it given the best parameters from the grid search ###

def mse_against_analytic_nn(model_nn, Nx=100, Nt=100):

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

def train(seed, n_layers, value_layers, activation, return_model=False, Nx = 100, Nt = 100):

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
    Nx : int
        Number of spatial points, default is 100
    Nt : int
        Number of temporal points, default is 100

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
    test_x, train_x, test_t, train_t = train_test_split(data.x, data.t, test_size=0.4) # we only need train data, but here in case one wants to study the loss for the test data
    train_dataset = TensorDataset(train_x, train_t)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(epochs)):
        train_loop(train_data, model_nn, total_cost, optimizer) 

    if return_model:
        return model_nn
    
    else:
        model_nn.eval()
        mse_final = mse_against_analytic(model_nn)
        return mse_final




### Plotting code ### 

def box_plot(data_mse, varying_param):

    """
    Create boxplots for the final mean squared error (MSE) for different values of a hyperparameter
    Must be "activation", "value_layers" or "n_layers"

    Save the plot as a pdf file

    Parameters
    ----------
    data_mse : list[dict]
        List of dictionaries containing the hyperparameters and the final MSE

    varying_param : str
        The hyperparameter that is varied, must be "activation", "value_layers" or "n_layers"
    """

    sns.set_theme()
    plt.style.use("../plot_settings.mplstyle")

    labels = []
    values = [] 

    if varying_param == "activation":
        title = "activation functions"
        x_label = "Activation function"
        plt.figure(figsize=(10, 8)) 

    elif varying_param == "value_layers":
        title = "size of hidden layers"
        plt.figure(figsize=(10, 7)) 
        plt.ylim(0, 0.0015)
        x_label = "Size of hidden layers"
    
    elif varying_param == "n_layers":
        title = "number of hidden layers"
        plt.figure(figsize=(10, 7)) 
        x_label = "Number of hidden layers"
    
    else: 
        raise ValueError("Invalid varying parameter, must be 'activation', 'value_layers' or 'n_layers'")

    for result in data_mse:
        labels.append(result[varying_param])
        values.append(result["final_mse"])

    # making the boxes pink with black edges
    plt.boxplot(values, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="hotpink", color="black"),
                capprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                flierprops=dict(color="hotpink", markeredgecolor="black"),
                medianprops=dict(color="black"))
    
    plt.xlabel(f"{x_label}")
    plt.ylabel("Final MSE")
    plt.title(f"Final MSE for {title}")
    plt.savefig(f"../plots/{varying_param}_search.pdf")
    plt.show()

    # print the mean of the final mse for each parameter
    for i in range(len(labels)):
        print(f"Mean MSE for {varying_param} = {labels[i]}: {np.mean(values[i])}")


def heatmaps(model_nn, model_fd, model_fd_100):

    """
    Plot heatmaps of the analytical solution, the neural network solution and the finite difference solution for N = 10
    Printing the MSE for the neural network compared to the analytical solution 
    Printing the MSE for the finite difference solution for N = 10 and N = 100 

    Save the plot as a png file

    Parameters
    ----------
    model_nn : NeuralNetwork
        Neural network model
    model_fd : dict
        Finite difference solution for N = 10
    model_fd_100 : dict
        Finite difference solution for N = 100  
    """

    plt.style.use("../plot_settings.mplstyle")
    plt.rcParams.update({"axes.grid": False}) # turning off grid

    # colormap settings
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = "hot"

    fig, axs = plt.subplots(3, 1, figsize=(10, 22), sharex=True, sharey=True, constrained_layout=False)

    # analytical solution
    x_ex = np.linspace(0, 1, 1000)
    t_ex = np.linspace(0, 0.5, 1000)
    X_ex, T_ex = np.meshgrid(x_ex, t_ex)
    Z_ex = exact_sol(X_ex, T_ex)

    im0 = axs[0].contourf(X_ex, T_ex, Z_ex, cmap=cmap, levels=500, norm=norm)
    axs[0].axhline(y=0.03, color="cyan", linestyle="-")
    axs[0].axhline(y=0.40, color="cyan", linestyle="-")
    axs[0].set_title("Analytical Solution")
    axs[0].set_xlabel("Position x []")
    axs[0].set_ylabel("Time [s]")

    # NN solution
    Nx_nn = 100
    Nt_nn = 100
    X_nn = torch.linspace(0, 1, Nx_nn + 1)
    T_nn = torch.linspace(0, 0.5, Nt_nn + 1)
    X_nn, T_nn = torch.meshgrid(X_nn, T_nn)
    X_nn_ = X_nn.flatten().reshape([(Nx_nn + 1) * (Nt_nn + 1), 1])
    T_nn_ = T_nn.flatten().reshape([(Nx_nn + 1) * (Nt_nn + 1), 1])
    Z_nn = model_nn(X_nn_, T_nn_).detach().reshape([(Nx_nn + 1), (Nt_nn + 1)])

    im1 = axs[1].contourf(X_nn, T_nn, Z_nn, cmap=cmap, levels=500, norm=norm)
    axs[1].axhline(y=0.03, color="cyan", linestyle="-")
    axs[1].axhline(y=0.40, color="cyan", linestyle="-")
    axs[1].set_title("Neural Network Solution")
    axs[1].set_xlabel("Position x []")
    axs[1].set_ylabel("Time [s]")

    # Finite difference solution
    X_fd = np.linspace(0, 1, 10 + 1)
    T_fd = model_fd["time_steps"]
    X_fd, T_fd = np.meshgrid(X_fd, T_fd)
    Z_fd = model_fd["values"]

    im2 = axs[2].contourf(X_fd, T_fd, Z_fd, cmap=cmap, levels=500, norm=norm)
    axs[2].axhline(y=0.03, color="cyan", linestyle="-")
    axs[2].axhline(y=0.40, color="cyan", linestyle="-")
    axs[2].set_title("Finite Difference Solution")
    axs[2].set_xlabel("Position x []")
    axs[2].set_ylabel("Time [s]")

    # common colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
    cbar = fig.colorbar(im2, cax=cbar_ax)
    #cbar.set_label("Temperature [normalized]") # taken out because it is cut of when saving the figure 

    plt.savefig("../plots/heat_map_comparison.png")
    plt.show()

    # fd for N = 100 as well, in order to calculate the MSE, not included in the plot
    X_fd_100 = np.linspace(0, 1, 100 + 1)
    T_fd_100 = model_fd_100["time_steps"]
    X_fd_100, T_fd_100 = np.meshgrid(X_fd_100, T_fd_100)
    Z_fd_100 = model_fd_100["values"]

    # printting MSE values
    print("MSE for neural network compared to analytical solution: ", np.mean((np.array(Z_nn) - np.array(exact_sol(X_nn, T_nn)))**2))
    print("MSE for finite difference with N = 10 compared to analytical solution: ", np.mean((np.array(Z_fd) - np.array(exact_sol(X_fd, T_fd)))**2))
    print("MSE for finite difference with N = 100 compared to analytical solution: ", np.mean((np.array(Z_fd_100) - np.array(exact_sol(X_fd_100, T_fd_100)))**2))

def time_slices(model_nn, model_fd, model_fd_100, t1 = 0.03, t2 = 0.40):

    """
    Plot temperature as a function of position at two different time slices t1 and t2
    for the analytical solution, the neural network solution and the finite difference solution for N = 10 and N = 100

    Save the plot as a pdf file

    Parameters
    ----------  
    model_nn : NeuralNetwork
        Neural network model
    model_fd : dict
        Finite difference solution for N = 10
    model_fd_100 : dict
        Finite difference solution for N = 100
    t1 : float
        Time slice 1, default is 0.03
    t2 : float
        Time slice 2, default is 0.40
    """

    sns.set_theme()
    plt.style.use("../plot_settings.mplstyle")

    fig, axs = plt.subplots(2, 1, figsize=(10, 15), sharex=True, sharey=True)

    # exact solution
    x = np.linspace(0, 1, 1000) # high number of points to have a smooth curve
    t = np.linspace(0, 0.5, 1000)
    X, T = np.meshgrid(x, t)
    Z = exact_sol(X, T)

    axs[0].plot(x, Z[int(t1*1000), :], label="Analytical", color="cornflowerblue")
    axs[1].plot(x, Z[int(t2*1000), :], label="Analytical", color="cornflowerblue")

    # neural network
    Nx = 100
    Nt = 100

    X = torch.linspace(0, 1, Nx + 1)
    T = torch.linspace(0, 0.5, Nt + 1)

    X, T = torch.meshgrid(X, T)
    X_ = X.flatten().reshape([(Nx + 1) * (Nt + 1), 1])
    T_ = T.flatten().reshape([(Nx + 1) * (Nt + 1), 1])

    Z = model_nn(X_, T_).detach().reshape([(Nx + 1), (Nt + 1)])

    axs[0].plot(np.linspace(0, 1, Nx + 1), Z[:,int(t1 * Nt)], label="Neural Network", color="orange", linestyle="--")
    axs[1].plot(np.linspace(0, 1, Nx + 1), Z[:,int(t2 * Nt)], label="Neural Network", color="orange", linestyle="--")

    # finite difference with N = 100

    X = np.linspace(0, 1, 100 + 1)  
    Y = np.array(model_fd_100["time_steps"])  
    n = len(Y)
    Z = np.array(model_fd_100["values"])  
   
    axs[0].plot(X, Z[int(t1 * n), :], label=r"Finite Difference, N = 101", color = "green", marker="o", linestyle="none", markersize=2)
    axs[1].plot(X, Z[int(t2 * n), :], label=r"Finite Difference, N = 101", color = "green", marker="o", linestyle="none", markersize=2)

  
    # finite difference with N = 10
    X = np.linspace(0, 1, 10 + 1)  
    Y = np.array(model_fd["time_steps"])  
    n = len(Y)
    Z = np.array(model_fd["values"])  
   
    axs[0].plot(X, Z[int(t1 * n), :], label=r"Finite Difference, N = 11", color = "hotpink", marker="*", linestyle="none", markersize=8)
    axs[1].plot(X, Z[int(t2 * n), :], label=r"Finite Difference, N = 11", color = "hotpink", marker="*", linestyle="none", markersize=8)

   
   # common settings for title and axis labels
    axs[0].set_title(r"Temperature as a function of position at $t_1$ = 0.03 s")
    axs[1].set_title(r"Temperature as a function of position at $t_2$ = 0.40 s")

    axs[0].set_xlabel("Position x")
    axs[1].set_xlabel("Position x")

    axs[0].set_ylabel("Temperature T")
    axs[1].set_ylabel("Temperature T")

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("../plots/time_slices_comparison.pdf")
    plt.show()




if __name__ == "__main__":

    seed = 123
    initial_layer_size = 2
    final_layer_size = 1
    initialization = "xavier"
    learning_rate = 1e-3
    epochs = 1000
    batch_size = 3000

    if len(sys.argv) == 1:
        print("No command line argument given, use 'train_best', 'boxplots', 'heatmaps' and/or 'time'")
        sys.exit(1)

    else:

        if "train_best" in sys.argv:

            torch.manual_seed(seed)

            with open("../results/grid_search.json", "r") as f:
                grid_search_results = json.load(f)

            with open("../results/grid_search.json", "r") as f:
                grid_search_results = json.load(f)

            # best neural network model
            best_result = min(grid_search_results, key=lambda x: x['final_mse'])
            best_n_layers = best_result['n_layers']
            best_value_layers = best_result['value_layers']
            best_activation = best_result['activation']

            model = train(seed, best_n_layers, best_value_layers, best_activation, return_model=True)
            # save model 
            torch.save(model, "../results/best_model.pt")

        if "heatmaps" in sys.argv:

            # nn model, load best model
            model_nn = torch.load("../results/best_model.pt")
            
            # finite difference method
            init = lambda x: np.sin(np.pi * x)
            model = ForwardEuler(N=10, dt=0.001)
            model.set_init(init)
            model_fd = model(0.5)

            model_100 = ForwardEuler(N=100)
            model_100.set_init(init)
            model_fd_100 = model_100(0.5)

            # plotting heatmaps
            heatmaps(model_nn, model_fd, model_fd_100)

        if "boxplots" in sys.argv:

            # activations
            with open("../results/activation_search.json", "r") as f:
                activation_search_results = json.load(f)

            box_plot(activation_search_results, "activation")

            # value layers
            with open("../results/value_layers_search.json", "r") as f:
                value_layers_search_results = json.load(f)

            box_plot(value_layers_search_results, "value_layers")

            # n layers
            with open("../results/n_layers_search.json", "r") as f:
                n_layers_search_results = json.load(f)

            box_plot(n_layers_search_results, "n_layers")

        if "time" in sys.argv: 

            model_nn = torch.load("../results/best_model.pt")
            
            # finite difference method
            init = lambda x: np.sin(np.pi * x)
            model = ForwardEuler(N=10, dt=0.001)
            model.set_init(init)
            model_fd = model(0.5)

            model_100 = ForwardEuler(N=100)
            model_100.set_init(init)
            model_fd_100 = model_100(0.5)

            # make time slice plots
            time_slices(model_nn, model_fd, model_fd_100)
