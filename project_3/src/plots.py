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

def train_loop(dataloader, model_nn, loss_fn, optimizer):
    
    model_nn.train()  
    epoch_loss_train = 0

    for x_batch, t_batch in dataloader:
        
        optimizer.zero_grad()
        loss = loss_fn(x_batch, t_batch, model_nn)
        loss.backward()
        optimizer.step()

        epoch_loss_train += loss.item()

    #return epoch_loss_train / len(dataloader)  


def test_loop(dataloader, model_nn, loss_fn):
    
    model_nn.eval()
    epoch_loss_test = 0

    for x_batch, t_batch in dataloader:

        loss = loss_fn(x_batch, t_batch, model_nn)
        epoch_loss_test += loss.item()

    #return epoch_loss_test / len(dataloader)  

def mse_against_analytic(model_nn):

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
        
    # loss_history_train = []
    # loss_history_test = []

    for epoch in tqdm(range(epochs)):
        train_loop(train_data, model_nn, total_cost, optimizer) # train_loss = 
        # loss_history_train.append(train_loss)

        test_loop(test_data, model_nn, total_cost) # test_loss = 
        # loss_history_test.append(test_loss)

        # check if test loss is increasing the 10 last epochs
        # if epoch > 10: 
        #     if (loss_history_test[-1] > loss_history_test[-11]):
        #         print("Early stopping")
        #         break

    if return_model:
        return model_nn
    else:
        model_nn.eval()
        mse_final = mse_against_analytic(model_nn)
        return mse_final


def box_plot(data_mse, varying_param):
    labels = []
    values = [] # final mse for different seeds 
    number_of_seeds = len(data_mse)

    if varying_param == "activation":
        title = "activation functions"
    elif varying_param == "value_layers":
        title = "size of hidden layers"
    elif varying_param == "n_layers":
        title = "number of hidden layers"
    else: 
        raise ValueError("Invalid varying parameter, must be 'activation', 'value_layers' or 'n_layers'")

    for result in data_mse:
        labels.append(result[varying_param])
        values.append(result["final_mse"])

    plt.boxplot(values, labels=labels, patch_artist=True)
    plt.xlabel(f"{title}")
    plt.ylabel("Final MSE")
    plt.title(f"Final MSE for different {title}, {number_of_seeds} seeds")
    plt.savefig(f"../plots/{varying_param}_search.png")
    plt.show()


def heatmaps(model_nn, model_fd):
    fig, axs = plt.subplots(3, 1, figsize=(5, 15))
    x_ex = np.linspace(0, 1, 1000)
    t_ex = np.linspace(0, 0.5, 1000)
    X_ex, T_ex = np.meshgrid(x_ex, t_ex)
    Z_ex = exact_sol(X_ex, T_ex)

    axs[0].contourf(X_ex, T_ex, Z_ex, cmap="hot", levels=500, vmin=0, vmax=1)
    axs[0].axhline(y=0.03, color="cyan", linestyle="-")
    axs[0].axhline(y=0.25, color="cyan", linestyle="-")
    axs[0].set_title("Analytical Solution")
    axs[0].set_xlabel("Position x []")
    axs[0].set_ylabel("Time [s]")

    # neural network
    Nx_nn = 100
    Nt_nn = 100

    X_nn = torch.linspace(0, 1, Nx_nn + 1)
    T_nn = torch.linspace(0, 0.5, Nt_nn + 1)

    X_nn, T_nn = torch.meshgrid(X_nn, T_nn)
    X_nn_ = X_nn.flatten().reshape([(Nx_nn + 1) * (Nt_nn + 1), 1])
    T_nn_ = T_nn.flatten().reshape([(Nx_nn + 1) * (Nt_nn + 1), 1])

    Z_nn = model_nn(X_nn_, T_nn_).detach().reshape([(Nx_nn + 1), (Nt_nn + 1)])

    axs[1].contourf(X_nn, T_nn, Z_nn, cmap="hot", levels=500, vmin=0, vmax=1)
    axs[1].axhline(y=0.03, color="cyan", linestyle="-")
    axs[1].axhline(y=0.25, color="cyan", linestyle="-")
    axs[1].set_title("Neural Network Solution")
    axs[1].set_xlabel("Position x []")
    axs[1].set_ylabel("Time [s]")

    # finite difference
    X_fd = np.linspace(0, 1, N + 1)
    T_fd = model_fd["time_steps"]
    X_fd, T_fd = np.meshgrid(X_fd, T_fd)
    Z_fd = model_fd["values"]

    axs[2].contourf(X_fd, T_fd, Z_fd, cmap="hot", levels=500, vmin=0, vmax=1)
    axs[2].axhline(y=0.03, color="cyan", linestyle="-")
    axs[2].axhline(y=0.25, color="cyan", linestyle="-")
    axs[2].set_title("Finite Difference Solution")
    axs[2].set_xlabel("Position x []")
    axs[2].set_ylabel("Time [s]")

    # common colorbar
    # fig.colorbar(axs[0].contourf(X, T, Z, cmap="hot", levels=500, vmin=0, vmax=1), ax=axs, orientation="horizontal", label="Temperature")

    plt.tight_layout()
    plt.savefig("../plots/heat_map_comparison.png")
    plt.show()

    print("MSE for neural network compared to analytical solution: ", np.mean((np.array(Z_nn) - np.array(exact_sol(X_nn, T_nn)))**2))
    print("MSE for finite difference compared to analytical solution: ", np.mean((np.array(Z_fd) - np.array(exact_sol(X_fd, T_fd)))**2))

def time_slices(model_nn, model_fd):
    # plot x for t = 0.03 and t = 0.25 for all three methods
    # 2x1 subplot, one for each time slice

    t1 = 0.03
    t2 = 0.25

    fig, axs = plt.subplots(2, 1, figsize=(8, 15))

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

    # finite difference
    X = np.linspace(0, 1, N + 1)  
    Y = np.array(model_fd["time_steps"])  
    n = len(Y)
    Z = np.array(model_fd["values"])  
   
    axs[0].plot(X, Z[int(t1 * n), :], label="Finite Difference", color = "hotpink", linestyle=":")
    axs[1].plot(X, Z[int(t2 * n), :], label="Finite Difference", color = "hotpink", linestyle=":")


    axs[0].set_title("Time slice at t = 0.03")
    axs[1].set_title("Time slice at t = 0.25")

    axs[0].set_xlabel("Position x []")
    axs[1].set_xlabel("Position x []")

    axs[0].set_ylabel("Temperature [Celsius]")
    axs[1].set_ylabel("Temperature [Celsius]")

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("../plots/time_slices_comparison.png")
    plt.show()




if __name__ == "__main__":

    seed = 123
    initial_layer_size = 2
    final_layer_size = 1
    initialization = "xavier"
    learning_rate = 1e-3
    epochs = 1000
    batch_size = 3000
    Nx = 100
    Nt = 100

    # pass "boxplots" as argument to make boxplots
    # pass "heatmaps" as argument to make heatmaps

    if len(sys.argv) == 1:
        print("No argument given, pass 'boxplots' and/or 'heatmaps' and/or 'time' as argument to make plots")
        sys.exit(1)

    else:

        if "train_best" in sys.argv:

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
            N = 10 
            dt = 0.001
            init = lambda x: np.sin(np.pi * x)
            model = ForwardEuler(N, dt)
            model.set_init(init)
            model_fd = model(0.5)

            # make plot of all three heatmaps
            heatmaps(model_nn, model_fd)

        if "boxplots" in sys.argv:

            plt.style.use("../plot_settings.mplstyle")
            sns.set()

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

            plt.style.use("../plot_settings.mplstyle")
            sns.set()

            model_nn = torch.load("../results/best_model.pt")
            
            # finite difference method
            N = 10 
            dt = 0.001
            init = lambda x: np.sin(np.pi * x)
            model = ForwardEuler(N, dt)
            model.set_init(init)
            model_fd = model(0.5)

            time_slices(model_nn, model_fd)