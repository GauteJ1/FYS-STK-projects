import matplotlib.pyplot as plt
import numpy as np
import json
import torch 
import sys

from analytic import exact_sol
from data_gen import RodDataGen
from neural_network import NeuralNetwork
from finite_difference import ForwardEuler


def box_plot(data_loss, varying_param):
    labels = []
    values = [] # final loss for different seeds 
    number_of_seeds = len(data_loss)

    if varying_param == "activation":
        title = "activation functions"
    elif varying_param == "value_layers":
        title = "size of hidden layers"
    elif varying_param == "n_layers":
        title = "number of hidden layers"
    else: 
        raise ValueError("Invalid varying parameter, must be 'activation', 'value_layers' or 'n_layers'")

    for result in data_loss:
        labels.append(result[varying_param])
        values.append(result["final_loss"])

    plt.boxplot(values, labels=labels, patch_artist=True)
    plt.xlabel(f"{title}")
    plt.ylabel("Final loss")
    plt.title(f"Final loss for different {title}, {number_of_seeds} seeds")
    plt.savefig(f"../plots/{varying_param}_search.png")
    plt.show()


def heatmaps(model_nn, model_fd):
    fig, axs = plt.subplots(3, 1, figsize=(5, 15))

    data = RodDataGen(Nx=10, Nt=10, L=1, T=0.5)

    # exact solution
    x = data.x
    t = data.t

    Nx = len(x) - 1
    Nt = len(t) - 1

    X, T = np.meshgrid(np.linspace(0, 1, Nx + 1), np.linspace(0, data.T, Nt + 1))
    Z = exact_sol(X, T)

    axs[0].contourf(X, T, Z, cmap="hot", levels=500, vmin=0, vmax=1)
    axs[0].axhline(y=0.03, color="blue", linestyle="-")
    axs[0].axhline(y=0.25, color="blue", linestyle="-")
    axs[0].set_title("Analytical Solution")
    axs[0].set_xlabel("Position x []")
    axs[0].set_ylabel("Time [s]")

    # neural network
    Nx = 100
    Nt = 100

    X = torch.linspace(0, 1, Nx + 1)
    T = torch.linspace(0, 0.5, Nt + 1)

    X, T = torch.meshgrid(X, T)
    X_ = X.flatten().reshape([(Nx + 1) * (Nt + 1), 1])
    T_ = T.flatten().reshape([(Nx + 1) * (Nt + 1), 1])

    Z = model_nn(X_, T_).detach().reshape([(Nx + 1), (Nt + 1)])

    axs[1].contourf(X, T, Z, cmap="hot", levels=500, vmin=0, vmax=1)
    axs[1].axhline(y=0.03, color="blue", linestyle="-")
    axs[1].axhline(y=0.25, color="blue", linestyle="-")
    axs[1].set_title("Neural Network Solution")
    axs[1].set_xlabel("Position x []")
    axs[1].set_ylabel("Time [s]")

    # finite difference
    X = np.linspace(0, 1, N + 1)
    Y = model_fd["time_steps"]
    X, Y = np.meshgrid(X, Y)
    Z = model_fd["values"]

    axs[2].contourf(X, Y, Z, cmap="hot", levels=500, vmin=0, vmax=1)
    axs[2].axhline(y=0.03, color="blue", linestyle="-")
    axs[2].axhline(y=0.25, color="blue", linestyle="-")
    axs[2].set_title("Finite Difference Solution")
    axs[2].set_xlabel("Position x []")
    axs[2].set_ylabel("Time [s]")

    # common colorbar
    # fig.colorbar(axs[0].contourf(X, T, Z, cmap="hot", levels=500, vmin=0, vmax=1), ax=axs, orientation="horizontal", label="Temperature")

    plt.tight_layout()
    plt.savefig("../plots/heat_map_comparison.png")
    plt.show()

def time_slices(model_nn, model_fd):
    # plot x for t = 0.03 and t = 0.25 for all three methods
    # 2x1 subplot, one for each time slice

    t1 = 0.03
    t2 = 0.25

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    # exact solution
    data = RodDataGen(Nx=100, Nt=100, L=1, T=0.5)
    x = data.x
    t = data.t

    Nx = len(x) - 1
    Nt = len(t) - 1

    X, T = np.meshgrid(np.linspace(0, 1, Nx + 1), np.linspace(0, data.T, Nt + 1))
    Z = exact_sol(X, T)


    axs[0].plot(x, Z[int(t1*Nt), :], label="Analytical")
    axs[1].plot(x, Z[int(t2*Nt), :], label="Analytical")

    # # neural network
    # Nx = 100
    # Nt = 100

    # X = torch.linspace(0, 1, Nx + 1)
    # T = torch.linspace(0, 0.5, Nt + 1)

    # X, T = torch.meshgrid(X, T)
    # X_ = X.flatten().reshape([(Nx + 1) * (Nt + 1), 1])
    # T_ = T.flatten().reshape([(Nx + 1) * (Nt + 1), 1])

    # Z = model_nn(X_, T_).detach().reshape([(Nx + 1), (Nt + 1)])

    # axs[0].plot(x, Z[:, int(t1*Nt)], label="Neural Network")
    # axs[1].plot(x, Z[:, int(t2*Nt)], label="Neural Network")

    # # finite difference
    # X = np.linspace(0, 1, N + 1)
    # Y = model_fd["time_steps"]
    # X, Y = np.meshgrid(X, Y)
    # Z = model_fd["values"]

    # axs[0].plot(x, Z[:, int(t1*len(Y))], label="Finite Difference")
    # axs[1].plot(x, Z[:, int(t1*len(Y))], label="Finite Difference")

    # axs[0].set_title("Time slice at t = 0.03")
    # axs[1].set_title("Time slice at t = 0.25")

    # axs[0].set_xlabel("Position x []")
    # axs[1].set_xlabel("Position x []")

    # axs[0].set_ylabel("Temperature [Celsius]")
    # axs[1].set_ylabel("Temperature [Celsius]")

    # axs[0].legend()
    # axs[1].legend()

    # plt.tight_layout()
    plt.savefig("../plots/time_slices_comparison.png")
    plt.show()




if __name__ == "__main__":

    # pass "boxplots" as argument to make boxplots
    # pass "heatmaps" as argument to make heatmaps

    if len(sys.argv) == 1:
        print("No argument given, pass 'boxplots' and/or 'heatmaps' as argument to make plots")
        sys.exit(1)

    else:

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

        if "heatmaps" in sys.argv:

            with open("../results/grid_search.json", "r") as f:
                grid_search_results = json.load(f)

            # best neural network model
            best_result = min(grid_search_results, key=lambda x: x['final_loss'])
            best_n_layers = best_result['n_layers']
            best_value_layers = best_result['value_layers']
            best_activation = best_result['activation']

            model_nn = NeuralNetwork([2] + [best_value_layers]*best_n_layers + [1], [best_activation]*best_n_layers + ["sigmoid"], "he")

            # finite difference method
            N = 10 
            dt = 0.001
            init = lambda x: np.sin(np.pi * x)
            model = ForwardEuler(N, dt)
            model.set_init(init)
            model_fd = model(0.5)

            # make plot of all three heatmaps
            heatmaps(model_nn, model_fd)

        if "time" in sys.argv: 

            with open("../results/grid_search.json", "r") as f:
                grid_search_results = json.load(f)

            # best neural network model
            best_result = min(grid_search_results, key=lambda x: x['final_loss'])
            best_n_layers = best_result['n_layers']
            best_value_layers = best_result['value_layers']
            best_activation = best_result['activation']

            model_nn = NeuralNetwork([2] + [best_value_layers]*best_n_layers + [1], [best_activation]*best_n_layers + ["sigmoid"], "he")

            # finite difference method
            N = 10 
            dt = 0.001
            init = lambda x: np.sin(np.pi * x)
            model = ForwardEuler(N, dt)
            model.set_init(init)
            model_fd = model(0.5)

            time_slices(model_nn, model_fd)