import numpy as np
import matplotlib.pyplot as plt
from data_gen import RodDataGen
import json 

def exact_sol(x, t):
    # Analytical solution for the heat equation
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

def plot_heatmap(data):

    x = data.x
    t = data.t

    Nx = len(x) - 1
    Nt = len(t) - 1

    X, T = np.meshgrid(np.linspace(0, 1, Nx + 1), np.linspace(0, data.T, Nt + 1))
    Z = exact_sol(X, T)

    plt.figure(figsize=(8, 6))
    heatmap = plt.contourf(X, T, Z, cmap="hot", levels=500, vmin=0, vmax=1)
    plt.colorbar(heatmap, label="Temperature")
    plt.xlabel("Position (x)")
    plt.ylabel("Time (t)")
    plt.title("Heatmap of Analytical Solution")
    plt.savefig("../plots/heat_map_analytic.png")
    plt.show()
    

if __name__ == "__main__":
    data = RodDataGen(Nx=100, Nt=100, L=1, T=0.5)
    plot_heatmap(data)

