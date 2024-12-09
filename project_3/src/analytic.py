import numpy as np
import matplotlib.pyplot as plt

from data_gen import RodDataGen 

def exact_sol(x, t):

    """
    Analytical solution for the heat equation

    Returns
    -------
    x : np.ndarray or torch.Tensor
        Position 
    t : np.ndarray or torch.Tensor
        Time

    Returns
    -------
    sol : np.ndarray or torch.Tensor
        Analytical solution of the heat equation
    """
    sol = np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    
    return sol

def plot_heatmap(data):
    """
    Plot heatmap of the analytical solution

    Parameters
    ----------
    data : RodDataGen
        Data object containing the meshgrid of x and t
    """

    x = data.x
    t = data.t

    X, T = np.meshgrid(x, t)
    Z = exact_sol(X, T)

    plt.figure(figsize=(8, 6))
    heatmap = plt.contourf(X, T, Z, cmap="hot", levels=500, vmin=0, vmax=1)
    plt.colorbar(heatmap, label="Temperature")
    plt.xlabel("Position (x)")
    plt.ylabel("Time (t)")
    plt.title("Heatmap of Analytical Solution")
    plt.show()
    

if __name__ == "__main__":
    data = RodDataGen(Nx=100, Nt=100, L=1, T=0.5)
    plot_heatmap(data)

