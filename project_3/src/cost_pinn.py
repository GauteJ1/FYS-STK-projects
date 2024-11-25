import numpy as np
from torch import nn

def dummy(x: np.ndarray, t: np.ndarray) -> float:
    return x**2 + 4 * t


def cost_PDE(x, t, nn_pred):

    pass


def cost_initial(x, t, nn_pred):

    t = np.zeros_like(t)

    nn_out = nn_pred(x, t)
    initial = np.sin(np.pi * x)

    return np.mean((nn_out - initial) ** 2)


def cost_boundary(x, t, nn_pred):

    L = 1

    x0 = np.zeros_like(x)
    x1 = np.ones_like(x) * L

    x0_out = nn_pred(x0, t)
    x1_out = nn_pred(x1, t)

    return np.mean(x0_out**2 + x1_out**2)


def total_cost(x, t, nn_pred):

    return (
        cost_PDE(x, t, nn_pred)
        + cost_initial(x, t, nn_pred)
        + cost_boundary(x, t, nn_pred)
    )


if __name__ == "__main__":
    x = np.linspace(0, 1, 20)
    t = np.linspace(0, 1, 20)

    x, t = np.meshgrid(x, t)

    cost_PDE(x, t, dummy)
