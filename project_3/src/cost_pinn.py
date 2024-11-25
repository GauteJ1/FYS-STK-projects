import torch
from torch.autograd import grad
from neural_network import NeuralNetwork


def cost_PDE(x, t, nn_pred):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    nn_out = nn_pred(x, t)

    dx = torch.autograd.grad(nn_out, x, torch.ones_like(nn_out), create_graph=True)[0]
    dxx = torch.autograd.grad(dx, x, torch.ones_like(dx), create_graph=True)[0]
    dt = torch.autograd.grad(nn_out, t, torch.ones_like(nn_out), create_graph=True)[0]

    residual = dxx - dt
    cost = torch.mean(residual**2)

    return cost


def cost_initial(x, t, nn_pred):

    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    t = torch.zeros_like(t)

    nn_out = nn_pred(x, t)
    initial = torch.sin(torch.pi * x)

    return torch.mean((nn_out - initial) ** 2)


def cost_boundary(x, t, nn_pred):

    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    L = 1

    x0 = torch.zeros_like(x)
    x1 = torch.ones_like(x) * L

    x0_out = nn_pred(x0, t)
    x1_out = nn_pred(x1, t)

    return torch.mean(x0_out**2 + x1_out**2)


def total_cost(x, t, nn_pred):

    return (
        cost_PDE(x, t, nn_pred)
        + cost_initial(x, t, nn_pred)
        + cost_boundary(x, t, nn_pred)
    )


if __name__ == "__main__":
    def dummy(x, t):
        return x**2 * 4*t
    
    x = torch.linspace(0, 9, steps= 10)
    t = torch.linspace(0, 9, steps= 10)
    
    x, t = torch.meshgrid(x, t)

    print(cost_PDE(x, t, dummy))