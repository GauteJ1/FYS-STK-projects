import torch
from torch.autograd import grad


def cost_PDE(x, t, nn_pred):

    nn_out = nn_pred(x, t)

    x_deriv = grad(nn_out, x, grad_outputs=torch.ones_like(nn_out))
    xx_deriv = grad(x_deriv, x)

    t_deriv = grad(nn_out, t)

    return torch.mean((xx_deriv - t_deriv) ** 2)


def cost_initial(x, t, nn_pred):

    t = torch.zeros_like(t)

    nn_out = nn_pred(x, t)
    initial = torch.sin(torch.pi * x)

    return torch.mean((nn_out - initial) ** 2).item()


def cost_boundary(x, t, nn_pred):

    L = 1

    x0 = torch.zeros_like(x)
    x1 = torch.ones_like(x) * L

    x0_out = nn_pred(x0, t)
    x1_out = nn_pred(x1, t)

    return torch.mean(x0_out**2 + x1_out**2).item()


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