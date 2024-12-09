import torch


def cost_PDE(x, t, nn_pred):

    """
    Cost function for the PDE
    Cost = the mean squared residuals of the PDE

    Parameters
    ----------
    x : torch.Tensor
        Position
    t : torch.Tensor
        Time
    nn_pred : NeuralNetwork 
        Neural network model

    Returns
    -------
    cost : torch.Tensor
        Cost function for the PDE
    """

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

    """ 
    Cost function for the initial condition
    Cost = the mean squared residuals of the initial condition

    Parameters
    ----------
    x : torch.Tensor
        Position
    t : torch.Tensor
        Time
    nn_pred : NeuralNetwork
        Neural network model

    Returns
    -------
    cost : torch.Tensor
        Cost function for the initial condition
    """

    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    t = torch.zeros_like(t)

    nn_out = nn_pred(x, t)
    initial = torch.sin(torch.pi * x)

    cost = torch.mean((nn_out - initial) ** 2)

    return cost


def cost_boundary(x, t, nn_pred):

    """
    Cost function for the boundary condition
    Cost = the mean squared residuals of the boundary condition

    Parameters
    ----------
    x : torch.Tensor
        Position
    t : torch.Tensor
        Time
    nn_pred : NeuralNetwork
        Neural network model

    Returns
    -------
    cost : torch.Tensor
        Cost function for the boundary condition
    """

    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    L = 1

    x0 = torch.zeros_like(x)
    x1 = torch.ones_like(x) * L

    x0_out = nn_pred(x0, t)
    x1_out = nn_pred(x1, t)

    cost = torch.mean(x0_out**2 + x1_out**2)

    return cost


def total_cost(x, t, nn_pred):

    """
    Total cost function
    Cost = cost_PDE + cost_initial + cost_boundary

    Parameters
    ----------
    x : torch.Tensor
        Position
    t : torch.Tensor
        Time
    nn_pred : NeuralNetwork
        Neural network model

    Returns
    -------
    cost : torch.Tensor
        Total cost function
    """

    return (
        cost_PDE(x, t, nn_pred)
        + cost_initial(x, t, nn_pred)
        + cost_boundary(x, t, nn_pred)
    )
