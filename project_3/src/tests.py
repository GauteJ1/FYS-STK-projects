from finite_difference import ForwardEuler
import numpy as np
from analytic import exact_sol

def test_mse_fd():
    """
    Testing that the output of the finite difference method is suffenciently close to the exact solution
    """

    init = lambda x: np.sin(np.pi * x)
    model = ForwardEuler(10, 0.001)
    model.set_init(init)
    data = model(0.5)

    mse = model.total_mse(data, exact_sol) 

    assert mse < 1e-6, "The mean squared error is too high"

def test_values_heatmap(): 
    """
    Using np.allclose to test that the resulting values from the finite difference method is close to the exact solution
    """

    N = 100
    init = lambda x: np.sin(np.pi * x)
    model = ForwardEuler(N)
    model.set_init(init)
    data = model(0.5)

    x = np.linspace(0, 1, N+1)
    t = data["time_steps"]
    X, T = np.meshgrid(x, t)
    exact_solution = exact_sol(X, T)

    Z = data["values"]

    assert np.allclose(Z, exact_solution, atol=1e-4), "The values are not close to the exact solution"


def test_high_T(): 
    """
    Testing that the solution of the finite difference methos is close to zero at high T
    """

    N = 100
    T = 10
    init = lambda x: np.sin(np.pi * x)
    model = ForwardEuler(N)
    model.set_init(init)
    data = model(T)

    Z = data["values"]
    
    assert np.allclose(Z[-1], 0, atol=1e-4), "The values are not close to zero at high T"