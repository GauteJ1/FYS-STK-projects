import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from data_gen import DataGen, Poly1D2Deg
from learn_rate import Update_Beta

# We will have to revisit structuring, this i just an initial idea based on the first couple of tasks in the weekly assignment
# Possibly, the main class should not be OLS/Ridge, and they would rather be possible function options for the gradient or something

# Have not tuned the learning rate yet


class Model:

    def __init__(self, data: DataGen, model_type: str) -> None:
        self.x = data.x
        self.y = data.y
        self.n = data.data_points
        self.model_type = model_type

        self.a = data.a
        self.b = data.b
        self.c = data.c

        self.custom_init = False

    def makeX(self, deg: int):
        # Add dimension 2 option to this if we want to use Franke/Terrain

        # Design matrix including the intercept
        # No scaling of data of and all data used for training (for now)
        X = np.zeros((self.n, deg))
        for d in range(deg):
            X[:, d] = self.x[:, 0] ** d
        return X

    def set_update(self, tpe, eta, gamma):
        update = Update_Beta()
        if tpe == "Constant":
            update.constant(eta)
        elif tpe == "Momentum":
            update.momentum_based(eta, gamma)
        elif tpe == "Adagrad":
            update.adagrad(eta)
        elif tpe == "Adagrad_Momentum":
            update.adagrad_momentum(eta, gamma)
        elif tpe == "Adam":
            update.adam(eta=eta)
        elif tpe == "RMSprop":
            update.rmsprop(eta=eta)

        return update

    def analytical_gradient(self, X, y, Lambda=0.1):
        if self.model_type == "OLS":
            return lambda beta: 2.0 / self.n * X.T @ (X @ beta - y)
        elif self.model_type == "Ridge":
            return lambda beta: 2.0 / self.n * X.T @ (X @ beta - y) + 2 * Lambda * beta

    def gradient(self, X, y, Lambda=0.1):
        if self.model_type == "OLS":
            f = lambda beta: np.mean((X @ beta - y) ** 2)
        elif self.model_type == "Ridge":
            f = lambda beta: np.mean( (X @ beta - y) ** 2) + np.sum(Lambda * beta**2)
        return jax.grad(f)

    def set_custom_initial_val(self, init_beta: np.ndarray) -> None:
        self.custom_init = True
        self.init_beta = init_beta

    def gradient_descent(
        self,
        tpe: str = "Constant",
        eta: float = 0.05,
        gamma: float = 0.05,
        epochs: int = 10000,
        batch_size: int = 0,
    ):

        if batch_size == 0:
            batch_size = self.n

        self.epoch_list = []
        self.MSE_list = []

        # This should be counfigurable from the function call (or elsewhere)
        if self.custom_init:
            self.beta = self.init_beta
        else:
            self.beta = np.random.randn(3, 1)
        X = self.makeX(3)

        gradients = self.gradient(X, self.y)

        # Learning rate and number of iterations
        update = self.set_update(tpe, eta, gamma)

        # Iterations (separate jit-compilable function?)
        iter = 0
        tolerance = 1e-6

        # pbar = tqdm(total = epochs * (self.n // batch_size))
        for epoch in range(epochs):
            iter += 1
            prev_beta_ridge = self.beta.copy()

            indices = np.random.permutation(
                self.n
            )  # must check if it should be random or not, have not checked Morten's notes
            X_shuffled = X[indices]
            y_shuffled = self.y[indices]

            for i in range(0, self.n, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                self.beta = update(
                    self.beta, gradients(self.beta), iter
                )

                # pbar.update(1)

            preds = X @ self.beta
            error = np.mean((self.y - preds) ** 2)
            self.MSE_list.append(error)
            self.epoch_list.append(epoch)

            if np.allclose(prev_beta_ridge, self.beta, tolerance):
                # print(f'Converged after {epoch} epochs for {update.rate_type}')
                break

        # pbar.close()


if __name__ == "__main__":
    data = Poly1D2Deg(101)
    model = Model(data, model_type="OLS")

    beta = np.zeros((3,1))
    X = model.makeX(3)
    y = model.y

    # print(model.analytical_gradient(X, y, beta))
    # print(model.gradient(X, y, beta))
    assert np.allclose(model.analytical_gradient(X, y)(beta), model.gradient(X, y)(beta), atol=1e-5)

    data = Poly1D2Deg(101)
    model = Model(data, model_type="Ridge")

    beta = np.zeros((3,1))
    X = model.makeX(3)
    y = model.y

    # print(model.analytical_gradient(X, y, beta))
    # print(model.gradient(X, y, beta))
    assert np.allclose(model.analytical_gradient(X, y)(beta), model.gradient(X, y)(beta), atol=1e-5)