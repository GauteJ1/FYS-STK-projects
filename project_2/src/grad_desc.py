import numpy as np
import jax
import jax.numpy as jnp
from data_gen import DataGen, Poly1D2Deg
from learn_rate import Update_Beta

class Model:

    """
    Class for the model to be used in the gradient descent
    """

    def __init__(self, data: DataGen, model_type: str) -> None:
        
        """
        Initializes the class and sets the data and model type
        """

        self.x = data.x
        self.y = data.y
        self.n = data.data_points
        self.model_type = model_type

        self.a = data.a
        self.b = data.b
        self.c = data.c

        self.custom_init = False

    def makeX(self, deg: int):

        """
        Creates the design matrix X
        """

        # Add dimension 2 option to this if we want to use Franke/Terrain
        # Design matrix including the intercept
        # No scaling of data of and all data used for training (for now)
        X = np.zeros((self.n, deg))
        for d in range(deg):
            X[:, d] = self.x[:, 0] ** d
        return X

    def set_update(self, tpe, eta, gamma):

        """
        Sets the update strategy (optimizer) for the gradient descent
        """
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
            
        """
        Calculates the analytical gradient for the model
        """

        if self.model_type == "OLS":
            return lambda beta: 2.0 / self.n * X.T @ (X @ beta - y)
        elif self.model_type == "Ridge":
            return lambda beta: 2.0 / self.n * X.T @ (X @ beta - y) + 2 * Lambda * beta

    def loss(self):
        """
        calculates the loss function for the model

        Returns
        -------
        loss : function
            The loss function
        """
        if self.model_type == "OLS":
            loss = lambda X, y, beta, Lambda: jnp.mean((X @ beta - y) ** 2)
        elif self.model_type == "Ridge":
            loss = lambda X, y, beta, Lambda: jnp.mean((X @ beta - y) ** 2) + jnp.sum(
                Lambda * beta**2
            )
        return loss

    def gradient(self, X, y, Lambda=0.1):
        """
        Calculates the gradient of the loss function

        Parameters
        ----------
        X : np.ndarray
            The design matrix
        y : np.ndarray
            The target values
        Lambda : float, optional
            The regularization parameter, by default 0.1

        Returns
        -------
        function
            The gradient of the loss function
        """

        f = lambda beta: self.loss()(X, y, beta, Lambda)

        return jax.grad(f)

    def set_custom_initial_val(self, init_beta: np.ndarray) -> None:

        """
        Sets the initial value of beta to a custom value

        Parameters
        ----------
        init_beta : np.ndarray
            The initial value of beta
        """

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
        
        """
        Performs the gradient descent

        Parameters
        ----------
        tpe : str, optional
            The type of optimizer, by default "Constant"
        eta : float, optional
            The learning rate, by default 0.05
        gamma : float, optional
            The momentum parameter, by default 0.05
        epochs : int, optional
            The number of epochs, by default 10000
        batch_size : int, optional
            The batch size, by default 0
        """

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

                gradients = self.gradient(X_batch, y_batch)

                self.beta = update(self.beta, gradients(self.beta), iter)

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

    beta = np.zeros((3, 1))
    X = model.makeX(3)
    y = model.y

    # print(model.analytical_gradient(X, y, beta))
    # print(model.gradient(X, y, beta))
    assert np.allclose(
        model.analytical_gradient(X, y)(beta), model.gradient(X, y)(beta), atol=1e-5
    )

    data = Poly1D2Deg(101)
    model = Model(data, model_type="Ridge")

    beta = np.zeros((3, 1))
    X = model.makeX(3)
    y = model.y

    # print(model.analytical_gradient(X, y, beta))
    # print(model.gradient(X, y, beta))
    assert np.allclose(
        model.analytical_gradient(X, y)(beta), model.gradient(X, y)(beta), atol=1e-5
    )
