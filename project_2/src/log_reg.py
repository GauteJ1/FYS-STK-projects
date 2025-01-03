import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from methods import sigmoid, recall, accuracy, precision, f1score
from learn_rate import Update_Beta


class LogReg:
    """
    Class for logistic regression
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        l2_reg_param: float,
        update_strategy: str,
        train_test_split: bool = True,
        multiple_accuracy_funcs: bool = True,
    ):

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.activation = sigmoid
        self.accuracy_func = accuracy

        self.l2_reg_param = l2_reg_param

        self.update_strategy = update_strategy
        self.update_beta = Update_Beta()

        self.train_test_split = train_test_split

        weights = np.random.rand(output_shape, input_shape)
        bias = np.random.rand(output_shape)
        self.model = (weights, bias)

        self.multiple_accuracy_funcs = multiple_accuracy_funcs
        if self.multiple_accuracy_funcs:
            self.accuracy_func2 = recall
            self.accuracy_func3 = precision
            self.accuracy_func4 = f1score

        self.eps = 1e-8  # Small constant to avoid division by zero errors due to machine precision

    def set_update_strategy(self, learning_rate):
        """
        Sets the update strategy for the gradient descent (i.e. the optimizer)

        Raises
        ------
        ValueError
            If the update strategy is not supported
        """

        if self.update_strategy == "Constant":
            self.update_beta.constant(learning_rate)
        elif self.update_strategy == "Momentum":
            self.update_beta.momentum_based(learning_rate, gamma=0.95)
        elif self.update_strategy == "Adagrad":
            self.update_beta.adagrad(learning_rate)
        elif self.update_strategy == "Adagrad_Momentum":
            self.update_beta.adagrad_momentum(learning_rate, gamma=0.95)
        elif self.update_strategy == "Adam":
            self.update_beta.adam(learning_rate)
        elif self.update_strategy == "RMSprop":
            self.update_beta.rmsprop(learning_rate)
        else:
            raise ValueError("Unsupported update strategy")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output of the model

        Returns
        -------
        y : np.ndarray
            The predicted output
        """

        weights, bias = self.model
        z = jnp.dot(x, weights.T) + bias
        y = self.activation(z)

        return y

    def cost(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates the cost function

        Returns
        -------
        cost : float
            The cost function
        """

        preds = np.clip(preds, self.eps, 1 - self.eps)

        cost = 0
        for p, y in zip(preds, targets):
            if y == 1:
                cost -= np.log(p)
            else:
                cost -= np.log(1 - p)

        weights = self.model[0]
        cost += self.l2_reg_param * np.mean(weights**2)

        return cost

    def gradient(self, inputs, targets):
        """
        Calculates the gradient of the cost function

        Returns
        -------
        gradients : np.ndarray
            The gradients of the cost function
        """

        def jax_cost(model, inputs, targets):
            """
            calculates the cost function for the model using jax

            Parameters
            ----------
            model : tuple
                The model weights and bias
            inputs : np.ndarray
                The input data
            targets : np.ndarray
                The target data

            Returns
            -------
            cost : float
            """

            weights, bias = model
            z = jnp.dot(inputs, weights.T) + bias
            y = self.activation(z)

            preds = jnp.clip(y, self.eps, 1 - self.eps)  # to avoid nan

            cost = 0
            for p, y in zip(preds, targets):
                if y == 1:
                    cost -= jnp.log(p)
                else:
                    cost -= jnp.log(1 - p)

            cost += self.l2_reg_param * jnp.mean(weights**2)

            return cost[0]

        # Use jax to calculate gradients
        gradients = jax.grad(jax_cost, 0)(self.model, inputs, targets)

        # Avoid exploding gradients
        # gradients = jnp.clip(gradients, -1e12, 1e12)

        return gradients

    def ravel_layers(self, model):
        """
        Ravel the layers of the model

        Parameters
        ----------
        model : tuple
            The model weights and bias

        Returns
        -------
        theta : np.ndarray
            The raveled model
        """

        theta = np.array([])
        theta = np.append(theta, np.ravel(model[0]))
        theta = np.append(theta, np.ravel(model[1]))

        return theta.ravel()

    def reshape_layers(self, theta) -> list:
        """
        Reshape the raveled model

        Parameters
        ----------
        theta : np.ndarray
            The raveled model

        Returns
        -------
        model : list
            The reshaped model
        """

        input_shape = self.input_shape
        output_shape = self.output_shape
        weight_size = input_shape * output_shape

        weights = np.reshape(
            theta[:weight_size],
            (output_shape, input_shape),
        )
        bias = np.reshape(theta[weight_size:], output_shape)

        return (weights, bias)

    def train_model(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 100,
    ) -> None:
        """
        Trains the model

        Parameters
        ----------
        inputs : np.ndarray
            The input data
        targets : np.ndarray
            The target data
        epochs : int
            The number of epochs
        learning_rate : float
            The learning rate
        batch_size : int, optional
            The batch size, by default 100
        """

        if self.train_test_split:

            train_inputs, test_inputs = inputs
            train_targets, test_targets = targets

        else:
            train_inputs = inputs
            train_targets = targets
            test_inputs = None
            test_targets = None

        self.loss = []
        self.accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        self.set_update_strategy(learning_rate)

        num_samples = train_inputs.shape[0]

        seed_numbers = [i for i in range(epochs)]
        i = 0

        for _ in range(epochs):

            np.random.seed(seed_numbers[i])
            i += 1

            if batch_size != 0:
                random_idx = np.linspace(0, num_samples - 1, num_samples, dtype=int)
                np.random.shuffle(random_idx)
                random_idx = random_idx[:batch_size]

                batch_inputs = train_inputs[random_idx]
                batch_targets = train_targets[random_idx]
            else:
                batch_inputs = train_inputs
                batch_targets = train_targets

            layers_grad = self.gradient(batch_inputs, batch_targets)

            theta = self.ravel_layers(self.model)
            theta_grad = self.ravel_layers(layers_grad)

            theta_updated = self.update_beta(theta, theta_grad)

            self.model = self.reshape_layers(theta_updated)

            train_predictions = self.predict(train_inputs)
            self.loss.append(self.cost(train_predictions, train_targets))
            self.accuracy.append(self.accuracy_func(train_predictions, train_targets))

            if test_inputs is not None:
                test_predictions = self.predict(test_inputs)
                self.test_loss.append(self.cost(test_predictions, test_targets))

                if self.multiple_accuracy_funcs:
                    accuracy1 = self.accuracy_func(test_predictions, test_targets)
                    accuracy2 = self.accuracy_func2(test_predictions, test_targets)
                    accuracy3 = self.accuracy_func3(test_predictions, test_targets)
                    accuracy4 = self.accuracy_func4(test_predictions, test_targets)

                    self.test_accuracy.append(
                        (accuracy1, accuracy2, accuracy3, accuracy4)
                    )

                else:
                    self.test_accuracy.append(
                        self.accuracy_func(test_predictions, test_targets)
                    )
