import numpy as np
import jax

from methods import sigmoid, recall
from neural_network import NeuralNetwork
from learn_rate import Update_Beta

class LogReg:
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        l2_reg_param: float,
        update_strategy: str,
        train_test_split: bool = True,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.activation = sigmoid
        self.accuracy_func = recall
        self.l2_reg_param = l2_reg_param

        self.update_strategy = update_strategy
        self.update_beta = Update_Beta()

        self.train_test_split = train_test_split

        weights = np.random.rand(output_shape, input_shape)
        bias = np.random.rand(output_shape)
        self.model = (weights, bias)

        self.eps = 1e-10  # Small constant to avoid division by zero errors due to machine precision

    def set_update_strategy(self, learning_rate):
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
        weights, bias = self.model
        z = weights @ x + bias
        y = self.activation(z)

        return y

    def cost(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        preds = self.predict(inputs)

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

        def jax_cost(model, inputs, targets):
            weights, bias = model
            z = weights @ inputs + bias
            y = self.activation(z)

            preds = np.clip(y, self.eps, 1 - self.eps)

            cost = 0
            for p, y in zip(preds, targets):
                if y == 1:
                    cost -= np.log(p)
                else:
                    cost -= np.log(1 - p)

            cost += self.l2_reg_param * np.mean(weights**2)

            return cost

        # Use jax to calculate gradients
        gradients = jax.grad(jax_cost, 0)(self.model, inputs, targets)

        # Avoid exploding gradients
        gradients = np.clip(gradients, -1e12, 1e12)

        return gradients


    def ravel_layers(self, model):
        theta = np.array([])
        theta = np.append(theta, np.ravel(model[0]))
        theta = np.append(theta, np.ravel(model[1]))

        return theta.ravel()

    def reshape_layers(self, theta) -> list:
        input_shape = self.input_shape
        output_shape = self.output_shape
        weight_size = input_shape*output_shape

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

            theta = self.ravel_layers(self.layers)
            theta_grad = self.ravel_layers(layers_grad)

            theta_updated = self.update_beta(theta, theta_grad)

            self.layers = self.reshape_layers(theta_updated)

            train_predictions = self.predict(train_inputs)
            self.loss.append(self.cost(train_predictions, train_targets))
            self.accuracy.append(self.accuracy_func(train_predictions, train_targets))

            if test_inputs is not None:
                test_predictions = self.predict(test_inputs)
                self.test_loss.append(self.cost(test_predictions, test_targets))
                self.test_accuracy.append(
                    self.accuracy_func(test_predictions, test_targets)
                )