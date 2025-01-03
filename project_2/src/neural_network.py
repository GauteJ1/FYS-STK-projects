import jax.numpy as jnp
import jax
import numpy as np
import json as json
from methods import *
from jax import grad


from learn_rate import Update_Beta


class NeuralNetwork:
    """
    A class for creating and training a neural network
    """

    def __init__(
        self,
        network_shape: list[int],
        activation_funcs: list[str],
        cost_func: str,
        type_of_network: str,
        update_strategy: str,
        manual_gradients: bool = False,
        train_test_split: bool = True,
        multiple_accuracy_funcs: bool = False,
    ) -> None:

        self.cost_func = cost_func

        self.activation_funcs = [globals()[func] for func in activation_funcs]
        self.activation_funcs_der = [
            globals()[func + "_der"] for func in activation_funcs
        ]

        self.network_shape = network_shape
        self.layers = self.create_layers(network_shape)
        self.type_of_network = type_of_network

        self.update_strategy = update_strategy
        self.update_beta = Update_Beta()  # import from learn_rate.py
        self.manual_gradients = manual_gradients
        self.train_test_split = train_test_split

        self.multiple_accuracy_funcs = multiple_accuracy_funcs

        self.epochs = 0

    def set_update_strategy(self) -> None:
        """
        Set the update strategy for the neural network

        Raises
        ------
        ValueError
            If the update strategy is not supported
        """

        if self.update_strategy == "Constant":
            self.update_beta.constant(self.learning_rate)
        elif self.update_strategy == "Momentum":
            self.update_beta.momentum_based(self.learning_rate, gamma=0.95)
        elif self.update_strategy == "Adagrad":
            self.update_beta.adagrad(self.learning_rate)
        elif self.update_strategy == "Adagrad_Momentum":
            self.update_beta.adagrad_momentum(self.learning_rate, gamma=0.95)
        elif self.update_strategy == "Adam":
            self.update_beta.adam(self.learning_rate)
        elif self.update_strategy == "RMSprop":
            self.update_beta.rmsprop(self.learning_rate)
        else:
            raise ValueError("Unsupported update strategy")

    def set_accuracy_function(self) -> None:
        """
        Set the accuracy function for the neural network

        Raises
        ------
        ValueError
            If the type of network is not supported
        """

        if self.type_of_network == "classification":
            self.accuracy_func = accuracy
        elif self.type_of_network == "continuous":
            self.accuracy_func = r_2
        else:
            raise ValueError("Invalid type of network")

        # for the classification case, we can also calculate recall, precision and f1-score
        if self.multiple_accuracy_funcs and self.type_of_network == "classification":
            self.accuracy_func2 = recall
            self.accuracy_func3 = precision
            self.accuracy_func4 = f1score

    def set_grads(self) -> None:
        """
        Set the gradient function for the neural network
        """

        if self.manual_gradients:
            self.gradient = self.manual_gradient
        else:
            self.gradient = self.jaxgrad_gradient

    def set_cost_function(self) -> None:
        """
        Set the cost function for the neural network

        Raises
        ------
        ValueError
            If the cost function is not supported
        """

        if self.cost_func == "MSE":
            self.cost_fun = mse
        elif self.cost_func == "CrossEntropy":
            self.cost_fun = cross_entropy
        elif self.cost_func == "BinaryCrossEntropy":
            self.cost_fun = binary_cross_entropy
        else:
            raise ValueError("Unsupported cost function")

        self.cost_fun_der = grad(self.cost_fun, 0)

    def create_layers(self, network_shape: list[int]) -> list:
        """
        Create the layers of the neural network. Initialize the weights and biases with random values from a normal distribution.

        Returns
        -------
        list
            A list of tuples containing the weights and biases for each layer
        """

        layers = []
        i_size = network_shape[0]

        np.random.seed(4155)

        for layer_output_size in network_shape[1:]:
            W = np.random.randn(layer_output_size, i_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size

        return layers

    def ravel_layers(self, layers: list) -> np.ndarray:
        """
        Ravel the layers of the neural network

        Returns
        -------
        np.ndarray
            A raveled array of the layers
        """

        theta = np.array([])

        for layer in layers:
            theta = np.append(theta, np.ravel(layer[0]))
            theta = np.append(theta, np.ravel(layer[1]))

        return theta.ravel()

    def reshape_layers(self, theta: np.ndarray) -> list:
        """
        Reshape the layers of the neural network

        Returns
        -------
        list
            A list of tuples containing the weights and biases for each layer
        """

        network_shape = self.network_shape

        layers = []
        i_size = network_shape[0]

        index = 0

        np.random.seed(4155)

        for layer_output_size in network_shape[1:]:
            W = np.reshape(
                theta[index : index + (layer_output_size * i_size)],
                (layer_output_size, i_size),
            )
            index += layer_output_size * i_size
            b = np.reshape(theta[index : index + layer_output_size], layer_output_size)
            index += layer_output_size
            layers.append((W, b))
            i_size = layer_output_size

        return layers

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict the output of the neural network

        Parameters
        ----------
        inputs : np.ndarray
            The input data

        Returns
        -------
        np.ndarray
            The predicted output
        """

        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = jnp.dot(a, W.T) + b
            a = activation_func(z)

        return a

    def feed_forward_saver(self, inputs: np.ndarray) -> tuple:
        """
        Feed forward the input data through the neural network and save the inputs and activations for each layer

        Parameters
        ----------
        inputs : np.ndarray
            The input data

        Returns
        -------
        tuple
            A tuple containing the inputs, zs and the final activation
        """

        layer_inputs = []
        zs = []  # Save the z values for later use
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = jnp.dot(a, W.T) + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a

    def train_network(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 100,
    ) -> None:
        """
        Train the neural network

        Parameters
        ----------
        inputs : np.ndarray
            The input data
        targets : np.ndarray
            The target data
        epochs : int
            The number of epochs to train the network
        learning_rate : float
            The learning rate
        batch_size : int, optional
            The batch size, by default 100
        """

        # splitting into training and testing data if train_test_split is True
        if self.train_test_split:

            train_inputs, test_inputs = inputs
            train_targets, test_targets = targets

        else:
            train_inputs = inputs
            train_targets = targets
            test_inputs = None
            test_targets = None

        # Initialize lists only if they are empty (for first training session)
        if not hasattr(self, "loss") or not self.loss:
            self.loss = []
        if not hasattr(self, "accuracy") or not self.accuracy:
            self.accuracy = []
        if not hasattr(self, "test_loss") or not self.test_loss:
            self.test_loss = []
        if not hasattr(self, "test_accuracy") or not self.test_accuracy:
            self.test_accuracy = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs += epochs

        # Set the accuracy function, update strategy (optimizer), gradients and cost function
        self.set_accuracy_function()
        self.set_update_strategy()
        self.set_grads()
        self.set_cost_function()

        num_samples = train_inputs.shape[0]

        # ensuring new seed for each epoch, but reproducibility between runs
        seed_numbers = [i for i in range(epochs)]
        i = 0

        for epoch in range(epochs):

            np.random.seed(seed_numbers[i])
            i += 1

            # Shuffle the data and create the batch
            if batch_size != 0:
                random_idx = np.linspace(0, num_samples - 1, num_samples, dtype=int)
                np.random.shuffle(random_idx)
                random_idx = random_idx[:batch_size]

                batch_inputs = train_inputs[random_idx]
                batch_targets = train_targets[random_idx]
            else:
                batch_inputs = train_inputs
                batch_targets = train_targets

            # Calculate the gradients
            layers_grad = self.gradient(batch_inputs, batch_targets)

            theta = self.ravel_layers(self.layers)
            theta_grad = self.ravel_layers(layers_grad)
            theta_grad = jnp.clip(theta_grad, -1e12, 1e12)

            theta_updated = self.update_beta(theta, theta_grad)

            self.layers = self.reshape_layers(theta_updated)

            # predictions, loss and accuracy
            train_predictions = self.predict(train_inputs)
            self.loss.append(self.cost_fun(train_predictions, train_targets))
            self.accuracy.append(self.accuracy_func(train_predictions, train_targets))

            # calculate test loss and accuracy if test data is provided
            if test_inputs is not None:
                test_predictions = self.predict(test_inputs)
                self.test_loss.append(self.cost_fun(test_predictions, test_targets))

                if (
                    self.multiple_accuracy_funcs
                    and self.type_of_network == "classification"
                ):
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

    def manual_gradient(self, inputs: np.ndarray, target: np.ndarray) -> list[float]:
        """
        Calculate the gradients manually

        Parameters
        ----------
        inputs : np.ndarray
            The input data
        target : np.ndarray
            The target data

        Returns
        -------
        list
            A list of tuples containing the gradients for each layer
        """

        layer_inputs, zs, predict = self.feed_forward_saver(inputs)

        layer_grads = [() for layer in self.layers]

        # We loop over the self.layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = (
                layer_inputs[i],
                zs[i],
                self.activation_funcs_der[i],
            )

            if i == len(self.layers) - 1:
                dC_da = self.cost_fun_der(predict, target)
            else:
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W

            dC_dz = dC_da * activation_der(z)
            dC_dW = jnp.dot(dC_dz.T, layer_input)
            dC_db = jnp.mean(dC_dz, axis=0)

            dC_db *= inputs.shape[0]

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def jaxgrad_gradient(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Calculate the gradients using JAX

        Parameters
        ----------
        inputs : np.ndarray
            The input data
        targets : np.ndarray
            The target data

        Returns
        -------
        list
            A list of tuples containing the gradients for each layer
        """

        def jax_grad_predict(layers: list[float], inputs: np.ndarray) -> np.ndarray:
            """

            Predict the output of the neural network using JAX

            Parameters
            ----------
            layers : list[float]
                The layers of the neural network
            inputs : np.ndarray
                The input data

            Returns
            -------
            np.ndarray
                The predicted output
            """

            a = inputs
            for (W, b), activation_func in zip(layers, self.activation_funcs):
                z = jnp.dot(a, W.T) + b
                a = activation_func(z)
            return a

        def jax_grad_cost(
            layers: list[float], inputs: np.ndarray, targets: np.ndarray
        ) -> np.ndarray:
            """
            Calculate the cost function using JAX

            Parameters
            ----------
            layers : list[float]
                The layers of the neural network
            inputs : np.ndarray
                The input data
            targets : np.ndarray
                The target data

            Returns
            -------
            np.ndarray
                The cost function
            """

            predictions = jax_grad_predict(layers, inputs)
            return self.cost_fun(predictions, targets)

        gradients = grad(jax_grad_cost, argnums=0)(self.layers, inputs, targets)

        return gradients

    def save_network(self, file_name: str) -> None:
        """
        Save the neural network to a JSON file

        Parameters
        ----------
        file_name : str
            The name of the file
        """

        network_info = {
            "network_shape": [layer[0].shape[1] for layer in self.layers],
            "activation_funcs": [func.__name__ for func in self.activation_funcs],
            "cost_func": self.cost_func,
            "type_of_network": self.type_of_network,
            "update_strategy": self.update_strategy,
            "manual_gradients": self.manual_gradients,
            "layers": [
                (np.array(W).tolist(), np.array(b).tolist()) for W, b in self.layers
            ],
            "loss_history": [float(loss) for loss in self.loss],
            "accuracy_history": [float(acc) for acc in self.accuracy],
            "loss_history_test": [float(loss) for loss in self.test_loss],
            "accuracy_history_test": [float(acc) for acc in self.test_accuracy],
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

        with open(file_name, "w") as file:
            json.dump(network_info, file, indent=4)

    @classmethod
    def load_network(cls, file_name: str) -> "NeuralNetwork":
        """
        Load a neural network from a JSON file

        Parameters
        ----------
        file_name : str
            The name of the file

        Returns
        -------
        NeuralNetwork
            A new instance of the NeuralNetwork class
        """

        with open(file_name, "r") as file:
            network_info = json.load(file)

        model = cls(
            network_info["network_shape"],
            network_info["activation_funcs"],
            network_info["cost_func"],
            network_info["type_of_network"],
            network_info["update_strategy"],
            network_info["manual_gradients"],
        )

        model.layers = [(np.array(W), np.array(b)) for W, b in network_info["layers"]]
        model.loss = network_info.get("loss_history", [])
        model.accuracy = network_info.get("accuracy_history", [])
        model.test_loss = network_info.get("loss_history_test", [])
        model.test_accuracy = network_info.get("accuracy_history_test", [])
        model.learning_rate = network_info.get("learning_rate")
        model.batch_size = network_info.get("batch_size")
        model.epochs = network_info.get("epochs")

        return model
