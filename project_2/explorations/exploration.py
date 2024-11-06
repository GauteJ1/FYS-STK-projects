import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm
import sys
import os

# ensuring the src folder is in the python path for the imports
sys.path.append(os.path.abspath("../src"))

from neural_network import NeuralNetwork
from methods import *
from data_gen import FrankeDataGen, CancerData

np.random.seed(4155)  # FYS-STK4155

# plot settings
sns.set_theme()
plt.style.use("../plot_settings.mplstyle")


class Exploration:
    """
    Class for exploring different hyperparameters for neural networks.
    """

    def __init__(
        self,
        type_model: str,
        optimizers: list[str],
        layer_sizes: list[int],
        num_hidden_layers: list[int],
        learning_rates: list[float],
        batch_sizes: list[float],
        activation_functions: list[str],
    ) -> None:
        """
        Initializes the Exploration class for neural network experimentation.

        Parameters/Attributes:
        ----------
        type_model : str
            Type of model to train, either 'continuous' or 'classification'.
        optimizers : list[str]
            List of optimizers to test.
        layer_sizes : list[int]
            List of sizes for hidden layers.
        num_hidden_layers : list[int]
            List of number of hidden layers to test.
        learning_rates : list[float]
            List of learning rates to test.
        batch_sizes : list[float]
            List of batch sizes to test.
        activation_functions : list[str]
            List of activation functions to test.
        """

        self.type_model = type_model
        self.optimizers = optimizers
        self.layer_sizes = layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.activation_functions = activation_functions

        self.accuracy_measure = (
            r"$R^2$" if self.type_model == "continuous" else "Accuracy"
        )

    def generate_data(self) -> None:
        """
        Generates and prepares the dataset based on the specified model type (either 'continuous' or 'classification').

        For 'continuous' models, it uses the Franke function data:
            - Generates 2D input features from the FrankeDataGen class.
            - Sets the cost function to Mean Squared Error (MSE).
            - Input size is set to 2, and output size is set to 1.

        For 'classification' models, it uses the breast cancer dataset:
            - Loads data from the CancerData class.
            - Sets the cost function to Binary Cross Entropy.
            - Input size is set to 30, and output size is set to 1.

        Splits data into training and testing sets (70/30 split) with a fixed random seed for reproducibility.

        Raises:
            ValueError: If `type_model` is not 'continuous' or 'classification'.
        """

        if self.type_model == "continuous":
            data = FrankeDataGen(noise=True)
            x_data = jnp.column_stack((data.x.flatten(), data.y.flatten()))
            y_data = data.z.ravel().reshape(-1, 1)

            self.input_size = 2
            self.output_size = 1
            self.cost_function = "MSE"
            self.cost_function_name = "Mean squared error"

        elif self.type_model == "classification":
            data = CancerData()
            x_data = np.array(data.x)
            y_data = np.array(data.y).reshape(-1, 1)

            self.input_size = 30
            self.output_size = 1
            self.cost_function = "BinaryCrossEntropy"
            self.cost_function_name = "Binary cross entropy"

        else:
            raise ValueError(
                "Type model must be either 'continuous' or 'classification'"
            )

        np.random.seed(4155)

        X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3, random_state=4155
        )
        inputs = X_train, X_test
        targets = y_train, y_test

        self.inputs = inputs
        self.targets = targets

    def find_maximal_accuracy(self, configuration: list, test_accuracy: list) -> None:
        """
        Finds the configuration with the highest accuracy from test results.

        Parameters:
            configurations: Either a tuple of (learning_rates, batch_sizes) or a list of configurations.
            test_accuracy: List of test accuracies for each configuration.

        Returns:
            Best configuration with the maximal final accuracy.
        """

        if isinstance(configuration, tuple):
            lr_values, batch_sizes = configuration
            max_accuracy = -np.inf
            best_config = (None, None, max_accuracy)

            for i, acc in enumerate(test_accuracy):
                if acc[-1] > max_accuracy:
                    max_accuracy = acc[-1]
                    best_config = (
                        lr_values[i // len(batch_sizes)],
                        batch_sizes[i % len(batch_sizes)],
                        float(max_accuracy),
                    )

            return best_config

        else:
            max_accuracy = -np.inf
            for i, acc in enumerate(test_accuracy):
                if acc[-1] > max_accuracy:
                    max_accuracy = acc[-1]
                    best = configuration[i]

        return best

    def plot(
        self, measure: list, configurations: list, title: str, y_label: str
    ) -> None:
        """
        Plots the given measure (accuracy or loss) for different configurations over epochs, starting from epoch 50.

        Parameters:
            measure: List of values (accuracy or loss) over epochs for each configuration.
            configurations: List of configuration names or labels for the plot legend.
            title: Title for the plot, indicating the measure being plotted.
            y_label: Label for the y-axis of the plot.

        Saves the plot as a PDF file in the figures folder and displays it.
        """

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        for i, mes in enumerate(measure):

            x_axis = np.arange(50, len(mes))
            ax.plot(x_axis, mes[50:], label=configurations[i])

        ax.set_xlabel("Epochs")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()

        plt.savefig(f"../figures/{title}_{self.type_model}.pdf")

        plt.show()

    def find_ok_optimizer(self) -> None:
        """
        Performs an initial search to identify a good enough optimizer for the given model type.

        Sets up intermediary hyperparameters (learning rate, epochs, layer sizes) and trains the model
        with each optimizer in self.optimizers. Selects the optimizer that yields the highest accuracy.

        Attributes Updated:
            best_optimizer: Optimizer with the highest accuracy in this initial search.
        """

        self.intermediary_lr = 0.005
        self.intermediary_epochs = 300
        self.intermediary_hidden_layers = [4, 8]

        if self.type_model == "classification":
            self.intermediary_activation_funcs = ["ReLU", "ReLU", "sigmoid"]
            self.intermediary_batch_size = 200
        else:
            self.intermediary_activation_funcs = ["ReLU", "ReLU", "identity"]
            self.intermediary_batch_size = 1000

        network_shape = (
            [self.input_size] + self.intermediary_hidden_layers + [self.output_size]
        )
        test_loss = []
        test_accuracy = []

        for optim in self.optimizers:
            model = NeuralNetwork(
                network_shape,
                self.intermediary_activation_funcs,
                self.cost_function,
                self.type_model,
                optim,
            )
            model.train_network(
                self.inputs,
                self.targets,
                self.intermediary_epochs,
                self.intermediary_lr,
                self.intermediary_batch_size,
            )
            test_loss.append(model.test_loss)
            test_accuracy.append(model.test_accuracy)

        self.best_optimizer = self.find_maximal_accuracy(self.optimizers, test_accuracy)
        self.top_3_optimizers = [
            self.optimizers[i]
                for i in sorted(
                    range(len(test_accuracy)), key=lambda x: test_accuracy[x][-1], reverse=True
                )[:3]
        ]
        print(f"Top 3 optimizers: {self.top_3_optimizers}")
        # test_loss_top_3 = [test_loss[i] for i in sorted(range(len(test_accuracy)), key=lambda x: test_accuracy[x], reverse=True)[:3]]
        # test_accuracy_top_3 = [test_accuracy[i] for i in sorted(range(len(test_accuracy)), key=lambda x: test_accuracy[x], reverse=True)[:3]]

        if self.print_info:
            print(
                rf"Best optimizer (gives the highest {self.accuracy_measure}): {self.best_optimizer}"
            )
            self.plot(
                test_loss,
                self.optimizers,
                f"{self.cost_function_name} for different optimizers",
                self.cost_function_name,
            )
            self.plot(
                test_accuracy,
                self.optimizers,
                f"{self.accuracy_measure} for different optimizers",
                self.accuracy_measure,
            )

    def find_structure(self) -> None:
        """
        Finds the best network structure by testing different combinations of hidden layer sizes.

        For each combination of hidden layer sizes, trains the model and records the test loss and accuracy.
        Selects the structure with the highest accuracy.

        Attributes Updated:
            best_structure: Optimal network structure (layer sizes) based on test accuracy.
            num_hidden_layers: Number of hidden layers in the best structure.
        """

        test_loss = []
        test_accuracy = []
        permutations = []

        for layers in tqdm(self.num_hidden_layers):
            for sizes in product(self.layer_sizes, repeat=layers):
                permutations.append(sizes)

                network_shape = [self.input_size] + list(sizes) + [self.output_size]

                model = NeuralNetwork(
                    network_shape,
                    self.intermediary_activation_funcs,
                    self.cost_function,
                    "continuous",
                    self.best_optimizer,
                )
                model.train_network(
                    self.inputs,
                    self.targets,
                    epochs=self.intermediary_epochs,
                    learning_rate=self.intermediary_lr,
                    batch_size=self.intermediary_batch_size,
                )

                test_loss.append(model.test_loss)
                test_accuracy.append(model.test_accuracy)

        best = self.find_maximal_accuracy(permutations, test_accuracy)
        best_list = list(best)

        self.best_structure = [self.input_size] + best_list + [self.output_size]
        self.num_hidden_layers = len(
            best_list
        )  # update number of hidden layers to the best

        if self.print_info:
            print(
                rf"Best structure (gives the highest {self.accuracy_measure}): {self.best_structure}"
            )
            self.plot(
                test_loss,
                permutations,
                f"{self.cost_function_name} for different structures",
                self.cost_function_name,
            )
            self.plot(
                test_accuracy,
                permutations,
                f"{self.accuracy_measure} for different structures",
                self.accuracy_measure,
            )

    def find_activation_functions(self) -> None:
        """
        Identifies the best activation functions for each hidden layer by testing different combinations.

        For each combination of activation functions, trains the model and records test accuracy.
        Selects the activation functions that yield the highest accuracy.

        Attributes Updated:
            best_activation_functions: List of optimal activation functions for each layer.
        """

        combinations_of_activations = list(
            product(self.activation_functions, repeat=self.num_hidden_layers)
        )
        final_activation = (
            "sigmoid" if self.type_model == "classification" else "identity"
        )

        combinations_of_activations = [
            list(comb) + [final_activation] for comb in combinations_of_activations
        ]

        test_loss = []
        test_accuracy = []
        for combination in combinations_of_activations:
            model = NeuralNetwork(
                self.best_structure,
                combination,
                self.cost_function,
                "continuous",
                self.best_optimizer,
            )
            model.train_network(
                self.inputs,
                self.targets,
                epochs=self.intermediary_epochs,
                learning_rate=self.intermediary_lr,
                batch_size=self.intermediary_batch_size,
            )
            test_loss.append(model.test_loss)
            test_accuracy.append(model.test_accuracy)

        best = self.find_maximal_accuracy(combinations_of_activations, test_accuracy)
        self.best_activation_functions = list(best)

        if self.print_info:
            print(
                rf"Best activation functions (gives the highest {self.accuracy_measure}): {self.best_activation_functions}"
            )
            self.plot(
                test_loss,
                combinations_of_activations,
                f"{self.cost_function_name} for different activation functions",
                self.cost_function_name,
            )
            self.plot(
                test_accuracy,
                combinations_of_activations,
                f"{self.accuracy_measure} for different activation functions",
                self.accuracy_measure,
            )

    def plot_grid(self, measure: dict, title: str):
        """
        Plots a heatmap of the given measure (accuracy or loss) for each optimizer, across learning rates and batch sizes.

        Parameters:
            measure: Dictionary with optimizers as keys and 3D matrices of values (accuracy or loss) as values.
            title: Title for the plot, indicating the measure being plotted.

        Saves the plot as a PDF file in the figures folder and displays it.
        """
        num_optimizers = len(self.top_3_optimizers)

        # Create subplots and shared colorbar
        fig, axes = plt.subplots(1, num_optimizers, figsize=(14, 12))
        axes = axes.flatten()

        # Get global min and max values for the color scale across all matrices
        vmin = min(matrix[:, :, -1].min() for matrix in measure.values())
        vmax = max(matrix[:, :, -1].max() for matrix in measure.values())

        # Plot each heatmap
        for i, (optimizer, matrix) in enumerate(measure.items()):
            matrix = matrix[:, :, -1]

            sns.heatmap(
                matrix,
                annot=True,
                fmt=".4f",
                xticklabels=self.batch_sizes,
                yticklabels=self.learning_rates,
                cmap="viridis",
                ax=axes[i],
                vmin=vmin,
                vmax=vmax,
                cbar=False,  # Only create colorbar for the first subplot
                cbar_kws={"label": title},
                annot_kws={"size": 20},
            )

            axes[i].set_title(f"{optimizer}", fontsize=30)
            axes[i].grid(False)
            if i == 1:
                axes[i].set_xlabel("Batch Size", fontsize=25)
            if i == 0:
                axes[i].set_ylabel("Learning Rate", fontsize=25)

        # Set overall title and layout
        plt.suptitle(title + " for different optimizers", fontsize=40)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Add common colorbar on the right
        fig.colorbar(
            axes[0].collections[0],
            ax=axes,
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )

        plt.savefig(f"../figures/{title}_grid_{self.type_model}.pdf")
        plt.savefig(f"../figures/{title}_grid_{self.type_model}.png")
        plt.show()

    def grid_search_lr_batch(self, optimizer: str) -> tuple:
        """
        Conducts a grid search over learning rates and batch sizes for a given optimizer.

        For each combination of learning rate and batch size, trains the model and records test loss and accuracy.

        Parameters:
            optimizer: Optimizer to use for this grid search.

        Returns:
            A tuple of arrays: (test_accuracy, test_loss) for each combination of learning rate and batch size.
        """

        test_loss = []
        test_accuracy = []

        for lr in self.learning_rates:
            lr_losses = []
            lr_accuracies = []
            for batch_size in self.batch_sizes:
                model = NeuralNetwork(
                    self.best_structure,
                    self.best_activation_functions,
                    self.cost_function,
                    self.type_model,
                    optimizer,
                )
                model.train_network(
                    self.inputs,
                    self.targets,
                    epochs=self.intermediary_epochs,
                    learning_rate=lr,
                    batch_size=batch_size,
                )
                lr_losses.append(model.test_loss)
                lr_accuracies.append(model.test_accuracy)
            test_loss.append(lr_losses)
            test_accuracy.append(lr_accuracies)

        return np.array(test_accuracy), np.array(test_loss)

    def grid_search_for_optimizer(self) -> None:
        """
        Performs a grid search over best three optimizers, storing test loss and accuracy for each combination of
        learning rate and batch size.

        Attributes Updated:
            best_lr_batch: Dictionary with each optimizer's best learning rate and batch size based on test accuracy.
        """

        all_acc = {}
        all_loss = {}

        for optim in self.top_3_optimizers:
            test_accuracy, test_loss = self.grid_search_lr_batch(optimizer=optim)
            all_acc[optim] = test_accuracy
            all_loss[optim] = test_loss

        if self.print_info:
            self.plot_grid(all_loss, f"{self.cost_function_name}")
            self.plot_grid(all_acc, f"{self.accuracy_measure}")

        best = {
            optim: self.find_maximal_accuracy(
                (self.learning_rates, self.batch_sizes), all_acc[optim][:, :, -1]
            )
            for optim in self.top_3_optimizers
        }

        self.best_lr_batch = best

    def make_best(self) -> None:
        """
        Trains the model with the best parameters found (learning rate, batch size, structure, and activation functions) for the three
        top optimizers for 500 epochs. Records test loss and accuracy for each optimizer.

        For the classification model, also records recall, precision, and F1 score.

        Attributes Updated:
            best_losses: List of final test losses for each top three optimizer.
            best_accuracies: List of final test accuracies for each top three optimizer.
            best_optimizer: Optimizer with the highest final accuracy.
        """

        self.long_epochs = 500
        losses = []
        accuracies = []
        accuracies2 = []
        accuracies3 = []
        accuracies4 = []

        self.multiple_accuracy_funcs = True

        for optim in tqdm(self.top_3_optimizers):
            best_lr, best_batch, _ = self.best_lr_batch[optim]
            model = NeuralNetwork(
                self.best_structure,
                self.best_activation_functions,
                self.cost_function,
                self.type_model,
                optim,
                multiple_accuracy_funcs=self.multiple_accuracy_funcs,
            )
            model.train_network(
                self.inputs,
                self.targets,
                epochs=self.long_epochs,
                learning_rate=best_lr,
                batch_size=best_batch,
            )
            losses.append(model.test_loss)

            if self.type_model == "classification" and self.multiple_accuracy_funcs:

                accuracy, recall, precision, f1 = zip(
                    *model.test_accuracy
                )  # Unpack each accuracy type across epochs
                accuracies.append(accuracy)
                accuracies2.append(recall)
                accuracies3.append(precision)
                accuracies4.append(f1)

            else:

                accuracies.append(model.test_accuracy)

        self.best_losses = losses
        self.best_accuracies = accuracies
        self.recall = accuracies2
        self.precision = accuracies3
        self.f1 = accuracies4

        self.best_optimizer = self.find_maximal_accuracy(
            self.top_3_optimizers, accuracies
        )

    def plot_best(self) -> None:
        """
        Plots the final test accuracy and loss for the best hyperparameters found for each top three optimizer optimizer.

        Saves the plots as PDF files in the figures folder and displays them.
        """

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        x_axis = np.arange(50, len(self.best_accuracies[0]))
        for i, acc in enumerate(self.best_accuracies):
            ax.plot(x_axis, acc[50:], label=self.top_3_optimizers[i])

        ax.set_xlabel("Epochs")
        ax.set_ylabel(f"{self.accuracy_measure} value")
        ax.set_title(f"{self.accuracy_measure} for best hyperparameters")
        ax.legend()

        plt.savefig(f"../figures/best_{self.type_model}.pdf")

        plt.show()

        # plot MSE/BCE
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        for i, loss in enumerate(self.best_losses):
            x_axis = np.arange(50, len(loss))
            ax.plot(x_axis, loss[50:], label=self.top_3_optimizers[i])

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss value")
        ax.set_title("Loss for best hyperparameters")
        ax.legend()

        plt.savefig(f"../figures/best_loss_{self.type_model}.pdf")

        plt.show()

    def print_best(self) -> None:
        """
        Prints the best hyperparameters found in the search, including optimizer, structure, activation functions,
        learning rate, batch size, and final accuracy or R^2 score.
        For classification models, also prints final recall, precision, and F1 score.
        """

        print(f"Best optimizer: {self.best_optimizer}")
        print(f"Best structure: {self.best_structure}")
        print(f"Best activation functions: {self.best_activation_functions}")
        print(
            f"Best learning rate and batch size: {self.best_lr_batch[self.best_optimizer][:-1]}"
        )

        # print final accuracy for the best optimizer
        if self.type_model == "continuous":
            print(
                f"Final r^2 score: {self.best_accuracies[self.top_3_optimizers.index(self.best_optimizer)][-1]}"
            )
        elif self.type_model == "classification":
            print(
                f"Final accuracy: {self.best_accuracies[self.top_3_optimizers.index(self.best_optimizer)][-1]}"
            )
            print(
                f"Final recall: {self.recall[self.top_3_optimizers.index(self.best_optimizer)][-1]}"
            )
            print(
                f"Final precision: {self.precision[self.top_3_optimizers.index(self.best_optimizer)][-1]}"
            )
            print(
                f"Final f1: {self.f1[self.top_3_optimizers.index(self.best_optimizer)][-1]}"
            )

    def do_all(self, print_into: bool = False) -> None:
        """
        Executes the entire process for finding the best model configuration, including data generation,
        optimizer selection, structure, activation function search, grid search over learning rates and
        batch sizes, and final training.

        Parameters:
            print_info (bool): If True, displays information and plots at each stage of the search process.
        """

        self.print_info = print_into
        self.generate_data()
        self.find_ok_optimizer()
        self.find_structure()
        self.find_activation_functions()
        self.grid_search_for_optimizer()
        self.make_best()
        self.plot_best()
        self.print_best()
