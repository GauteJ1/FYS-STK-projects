import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm

from neural_network import NeuralNetwork
from methods import *
from data_gen import FrankeDataGen, CancerData

np.random.seed(4155) # FYS-STK4155 

class Exploration:

    def __init__(self, type_model: str,
                 optimizers: list[str], 
                 layer_sizes: list[int],
                 num_hidden_layers: list[int],
                 learning_rates: list[float],
                 batch_sizes: list[float],
                 activation_functions: list[str],
                 ) -> None:
        
        """
        Initializes the Exploration class for neural network experimentation.

        Parameters:
            type_model: Model type ('continuous' or 'classification').
            optimizers: Optimizers to test.
            layer_sizes: Hidden layer sizes to explore.
            num_hidden_layers: Number of hidden layers to explore.
            learning_rates: Learning rates to test.
            batch_sizes: Batch sizes to test.
            activation_functions: Activation functions to test.
        """

        self.type_model = type_model    
        self.optimizers = optimizers
        self.layer_sizes = layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.activation_functions = activation_functions

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

        if self.type_model == 'continuous':
            data = FrankeDataGen()
            x_data = jnp.column_stack((data.x.flatten(), data.y.flatten()))  
            y_data = data.z.ravel().reshape(-1, 1) 

            self.input_size = 2
            self.output_size = 1
            self.cost_function = "MSE"
        
        elif self.type_model == 'classification':
            data = CancerData()
            x_data = np.array(data.x)
            y_data = np.array(data.y).reshape(-1, 1)

            self.input_size = 30
            self.output_size = 1
            self.cost_function = "BinaryCrossEntropy"

        else: 
            raise ValueError("Type model must be either 'continuous' or 'classification'")
        
        np.random.seed(4155)

        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=4155)
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
                    best_config = (lr_values[i // len(batch_sizes)], 
                                   batch_sizes[i % len(batch_sizes)], 
                                   float(max_accuracy))
                    
            return best_config
        
        else:
            max_accuracy = -np.inf
            for i, acc in enumerate(test_accuracy):
                if acc[-1] > max_accuracy:
                    max_accuracy = acc[-1]
                    best = configuration[i]

        return best
    
    def plot(self, measure: list, configurations: list, title: str) -> None: 
            
        """
        Plots the given measure (accuracy or loss) for different configurations over epochs.

        Parameters:
            measure: List of values (accuracy or loss) over epochs for each configuration.
            configurations: List of configuration names or labels for the plot legend.
            title: Title for the plot, indicating the measure being plotted.
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i, acc in enumerate(measure):
            ax.plot(acc, label=configurations[i])
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel(title, fontsize=14)
        ax.set_title(title + " for different structures", fontsize=16)
        ax.legend()
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
        self.intermediary_hidden_layers = [4,8]

        if self.type_model == 'classification':
            self.intermediary_activation_funcs = ["ReLU", "ReLU", "sigmoid"]
            self.intermediary_batch_size = 200
        else:
            self.intermediary_activation_funcs = ["ReLU", "ReLU", "identity"]  
            self.intermediary_batch_size = 1000

        network_shape = [self.input_size] + self.intermediary_hidden_layers + [self.output_size]
        test_loss = []
        test_accuracy = []

        for optim in self.optimizers: 
            model = NeuralNetwork(network_shape, self.intermediary_activation_funcs, self.cost_function, self.type_model, optim)
            model.train_network(self.inputs, self.targets, self.intermediary_epochs, self.intermediary_lr, self.intermediary_batch_size)
            test_loss.append(model.test_loss)
            test_accuracy.append(model.test_accuracy)

        self.best_optimizer = self.find_maximal_accuracy(self.optimizers, test_accuracy)

        if self.print_info:
            print(f"Best optimizer: {self.best_optimizer}")
            self.plot(test_loss, self.optimizers, "Loss")
            self.plot(test_accuracy, self.optimizers, "Accuracy")

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

        for layers in self.num_hidden_layers:
            for sizes in product(self.layer_sizes, repeat=layers):
                permutations.append(sizes)

                network_shape = [self.input_size] + list(sizes) + [self.output_size]
                
                model = NeuralNetwork(network_shape, self.intermediary_activation_funcs, self.cost_function, "continuous", self.best_optimizer)
                model.train_network(self.inputs, self.targets, epochs=self.intermediary_epochs, learning_rate=self.intermediary_lr, batch_size=self.intermediary_batch_size)
                
                test_loss.append(model.test_loss)
                test_accuracy.append(model.test_accuracy)

        best = self.find_maximal_accuracy(permutations, test_accuracy)
        best_list = list(best)    
        
        self.best_structure = [self.input_size] + best_list + [self.output_size]
        self.num_hidden_layers = len(best_list) # update number of hidden layers to the best

        if self.print_info:
            print(f"Best structure: {self.best_structure}")
            self.plot(test_loss, permutations, "Loss")
            self.plot(test_accuracy, permutations, "Accuracy")
        

    def find_activation_functions(self) -> None:

        """
        Identifies the best activation functions for each hidden layer by testing different combinations.

        For each combination of activation functions, trains the model and records test accuracy.
        Selects the activation functions that yield the highest accuracy.

        Attributes Updated:
            best_activation_functions: List of optimal activation functions for each layer.
        """

        combinations_of_activations = list(product(self.activation_functions, repeat=self.num_hidden_layers))
        final_activation = "sigmoid" if self.type_model == "classification" else "identity"

        combinations_of_activations = [list(comb) + [final_activation] for comb in combinations_of_activations]

        test_loss = []
        test_accuracy = []
        for combination in combinations_of_activations:
            model = NeuralNetwork(self.best_structure, combination, self.cost_function, "continuous", self.best_optimizer)
            model.train_network(self.inputs, self.targets, epochs=self.intermediary_epochs, learning_rate=self.intermediary_lr, batch_size=self.intermediary_batch_size)
            test_loss.append(model.test_loss)
            test_accuracy.append(model.test_accuracy)

        best = self.find_maximal_accuracy(combinations_of_activations, test_accuracy)
        self.best_activation_functions = list(best)

        if self.print_info:
            print(f"Best activation functions: {self.best_activation_functions}")
            self.plot(test_loss, combinations_of_activations, "Loss")
            self.plot(test_accuracy, combinations_of_activations, "Accuracy")

    def plot_grid(self, measure: dict, title: str):

        """
        Plots a heatmap of the given measure (accuracy or loss) for each optimizer, across learning rates and batch sizes.

        Parameters:
            measure: Dictionary with optimizers as keys and 3D matrices of values (accuracy or loss) as values.
            title: Title for the plot, indicating the measure being plotted.
        """

        num_optimizers = len(self.optimizers)


        fig, axes = plt.subplots(1,num_optimizers, figsize=(18, 12))  
        axes = axes.flatten() 

        for i, (optimizer, matrix) in enumerate(measure.items()):
            matrix = matrix[:, :, -1]  

            
            sns.heatmap(matrix, annot=True, fmt=".4f", 
                        xticklabels=self.batch_sizes, 
                        yticklabels=self.learning_rates, 
                        cmap="viridis", ax=axes[i])  
            
            axes[i].set_xlabel("Batch Size")
            axes[i].set_ylabel("Learning Rate")
            axes[i].set_title(f"{optimizer}", fontsize=10)  

        
        plt.suptitle("Test " + title + " for different Optimizers", fontsize=16)

        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  
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
                model = NeuralNetwork(self.best_structure, self.best_activation_functions, 
                                    self.cost_function, self.type_model, optimizer)  
                model.train_network(self.inputs, self.targets, epochs=self.intermediary_epochs, 
                                    learning_rate=lr, batch_size=batch_size)
                lr_losses.append(model.test_loss)
                lr_accuracies.append(model.test_accuracy)
            test_loss.append(lr_losses)
            test_accuracy.append(lr_accuracies)

        return np.array(test_accuracy), np.array(test_loss)

    
    def grid_search_for_optimizer(self) -> None:
        """
        Performs a grid search over all optimizers, storing test loss and accuracy for each combination of
        learning rate and batch size.

        Attributes Updated:
            best_lr_batch: Dictionary with each optimizer's best learning rate and batch size based on test accuracy.
        """

        all_acc = {}
        all_loss = {}

        for optim in self.optimizers:
            test_accuracy, test_loss = self.grid_search_lr_batch(optimizer=optim) 
            all_acc[optim] = test_accuracy   
            all_loss[optim] = test_loss      

        if self.print_info:
            self.plot_grid(all_loss, "Loss")
            self.plot_grid(all_acc, "Accuracy")

        best = {optim: self.find_maximal_accuracy((self.learning_rates, self.batch_sizes), all_acc[optim][:, :, -1])
                for optim in self.optimizers}

        self.best_lr_batch = best

    def make_best(self) -> None:
        
        """
        Trains the model with the best parameters found (optimizer, learning rate, batch size, structure, 
        and activation functions) for an extended number of epochs. Records test loss and accuracy for each optimizer.

        Attributes Updated:
            best_losses: List of final test losses for each optimizer.
            best_accuracies: List of final test accuracies for each optimizer.
            best_optimizer: Optimizer with the highest final accuracy.
        """

        self.long_epochs = 300
        losses = []
        accuracies = []

        for optim in tqdm(self.optimizers):
            best_lr, best_batch, _ = self.best_lr_batch[optim]
            model = NeuralNetwork(self.best_structure, self.best_activation_functions, self.cost_function, self.type_model, optim)
            model.train_network(self.inputs, self.targets, epochs=self.long_epochs, learning_rate=best_lr, batch_size=best_batch)
            losses.append(model.test_loss)
            accuracies.append(model.test_accuracy)

        self.best_losses = losses
        self.best_accuracies = accuracies

        self.best_optimizer = self.find_maximal_accuracy(self.optimizers, accuracies)
    
    def plot_best(self) -> None:
        
        """
        Plots the accuracy over epochs for each optimizer using the best configurations identified in the search.

        The plot displays either R^2 for continuous models or recall for classification models.
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i, acc in enumerate(self.best_accuracies):
            ax.plot(acc, label=self.optimizers[i])
        ax.set_xlabel("Epochs", fontsize=14)
        if self.type_model == 'continuous':
            ax.set_ylabel(r"$R^2$", fontsize=14)
            ax.set_title(r"$R^2$ for best hyperparameters", fontsize=16)
        elif self.type_model == 'classification':
            ax.set_ylabel("recall value", fontsize=14)
            ax.set_title("recall for best hyperparameters", fontsize=16)
        ax.legend()
        plt.show()

    def print_best(self) -> None:
        
        """
        Prints the best hyperparameters found in the search, including optimizer, structure, activation functions, 
        learning rate, batch size, and final accuracy or R^2 score.
        """

        print(f"Best optimizer: {self.best_optimizer}")
        print(f"Best structure: {self.best_structure}")
        print(f"Best activation functions: {self.best_activation_functions}")
        print(f"Best learning rate and batch size: {self.best_lr_batch[self.best_optimizer][:-1]}")

        # print final accuracy for the best optimizer
        if self.type_model == 'continuous':
            print(f"Final r^2 score: {self.best_accuracies[self.optimizers.index(self.best_optimizer)][-1]}")
        elif self.type_model == 'classification':
            print(f"Final recall: {self.best_accuracies[self.optimizers.index(self.best_optimizer)][-1]}")


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