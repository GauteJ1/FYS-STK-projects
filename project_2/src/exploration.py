import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import display, clear_output
from itertools import product, combinations_with_replacement
from tqdm import tqdm
import os
import sys

from neural_network import NeuralNetwork
from methods import *
from data_gen import FrankeDataGen, CancerData
from grad_desc import Model

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
        
        self.type_model = type_model    
        self.optimizers = optimizers
        self.layer_sizes = layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.activation_functions = activation_functions

    def generate_data(self) -> None:
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

        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
        inputs = X_train, X_test 
        targets = y_train, y_test

        self.inputs = inputs
        self.targets = targets

    def find_minimal_loss(self, thing_to_look_at, test_loss) -> None:
        
        if isinstance(thing_to_look_at, tuple):
            lr_values, batch_sizes = thing_to_look_at
            min_loss = np.inf
            best_config = (None, None, min_loss)  # (learning_rate, batch_size, final_loss)

            for i, loss in enumerate(test_loss):
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    best_config = (lr_values[i // len(batch_sizes)], 
                                   batch_sizes[i % len(batch_sizes)], 
                                   float(min_loss))

            return best_config

        
        else: 
            min_loss = np.inf
            for i, loss in enumerate(test_loss):
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    best = thing_to_look_at[i]

        return best
    
    def find_maximal_accuracy(self, thing_to_look_at, test_accuracy) -> None:

        if isinstance(thing_to_look_at, tuple):
            lr_values, batch_sizes = thing_to_look_at
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
                    best = thing_to_look_at[i]

        return best

    def find_ok_optimizer(self):
        
        # parameters just for this initial search for a good enough optimizer
        self.intermediary_lr = 0.001
        self.intermediary_epochs = 100
        self.intermediary_hidden_layers = [8,8]

        if self.type_model == 'classification':
            self.intermediary_activation_funcs = ["ReLU", "ReLU", "sigmoid"]
            self.intermediary_batch_size = 400
        else:
            self.intermediary_activation_funcs = ["ReLU", "ReLU", "identity"]  
            self.intermediary_batch_size = 8000

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

        if self.print_info:
            # plot the loss for each optimizer
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            for i, loss in enumerate(test_loss):
                print(f"for {self.optimizers[i]}: {loss[-1]}")
                ax.plot(loss, label=self.optimizers[i])
            ax.set_xlabel("Epochs", fontsize=14)
            ax.set_ylabel("Loss", fontsize=14)
            ax.set_title("Loss for different optimizers", fontsize=16)
            ax.legend()
            plt.show()

    def find_structure(self):

        test_loss = []
        test_accuracy = []
        permutations = []

        for layers in self.num_hidden_layers:
            for sizes in product(self.layer_sizes, repeat=layers):
                permutations.append(sizes)

                network_shape = [self.input_size] + list(sizes) + [self.output_size]
                
                #print(layers, sizes)
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

    def find_activation_functions(self):

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

    def grid_search_lr_batch(self):

        test_loss = []
        test_accuracy = []

        for lr in self.learning_rates: 
            for batch_size in self.batch_sizes: 
                model = NeuralNetwork(self.best_structure, self.best_activation_functions, self.cost_function, "continuous", self.best_optimizer)
                model.train_network(self.inputs, self.targets, epochs=self.intermediary_epochs, learning_rate=lr, batch_size=batch_size)
                test_loss.append(model.test_loss)
                test_accuracy.append(model.test_accuracy)

        return test_accuracy
    
    def grid_search_for_optimizer(self):

        all = {}

        for optim in self.optimizers: 
            test_accuracy = self.grid_search_lr_batch()
            all[optim] = test_accuracy

        # find the minimum for each optimizer
        best = {}
        for optim, accuracy in all.items():
            best_accuracy = self.find_maximal_accuracy((self.learning_rates, self.batch_sizes), accuracy)
            best[optim] = best_accuracy

        self.best_lr_batch = best


    def make_best(self):
        self.long_epochs = 500
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
    
    def plot_best(self):

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i, acc in enumerate(self.best_accuracies):
            ax.plot(acc, label=self.optimizers[i])
        ax.set_xlabel("Epochs", fontsize=14)
        if self.type_model == 'continuous':
            ax.set_ylabel(r"$R^2$", fontsize=14)
            ax.set_title(r"$R^2$ for best hyperparameters", fontsize=16)
        elif self.type_model == 'classification':
            ax.set_ylabel("f1 score", fontsize=14)
            ax.set_title("f1 score for best hyperparameters", fontsize=16)
        ax.legend()
        plt.show()

    def print_best(self):
        print(f"Best optimizer: {self.best_optimizer}")
        print(f"Best structure: {self.best_structure}")
        print(f"Best activation functions: {self.best_activation_functions}")
        print(f"Best learning rate and batch size: {self.best_lr_batch[self.best_optimizer][:-1]}")

        # print final accuracy for the best optimizer
        if self.type_model == 'continuous':
            print(f"Final r^2 score: {self.best_accuracies[self.optimizers.index(self.best_optimizer)][-1]}")
        elif self.type_model == 'classification':
            print(f"Final f1 socre: {self.best_accuracies[self.optimizers.index(self.best_optimizer)][-1]}")


    def do_all(self, print_into: bool = False) -> None:

        self.print_info = print_into        
        self.generate_data()
        self.find_ok_optimizer()
        self.find_structure()
        self.find_activation_functions()
        self.grid_search_for_optimizer()
        self.make_best()
        self.plot_best()
        self.print_best()
        

            


        
        
