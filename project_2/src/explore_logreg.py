import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm

from methods import *
from data_gen import CancerData
from log_reg import LogReg

np.random.seed(4155)  # FYS-STK4155


class ExploreLogReg:

    def __init__(self) -> None:
        data = CancerData()
        x_data = np.array(data.x)
        y_data = np.array(data.y).reshape(-1, 1)

        self.input_size = 30
        self.output_size = 1

        X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3, random_state=4155
        )
        inputs = X_train, X_test
        targets = y_train, y_test

        self.inputs = inputs
        self.targets = targets

    def find_minimal_loss(self, thing_to_look_at, test_loss) -> None:
        if isinstance(thing_to_look_at, tuple):
            lr_values, batch_sizes = thing_to_look_at
            min_loss = np.inf
            best_config = (
                None,
                None,
                min_loss,
            )

            for i, loss in enumerate(test_loss):
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    best_config = (
                        lr_values[i // len(batch_sizes)],
                        batch_sizes[i % len(batch_sizes)],
                        float(min_loss),
                    )

            return best_config

        else:
            min_loss = np.inf
            for i, loss in enumerate(test_loss):
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    best = thing_to_look_at[i]

        return best

    # def find_maximal_accuracy(self, thing_to_look_at, test_accuracy) -> None:

    #     if isinstance(thing_to_look_at, tuple):
    #         lr_values, batch_sizes = thing_to_look_at
    #         max_accuracy = -np.inf
    #         best_config = (None, None, max_accuracy)

    #         for i, acc in enumerate(test_accuracy):
    #             if acc[-1] > max_accuracy:
    #                 max_accuracy = acc[-1]
    #                 best_config = (
    #                     lr_values[i // len(batch_sizes)],
    #                     batch_sizes[i % len(batch_sizes)],
    #                     float(max_accuracy),
    #                 )

    #         return best_config

    #     else:
    #         max_accuracy = -np.inf
    #         for i, acc in enumerate(test_accuracy):
    #             if acc[-1] > max_accuracy:
    #                 max_accuracy = acc[-1]
    #                 best = thing_to_look_at[i]

    #     return best

    # def plot(self, measure, variations, title) -> None:

    #     # print the accuracy for each structure
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #     for i, acc in enumerate(measure):
    #         ax.plot(acc, label=variations[i])
    #     ax.set_xlabel("Epochs", fontsize=14)
    #     ax.set_ylabel(title, fontsize=14)
    #     ax.set_title(title + " for different structures", fontsize=16)
    #     ax.legend()
    #     plt.show()

    # def plot_grid(self, measure, title):

    #     num_optimizers = len(self.optimizers)

    #     fig, axes = plt.subplots(1, num_optimizers, figsize=(18, 12))
    #     axes = axes.flatten()

    #     for i, (optimizer, matrix) in enumerate(measure.items()):
    #         matrix = matrix[:, :, -1]

    #         sns.heatmap(
    #             matrix,
    #             annot=True,
    #             fmt=".4f",
    #             xticklabels=self.batch_sizes,
    #             yticklabels=self.learning_rates,
    #             cmap="viridis",
    #             ax=axes[i],
    #         )

    #         axes[i].set_xlabel("Batch Size")
    #         axes[i].set_ylabel("Learning Rate")
    #         axes[i].set_title(f"{optimizer}", fontsize=10)

    #     plt.suptitle("Test " + title + " for different Optimizers", fontsize=16)

    #     plt.tight_layout(rect=[0, 0, 1, 0.95])
    #     plt.show()

    # def grid_search_lr_batch(self):

    #     test_loss = []
    #     test_accuracy = []

    #     for lr in self.learning_rates:
    #         lr_losses = []
    #         lr_accuracies = []
    #         for batch_size in self.batch_sizes:
    #             model = NeuralNetwork(
    #                 self.best_structure,
    #                 self.best_activation_functions,
    #                 self.cost_function,
    #                 self.type_model,
    #                 self.best_optimizer,
    #             )
    #             model.train_network(
    #                 self.inputs,
    #                 self.targets,
    #                 epochs=self.intermediary_epochs,
    #                 learning_rate=lr,
    #                 batch_size=batch_size,
    #             )
    #             lr_losses.append(model.test_loss)
    #             lr_accuracies.append(model.test_accuracy)
    #         test_loss.append(lr_losses)
    #         test_accuracy.append(lr_accuracies)

    #     test_accuracy = np.array(test_accuracy)
    #     test_loss = np.array(test_loss)

    #     return test_accuracy, test_loss

    def grid_search_for_optimizer(self, optimizers):

        all_acc = {}
        all_loss = {}

        for optim in self:
            test_accuracy, test_loss = self.grid_search_lr_batch()
            all_acc[optim] = test_accuracy
            all_loss[optim] = test_loss

        if self.print_info:
            self.plot_grid(all_loss, "loss")
            self.plot_grid(all_acc, "accuracy")

        # find the minimum for each optimizer
        best = {}
        for optim, accuracy in all_acc.items():
            best[optim] = self.find_minimal_loss(
                (self.learning_rates, self.batch_sizes), accuracy
            )

        self.best_lr_batch = best

    # def make_best(self):
    #     self.long_epochs = 300
    #     losses = []
    #     accuracies = []

    #     for optim in tqdm(self):
    #         best_lr, best_batch, _ = self.best_lr_batch[optim]
    #         model = LogReg(
    #             self.input_size,
    #             self.output_size,
    #             l2_reg_param=0,
    #             update_strategy=optim,
    #             train_test_split=True,
    #         )
    #         model.train_network(
    #             self.inputs,
    #             self.targets,
    #             epochs=self.long_epochs,
    #             learning_rate=best_lr,
    #             batch_size=best_batch,
    #         )
    #         losses.append(model.test_loss)
    #         accuracies.append(model.test_accuracy)

    #     self.best_losses = losses
    #     self.best_accuracies = accuracies

    #     self.best_optimizer = self.find_maximal_accuracy(self.optimizers, accuracies)

    # def plot_best(self):

    #     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #     for i, acc in enumerate(self.best_accuracies):
    #         ax.plot(acc, label=self.optimizers[i])
    #     ax.set_xlabel("Epochs", fontsize=14)
    #     if self.type_model == "continuous":
    #         ax.set_ylabel(r"$R^2$", fontsize=14)
    #         ax.set_title(r"$R^2$ for best hyperparameters", fontsize=16)
    #     elif self.type_model == "classification":
    #         ax.set_ylabel("recall value", fontsize=14)
    #         ax.set_title("recall for best hyperparameters", fontsize=16)
    #     ax.legend()
    #     plt.show()

    # def print_best(self):
    #     print(f"Best optimizer: {self.best_optimizer}")
    #     print(f"Best structure: {self.best_structure}")
    #     print(f"Best activation functions: {self.best_activation_functions}")
    #     print(
    #         f"Best learning rate and batch size: {self.best_lr_batch[self.best_optimizer][:-1]}"
    #     )

    #     # print final accuracy for the best optimizer
    #     if self.type_model == "continuous":
    #         print(
    #             f"Final r^2 score: {self.best_accuracies[self.optimizers.index(self.best_optimizer)][-1]}"
    #         )
    #     elif self.type_model == "classification":
    #         print(
    #             f"Final recall: {self.best_accuracies[self.optimizers.index(self.best_optimizer)][-1]}"
    #         )

    def do_all(self, print_into: bool = False) -> None:

        self.print_info = print_into
        self.grid_search_for_optimizer()
        self.make_best()
        self.plot_best()
        self.print_best()
