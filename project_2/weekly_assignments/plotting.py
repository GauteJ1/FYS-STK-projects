import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_gen import DataGen, Poly1D2Deg
from learn_rate import Update_Beta
from grad_desc import Model



class Plotting:

    

    def __init__(self, model: Model) -> None:



        self.x = model.x
        self.y = model.y
        self.n = model.n

        self.MSE = model.MSE_list
        self.epochs = model.epoch_list
        self.beta = model.beta

        self.a = model.a
        self.b = model.b
        self.c = model.c


    def __config(self):
        sns.set_theme(palette="bright")
        sns.set_palette("Set2")
        #plt.style.use("plot_settings.mplstyle")

    def plot_loss(self, xlim = None, label = None):
        self.__config()
        plt.plot(self.epochs, self.MSE, label = label)
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title("Mean squared error (MSE) against epochs")
        plt.xlim(0,xlim)
        plt.legend()
        #plt.show()

    def plot_betas(self):
        self.__config()
        plt.plot(self.x, self.y, '*', label = "Data")
        #plt.plot(self.x, self.a + self.b*self.x + self.c*self.x**2, '*', label = "Real")
        plt.plot(self.x, self.beta[0] + self.beta[1]*self.x + self.beta[2]*self.x**2, 'o', label = "Predicted")
        plt.legend()
        plt.show()