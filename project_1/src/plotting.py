import matplotlib.pyplot as plt
import numpy as np

from reg_models import OLSModel, RidgeModel, LassoModel, RegModel
from data_gen import FrankeDataGen, TerrainDataGen
from data_handling import DataHandler
import plot_utils


class Plotting:
    def __init__(self, data_points: int = 51, real_data: bool = False):
        if real_data:
            data = TerrainDataGen(data_points)
        else:
            data = FrankeDataGen(data_points)

        self.handler = DataHandler(data)
        self.lmbdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
        np.random.seed(1234)

    def __config(self):
        plt.style.use("plot_settings.mplstyle")
        plot_utils.plot_config()

    def plot_OLS(self, y_axis: str = "MSE", min_deg: int = 1, max_deg: int = 6):
        self.__config()
        x_data = list(range(min_deg, max_deg + 1))
        y_data_train = []
        y_data_test = []

        ols = OLSModel(self.handler)

        if not (y_axis == "MSE" or y_axis == "R2"):
            raise RuntimeError("y_axis must be either MSE or R2")

        for deg in x_data:
            ols.fit_model(deg)
            if y_axis == "MSE":
                y_data_train.append(ols.MSE(train=True))
                y_data_test.append(ols.MSE(train=False))
            else:
                y_data_train.append(ols.R2(train=True))
                y_data_test.append(ols.R2(train=False))

        plt.plot(x_data, y_data_train, color=plot_utils.plot_colors(0), label="Train")
        plt.plot(x_data, y_data_test, color=plot_utils.plot_colors(1), label="Test")

        plt.title(f"OLS: {y_axis} for different model complexities")
        plt.legend()
        plt.xlabel("Degree")
        plt.ylabel(y_axis)

    def plot_lambda(self, model: str = "Ridge", y_axis: str = "MSE", deg: int = 2):
        self.__config()
        lmbdas = self.lmbdas
        y_data_train = []
        y_data_test = []

        mod: RegModel
        if model == "Ridge":
            mod = RidgeModel(self.handler)
        else:
            mod = LassoModel(self.handler)

        for lmbda in lmbdas:
            mod.fit_model(2, lmbda=lmbda)
            if y_axis == "MSE":
                y_data_train.append(mod.MSE(train=True))
                y_data_test.append(mod.MSE(train=False))
            else:
                y_data_train.append(mod.R2(train=True))
                y_data_test.append(mod.R2(train=False))

        plt.plot(lmbdas, y_data_train, color=plot_utils.plot_colors(0), label="Train")
        plt.plot(lmbdas, y_data_test, color=plot_utils.plot_colors(1), label="Test")
        plt.xscale("log")
        plt.title(f"{model}: {y_axis} for different lambdas")
        plt.legend()
        plt.xlabel("Lambda")
        plt.ylabel(y_axis)

    def __plot_betas_deg(
        self, model: str = "OLS", lmbda: float = 0.1, max_deg: int = 5
    ):
        x_data = list(range(1, max_deg + 1))
        y_data = []

        length = 21

        mod: RegModel
        if model == "OLS":
            mod = OLSModel(self.handler)
        elif model == "Ridge":
            mod = RidgeModel(self.handler)
        elif model == "Lasso":
            mod = LassoModel(self.handler)
        else:
            raise RuntimeError(f"{model} not a valid model type.")

        for deg in x_data:
            if model == "OLS":
                mod.fit_model(deg)
            else:
                mod.fit_model(deg, lmbda)
            betas = mod.opt_beta[:, 0]
            extra_zeros = length - len(betas)
            y_data.append(list(betas) + [0] * extra_zeros)

        return x_data, y_data

    def plot_betas(self, model: str = "OLS", lmbda: float = 0.1, max_deg: int = 5):
        self.__config()
        x_data, y_data = self.__plot_betas_deg(model, lmbda, max_deg)
        plt.plot(x_data, y_data)
        plt.xlabel("Degree")
        plt.ylabel(r"Values of $\beta$'s")
        plt.title(rf"Value of $\beta$'s for {model} with $\lambda = {lmbda}$")

    def plot_all_betas(self, lmbda: float = 0.1, max_deg: int = 5):
        models = ["OLS", "Ridge", "Lasso"]

        x_data, y_OLS = self.__plot_betas_deg("OLS", max_deg=max_deg)
        _, y_Ridge = self.__plot_betas_deg("Ridge", lmbda, max_deg)
        _, y_Lasso = self.__plot_betas_deg("Lasso", lmbda, max_deg)

        fig, axs = plt.subplots(1, 3, sharey="row")

        for y_data, ax, model in zip([y_OLS, y_Ridge, y_Lasso], axs, models):
            ax.plot(x_data, y_data)
            ax.set_title(model)
            ax.grid()
            ax.set_xlabel("Degree")
            if model == "OLS":
                ax.set_ylabel(r"Values of $\beta$'s")

        fig.set_figheight(3.6)
        fig.set_figwidth(6.4)
        fig.suptitle(rf"Value of $\beta$'s with $\lambda = {lmbda}$")
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

    def plot_betas_lambda(self, model: str = "OLS", deg: int = 3):
        x_data = self.lmbdas
        y_data = []

        length = 21

        mod: RegModel
        if model == "OLS":
            mod = OLSModel(self.handler)
        elif model == "Ridge":
            mod = RidgeModel(self.handler)
        elif model == "Lasso":
            mod = LassoModel(self.handler)
        else:
            raise RuntimeError(f"{model} not a valid model type.")

        for lmbda in x_data:
            if model == "OLS":
                mod.fit_model(deg)
            else:
                mod.fit_model(deg, lmbda)
            betas = mod.opt_beta[:, 0]
            extra_zeros = length - len(betas)
            y_data.append(list(betas) + [0] * extra_zeros)

        plt.plot(x_data, y_data)
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"Values of $\beta$'s")
        plt.xscale("log")
        plt.title(rf"Value of $\beta$'s for {model} with $deg = {deg}$")

    def plot_bias_var_bootstrap(
        self, samples: int = 100, min_deg: int = 1, max_deg: int = 10
    ):
        handler = self.handler
        model = OLSModel(handler)

        degs, errors, biases, vars = model.bootstrap_mult_degs(
            min_deg=min_deg, max_deg=max_deg, samples=samples
        )

        plt.plot(degs, errors, label="MSE")
        plt.plot(degs, biases, label="Bias")
        plt.plot(degs, vars, label="Var")

        plt.title(f"Title")
        plt.legend()
        plt.xlabel("Degree")

    def plot_mse_bootstrap(
        self, samples: int = 100, min_deg: int = 1, max_deg: int = 12
    ):

        handler = self.handler
        model = OLSModel(handler)

        degs, errors, biases, vars = model.bootstrap_mult_degs(
            min_deg=min_deg, max_deg=max_deg, samples=samples
        )

        plt.plot(degs, errors, label="MSE")

        plt.title(f"Title")
        plt.legend()
        plt.xlabel("Degree")

    def plot_mse_cv(
        self,
        kfolds: int = 5,
        min_deg: int = 1,
        max_deg: int = 12,
        model: str = "OLS",
        lmbda: float = 0,
    ):
        handler = self.handler

        if model == "OLS":
            model = OLSModel(handler)
            degs, errors = model.cv_mult_degs(
                min_deg=min_deg, max_deg=max_deg, kfolds=kfolds
            )
        elif model == "Ridge":
            model = RidgeModel(handler)
            degs, errors = model.cv_mult_degs(
                min_deg=min_deg, max_deg=max_deg, kfolds=kfolds, lmbda=lmbda
            )
        elif model == "Lasso":
            model = LassoModel(handler)
            degs, errors = model.cv_mult_degs(
                min_deg=min_deg, max_deg=max_deg, kfolds=kfolds, lmbda=lmbda
            )

        plt.plot(degs, errors, label="MSE")

        plt.title(f"Title")
        plt.legend()
        plt.xlabel("Degree")
