import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from reg_models import OLSModel, RidgeModel, LassoModel, RegModel
from data_gen import FrankeDataGen, TerrainDataGen
from data_handling import DataHandler


class Plotting:
    def __init__(self, data_points: int = 51, data: str = "Franke", seed: int = 41):
        np.random.seed(seed)

        if data == "Terrain":
            data = TerrainDataGen(data_points)
        elif data == "Franke":
            data = FrankeDataGen(data_points)
        elif data == "Franke_Noise":
            data = FrankeDataGen(data_points, noise=True)

        else:
            raise RuntimeError(
                "Not a valid data-type. Valid data-types are: \nTerrain, Franke and Franke_Noise"
            )

        self.handler = DataHandler(data)
        self.lmbdas = np.array(
            [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) * 10 ** (n) for n in range(-5, 1)]
        ).ravel()

    def __config(self):
        sns.set_theme(palette="bright")
        sns.set_palette("Set2")
        plt.style.use("plot_settings.mplstyle")

    def plot_MSE_R2(
        self,
        y_axis: str = "MSE",
        model_name: str = "OLS",
        lmbda: float = 0,
        max_deg: int = 10,
    ):
        self.__config()
        x_data = list(range(1, max_deg + 1))
        y_data_train = []
        y_data_test = []

        model: RegModel
        if model_name == "OLS":
            model = OLSModel(self.handler)
        elif model_name == "Ridge":
            model = RidgeModel(self.handler)
        elif model_name == "Lasso":
            model = LassoModel(self.handler)

        if not (y_axis == "MSE" or y_axis == "R2"):
            raise RuntimeError("y_axis must be either MSE or R2")

        for deg in x_data:
            model.fit_simple_model(deg=deg, lmbda=lmbda)
            z_tilde_train = model.predict(model.X_train)
            z_tilde_test = model.predict(model.X_test)
            if y_axis == "MSE":
                y_data_train.append(model.MSE(z_tilde_train, model.z_train))
                y_data_test.append(model.MSE(z_tilde_test, model.z_test))

            else:
                y_data_train.append(model.R2(z_tilde_train, model.z_train))
                y_data_test.append(model.R2(z_tilde_test, model.z_test))

        plt.plot(x_data, y_data_train, label="Train")
        plt.plot(x_data, y_data_test, label="Test")

        if y_axis == "MSE":
            min_mse = min(y_data_test)
            min_mse_deg = x_data[y_data_test.index(min_mse)]
            plt.plot(
                min_mse_deg,
                min_mse,
                "ro",
                label=f"Min MSE: {min_mse:.4f} at deg: {min_mse_deg}",
            )

        plt.title(f"{model_name}: {y_axis}")
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel(y_axis)

    def plot_lambda(self, model_name: str = "Ridge", y_axis: str = "MSE", deg: int = 2):
        self.__config()
        lmbdas = self.lmbdas
        y_data_train = []
        y_data_test = []

        model: RegModel
        if model_name == "Ridge":
            model = RidgeModel(self.handler)
        elif model_name == "Lasso":
            model = LassoModel(self.handler)

        for lmbda in lmbdas:
            model.fit_simple_model(deg=deg, lmbda=lmbda)
            z_tilde_train = model.predict(model.X_train)
            z_tilde_test = model.predict(model.X_test)
            if y_axis == "MSE":
                y_data_train.append(model.MSE(z_tilde_train, model.z_train))
                y_data_test.append(model.MSE(z_tilde_test, model.z_test))
            else:
                y_data_train.append(model.R2(z_tilde_train, model.z_train))
                y_data_test.append(model.R2(z_tilde_test, model.z_test))

        plt.plot(lmbdas, y_data_train, label="Train")
        plt.plot(lmbdas, y_data_test, label="Test")
        plt.xscale("log")
        plt.title(f"{model_name}: {y_axis} for different lambdas")
        plt.legend()
        plt.xlabel("Lambda")
        plt.ylabel(y_axis)

    def plot_betas_deg(
        self, model_name: str = "OLS", lmbda: float = 0., max_deg: int = 5
    ):
        x_data = list(range(1, max_deg + 1))
        y_data = []

        length = ((max_deg + 1) * (max_deg + 2)) // 2 - 1

        model: RegModel
        if model_name == "OLS":
            model = OLSModel(self.handler)
        elif model_name == "Ridge":
            model = RidgeModel(self.handler)
        elif model_name == "Lasso":
            model = LassoModel(self.handler)
        else:
            raise RuntimeError(f"{model_name} not a valid model type.")

        for deg in x_data:
            model.fit_simple_model(deg=deg, lmbda=lmbda)
            betas = model.opt_beta[:, 0]
            extra_zeros = length - len(betas)
            y_data.append(list(betas) + [0] * extra_zeros)

        self.__config()
        plt.plot(x_data, y_data)
        plt.xlabel("Polynomial degree")
        plt.ylabel(r"Values of $\beta$'s")
        plt.title(rf"Value of $\beta$'s for {model_name}")
        plt.legend()

    def plot_betas_lambda(
        self, model_name: str = "OLS", deg: int = 3, opt_lambda: float = 0.1
    ):
        x_data = self.lmbdas
        y_data = []

        length = 21

        model: RegModel
        if model_name == "OLS":
            model = OLSModel(self.handler)
        elif model_name == "Ridge":
            model = RidgeModel(self.handler)
        elif model_name == "Lasso":
            model = LassoModel(self.handler)
        else:
            raise RuntimeError(f"{model_name} not a valid model type.")

        for lmbda in x_data:
            model.fit_simple_model(deg=deg, lmbda=lmbda)
            betas = model.opt_beta[:, 0]
            extra_zeros = length - len(betas)
            y_data.append(list(betas) + [0] * extra_zeros)

        self.__config()
        plt.plot(x_data, y_data)
        plt.axvline(x=opt_lambda, color="b", linestyle="--", label=r"Optimal $\lambda$")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"Values of $\beta$'s")
        plt.xscale("log")
        plt.title(rf"Value of $\beta$'s for {model_name} with $deg = {deg}$")
        plt.legend()

    def plot_bootstrap_MSE_R2(
        self,
        y_axis: str = "MSE",
        model_name: str = "OLS",
        lmbda: float = 0,
        max_deg: int = 10,
        n_samples: int = 200,
    ):
        self.__config()
        x_data = list(range(1, max_deg + 1))
        y_data = []

        model: RegModel
        ridge_lambda = 0
        lasso_lambda = 0
        if model_name == "OLS":
            model = OLSModel(self.handler)
        elif model_name == "Ridge":
            model = RidgeModel(self.handler)
            ridge_lambda = lmbda
        elif model_name == "Lasso":
            model = LassoModel(self.handler)
            lasso_lambda = lmbda

        if not (y_axis == "MSE" or y_axis == "R2"):
            raise RuntimeError("y_axis must be either MSE or R2")

        for deg in x_data:
            if y_axis == "MSE":
                y_data.append(
                    model.MSE_bootstrap(
                        n_samples=n_samples,
                        degree=deg,
                        ridge_lambda=ridge_lambda,
                        lasso_lambda=lasso_lambda,
                    )
                )
            else:
                y_data.append(
                    model.R2_bootstrap(
                        n_samples=n_samples,
                        degree=deg,
                        ridge_lambda=ridge_lambda,
                        lasso_lambda=lasso_lambda,
                    )
                )

        plt.plot(x_data, y_data, label="MSE")

        plt.title(f"{model_name}: {y_axis}")
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel(y_axis)

    def plot_bootstrap_bias_var(
        self,
        model_name: str = "OLS",
        lmbda: float = 0,
        max_deg: int = 10,
        n_samples: int = 200,
    ):
        self.__config()
        x_data = list(range(1, max_deg + 1))
        MSE = []
        bias = []
        var = []

        model: RegModel
        if model_name == "OLS":
            model = OLSModel(self.handler)
        elif model_name == "Ridge":
            model = RidgeModel(self.handler)
        elif model_name == "Lasso":
            model = LassoModel(self.handler)

        for deg in x_data:
            MSE_, bias_, var_ = model.bias_var_bootstrap(n_samples, deg)
            MSE.append(MSE_)
            bias.append(bias_)
            var.append(var_)

        plt.plot(x_data, MSE, label="MSE")
        plt.plot(x_data, bias, label=r"Bias$^2$")
        plt.plot(x_data, var, label="Variance")

        min_mse = min(MSE)
        min_mse_deg = x_data[MSE.index(min_mse)]
        plt.plot(
            min_mse_deg,
            min_mse,
            "ro",
            label=f"Min MSE: {min_mse:.4f} at deg: {min_mse_deg}",
        )

        plt.title(
            f"{model_name}: MSE, bias and variance"
        )
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE / Bias / Variance")

    def plot_cross_val_MSE(
        self,
        model_name: str = "OLS",
        lmbda: float = 0,
        max_deg: int = 10,
        kfolds: int = 5,
    ):
        self.__config()
        x_data = list(range(1, max_deg + 1))
        train_data = []
        test_data = []

        model: RegModel
        ridge_lambda = 0
        lasso_lambda = 0
        if model_name == "OLS":
            model = OLSModel(self.handler)
        elif model_name == "Ridge":
            model = RidgeModel(self.handler)
            ridge_lambda = lmbda
        elif model_name == "Lasso":
            model = LassoModel(self.handler)
            lasso_lambda = lmbda

        model.make_cross_val_split(kfolds=kfolds)
        for deg in x_data:
            train, test = model.MSE_cross_validation(deg, ridge_lambda, lasso_lambda)
            train_data.append(train)
            test_data.append(test)

        plt.plot(x_data, train_data, label="MSE Train")
        plt.plot(x_data, test_data, label="MSE Test")

        plt.title(
            f"{model_name}: MSE for cross-validation, kfolds: {kfolds}"
        )
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")

    def plot_all(
        self,
        ridge_lambda: float,
        lasso_lambda: float,
        max_degree: int,
        max_bootstrap: int,
        n_samples: int,
        k_folds: int,
    ):
        self.__config()
        x_data = list(range(1, max_degree + 1))
        ols_reg = []
        ridge_reg = []
        lasso_reg = []
        ols_boot = []
        ridge_boot = []
        lasso_boot = []
        ols_cv = []
        ridge_cv = []
        lasso_cv = []

        ols = OLSModel(self.handler)
        ridge = RidgeModel(self.handler)
        lasso = LassoModel(self.handler)

        ols.make_cross_val_split(kfolds=k_folds)
        ridge.make_cross_val_split(kfolds=k_folds)
        lasso.make_cross_val_split(kfolds=k_folds)

        for deg in range(1, max_bootstrap):
            ols_boot.append(
                ols.MSE_bootstrap(
                    n_samples=n_samples,
                    degree=deg,
                )
            )
            ridge_boot.append(
                ridge.MSE_bootstrap(
                    n_samples=n_samples,
                    degree=deg,
                    ridge_lambda=ridge_lambda,
                )
            )
            lasso_boot.append(
                lasso.MSE_bootstrap(
                    n_samples=n_samples,
                    degree=deg,
                    lasso_lambda=lasso_lambda,
                )
            )

        for deg in x_data:
            ols.fit_simple_model(deg)
            z_tilde_test = ols.predict(ols.X_test)
            ols_reg.append(ols.MSE(z_tilde_test, ols.z_test))

            ridge.fit_simple_model(deg)
            z_tilde_test = ridge.predict(ridge.X_test)
            ridge_reg.append(ridge.MSE(z_tilde_test, ridge.z_test))

            lasso.fit_simple_model(deg)
            z_tilde_test = lasso.predict(lasso.X_test)
            lasso_reg.append(lasso.MSE(z_tilde_test, lasso.z_test))

            _, test = ols.MSE_cross_validation(deg)
            ols_cv.append(test)

            _, test = ridge.MSE_cross_validation(deg, ridge_lambda)
            ridge_cv.append(test)

            _, test = lasso.MSE_cross_validation(deg, lasso_lambda)
            lasso_cv.append(test)

        plt.plot(x_data[:max_bootstrap-1], ols_boot, label="OLS - Bootstrap", linestyle="--")
        plt.plot(x_data[:max_bootstrap-1], ridge_boot, label="Ridge - Bootstrap", linestyle="--")
        plt.plot(x_data[:max_bootstrap-1], lasso_boot, label="Lasso - Bootstrap", linestyle="--")

        plt.plot(x_data, ols_reg, label="OLS - Regular", linestyle="-.")
        plt.plot(x_data, ridge_reg, label="Ridge - Regular", linestyle="-.")
        plt.plot(x_data, lasso_reg, label="Lasso - Regular", linestyle="-.")

        plt.plot(x_data, ols_cv, label="OLS - Cross-Val")
        plt.plot(x_data, ridge_cv, label="Ridge - Cross-Val")
        plt.plot(x_data, lasso_cv, label="Lasso - Cross-Val")

        mse_results = [
            ols_boot,
            ridge_boot,
            lasso_boot,
            ols_reg,
            ridge_reg,
            lasso_reg,
            ols_cv,
            ridge_cv,
            lasso_cv,
        ]
        min_mse = min(min(mse_list) for mse_list in mse_results)
        for mse_list in mse_results:
            if min_mse in mse_list:
                min_index = mse_list.index(min_mse)
                min_mse_deg = x_data[min_index]
                break

        plt.plot(
            min_mse_deg,
            min_mse,
            "ro",
            label=f"Min MSE: {min_mse:.4f} at deg: {min_mse_deg}",
        )

        plt.title(f"MSE")
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")

    def plot_all_boot(
        self, ridge_lambda: float, lasso_lambda: float, max_degree: int, n_samples: int
    ):
        self.__config()
        x_data = list(range(1, max_degree + 1))

        ols_boot = []
        ridge_boot = []
        lasso_boot = []

        ols = OLSModel(self.handler)
        ridge = RidgeModel(self.handler)
        lasso = LassoModel(self.handler)

        for deg in x_data:
            ols_boot.append(
                ols.MSE_bootstrap(
                    n_samples=n_samples,
                    degree=deg,
                )
            )
            ridge_boot.append(
                ridge.MSE_bootstrap(
                    n_samples=n_samples,
                    degree=deg,
                    ridge_lambda=ridge_lambda,
                )
            )
            lasso_boot.append(
                lasso.MSE_bootstrap(
                    n_samples=n_samples,
                    degree=deg,
                    lasso_lambda=lasso_lambda,
                )
            )

        plt.plot(x_data, ols_boot, label="OLS - Bootstrap")
        plt.plot(x_data, ridge_boot, label="Ridge - Bootstrap")
        plt.plot(x_data, lasso_boot, label="Lasso - Bootstrap")

        plt.title(f"MSE")
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")

    def plot_all_cv(
        self, ridge_lambda: float, lasso_lambda: float, max_degree: int, k_folds: int
    ):
        self.__config()
        x_data = list(range(1, max_degree + 1))
        ols_cv = []
        ridge_cv = []
        lasso_cv = []

        ols = OLSModel(self.handler)
        ridge = RidgeModel(self.handler)
        lasso = LassoModel(self.handler)

        ols.make_cross_val_split(kfolds=k_folds)
        ridge.make_cross_val_split(kfolds=k_folds)
        lasso.make_cross_val_split(kfolds=k_folds)

        for deg in x_data:

            _, test = ols.MSE_cross_validation(deg)
            ols_cv.append(test)

            _, test = ridge.MSE_cross_validation(deg, ridge_lambda)
            ridge_cv.append(test)

            _, test = lasso.MSE_cross_validation(deg, lasso_lambda)
            lasso_cv.append(test)

        plt.plot(x_data, ols_cv, label="OLS - Cross-Val")
        plt.plot(x_data, ridge_cv, label="Ridge - Cross-Val")
        plt.plot(x_data, lasso_cv, label="Lasso - Cross-Val")

        plt.title(f"MSE")
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")

    def plot_all_ols(
        self,
        max_degree: int,
    ):
        self.__config()
        x_data = list(range(1, max_degree + 1))
        ols_reg = []
        ridge_reg = []
        lasso_reg = []

        ols = OLSModel(self.handler)
        ridge = RidgeModel(self.handler)
        lasso = LassoModel(self.handler)

        for deg in x_data:

            ols.fit_simple_model(deg)
            z_tilde_test = ols.predict(ols.X_test)
            ols_reg.append(ols.MSE(z_tilde_test, ols.z_test))

            ridge.fit_simple_model(deg)
            z_tilde_test = ridge.predict(ridge.X_test)
            ridge_reg.append(ridge.MSE(z_tilde_test, ridge.z_test))

            lasso.fit_simple_model(deg)
            z_tilde_test = lasso.predict(lasso.X_test)
            lasso_reg.append(lasso.MSE(z_tilde_test, lasso.z_test))

        plt.plot(x_data, ols_reg, label="OLS - Regular")
        plt.plot(x_data, ridge_reg, label="Ridge - Regular")
        plt.plot(x_data, lasso_reg, label="Lasso - Regular")

        plt.title(f"MSE")
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")

    def plot_cv_bs_ols(self, max_degree: int, n_samples: int, k_folds: int):
        self.__config()
        x_data = list(range(1, max_degree + 1))
        ols_reg = []
        ols_boot = []
        ols_cv = []

        ols = OLSModel(self.handler)

        ols.make_cross_val_split(kfolds=k_folds)

        for deg in x_data:
            ols_boot.append(
                ols.MSE_bootstrap(
                    n_samples=n_samples,
                    degree=deg,
                )
            )

            ols.fit_simple_model(deg)
            z_tilde_test = ols.predict(ols.X_test)
            ols_reg.append(ols.MSE(z_tilde_test, ols.z_test))

            _, test = ols.MSE_cross_validation(deg)
            ols_cv.append(test)

        plt.plot(x_data, ols_boot, label="OLS - Bootstrap", linestyle="--")
        plt.plot(x_data, ols_cv, label="OLS - Cross-Val")

        mse_results = [ols_boot, ols_cv]
        min_mse = min(min(mse_list) for mse_list in mse_results)
        for mse_list in mse_results:
            if min_mse in mse_list:
                min_index = mse_list.index(min_mse)
                min_mse_deg = x_data[min_index]
                break

        plt.plot(
            min_mse_deg,
            min_mse,
            "ro",
            label=f"Min MSE: {min_mse:.4f} at deg: {min_mse_deg}",
        )

        plt.title(f"MSE")
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")

    def plot_all_no_resampling(
        self,
        max_degree: int,
        ridge_lambda: float = 0,
        lasso_lambda: float = 0
    ):
        self.__config()
        x_data = list(range(1, max_degree + 1))
        ols_reg = []
        ridge_reg = []
        lasso_reg = []

        ols = OLSModel(self.handler)
        ridge = RidgeModel(self.handler)
        lasso = LassoModel(self.handler)

        for deg in x_data:

            ols.fit_simple_model(deg)
            z_tilde_test = ols.predict(ols.X_test)
            ols_reg.append(ols.MSE(z_tilde_test, ols.z_test))

            ridge.fit_simple_model(deg, ridge_lambda)
            z_tilde_test = ridge.predict(ridge.X_test)
            ridge_reg.append(ridge.MSE(z_tilde_test, ridge.z_test))

            lasso.fit_simple_model(deg, lasso_lambda)
            z_tilde_test = lasso.predict(lasso.X_test)
            lasso_reg.append(lasso.MSE(z_tilde_test, lasso.z_test))

        plt.plot(x_data, ols_reg, label="OLS")
        plt.plot(x_data, ridge_reg, label="Ridge", linestyle="--")
        plt.plot(x_data, lasso_reg, label="Lasso", linestyle="-.")

        mse_results = [ols_reg, ridge_reg, lasso_reg]
        min_mse = min(min(mse_list) for mse_list in mse_results)
        for mse_list in mse_results:
            if min_mse in mse_list:
                min_index = mse_list.index(min_mse)
                min_mse_deg = x_data[min_index]
                break

        plt.plot(
            min_mse_deg,
            min_mse,
            "ro",
            label=f"Min MSE: {min_mse:.4f} at deg: {min_mse_deg}",
        )

        plt.title(f"MSE")
        plt.legend()
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")
