from data_handling import DataHandler
from data_gen import FrankeDataGen
import numpy as np


class RegModel:
    def __init__(self, data: DataHandler) -> None:
        self.data = data

    def fit_model(self, degree: int, ridge_lambda: float = 0, lasso_lambda: float = 0):
        X_train, X_test, z_train, z_test = self.data.preprocess(
            degree=degree, scaling="Mean", test_size=0.3
        )

        self.intercept = np.mean(z_train)
        opt_beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train

        ## Apply eventual Ridge and Lasso:
        if ridge_lambda != 0:
            opt_beta = opt_beta / (1 + ridge_lambda)

        if lasso_lambda != 0:
            for i, beta in enumerate(opt_beta):
                if beta > lasso_lambda / 2:
                    opt_beta[i] = beta - lasso_lambda / 2
                elif beta < -lasso_lambda / 2:
                    opt_beta[i] = beta + lasso_lambda / 2
                else:
                    opt_beta[i] = 0

        self.opt_beta = opt_beta

        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

    def MSE(self, train: bool = False) -> float:
        if train:
            n = len(self.X_train)
            z_train_tilde = self.X_train @ self.opt_beta + self.intercept
            MSE = (1 / n) * (
                (self.z_train - z_train_tilde).T @ (self.z_train - z_train_tilde)
            )

        else:
            n = len(self.X_test)
            z_test_tilde = self.X_test @ self.opt_beta + self.intercept
            MSE = (1 / n) * (
                (self.z_test - z_test_tilde).T @ (self.z_test - z_test_tilde)
            )

        return MSE

    def R2(self, train: bool = False) -> float:
        if train:
            n = len(self.X_train)
            z_train_tilde = self.X_train @ self.opt_beta + self.intercept
            R2 = 1 - (
                (self.z_train - z_train_tilde).T @ (self.z_train - z_train_tilde)
            ) / (
                (self.z_train - np.mean(self.z_train)).T
                @ (self.z_train - np.mean(self.z_train))
            )

        else:
            n = len(self.X_test)
            z_test_tilde = self.X_test @ self.opt_beta + self.intercept
            R2 = 1 - (
                (self.z_test - z_test_tilde).T @ (self.z_test - z_test_tilde)
            ) / (
                (self.z_test - np.mean(self.z_test)).T
                @ (self.z_test - np.mean(self.z_test))
            )

        return R2


class OLSModel(RegModel):
    def __init__(self, data: DataHandler) -> None:
        super().__init__(data)

    def fit_model(self, degree: int):
        super().fit_model(degree=degree)


class RidgeModel(RegModel):
    def __init__(self, data: DataHandler) -> None:
        super().__init__(data)

    def fit_model(self, degree: int, lmbda: float):
        super().fit_model(degree=degree, ridge_lambda=lmbda)


class LassoModel(RegModel):
    def __init__(self, data: DataHandler) -> None:
        super().__init__(data)

    def fit_model(self, degree: int, lmbda: float):
        super().fit_model(degree=degree, lasso_lambda=lmbda)


if __name__ == "__main__":
    data = FrankeDataGen(101)
    handler = DataHandler(data)
    ols = OLSModel(handler)
    ols.fit_model(5)
    print(f"Train MSE: {ols.MSE(True)}.")
    print(f"Test MSE: {ols.MSE(False)}.")
    print(f"Train R2: {ols.R2(True)}.")
    print(f"Test R2: {ols.R2(False)}.")