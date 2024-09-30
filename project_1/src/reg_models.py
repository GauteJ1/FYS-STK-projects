from data_handling import DataHandler
from data_gen import FrankeDataGen
import numpy as np


class RegModel:
    def __init__(self, data: DataHandler) -> None:
        self.data = data

    def get_preprocessed_data(self, degree: int):
        X_train, X_test, z_train, z_test = self.data.preprocess(
            degree=degree, scaling="Mean", test_size=0.25
        )
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

    def fit_model_on_data(
        self, X_train, z_train, ridge_lambda: float = 0, lasso_lambda: float = 0
    ):
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
        return opt_beta

    def fit_model(self, degree: int, ridge_lambda: float = 0, lasso_lambda: float = 0):
        self.get_preprocessed_data(degree=degree)
        self.fit_model_on_data(
            self.X_train,
            self.z_train,
        )

    def MSE(self, train: bool = False) -> float:
        if train:
            z_train_tilde = self.X_train @ self.opt_beta + self.intercept
            MSE = np.mean((self.z_train - z_train_tilde)**2)

        else:
            z_test_tilde = self.X_test @ self.opt_beta + self.intercept
            MSE = np.mean((self.z_test - z_test_tilde)**2)

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
            R2 = 1 - ((self.z_test - z_test_tilde).T @ (self.z_test - z_test_tilde)) / (
                (self.z_test - np.mean(self.z_test)).T
                @ (self.z_test - np.mean(self.z_test))
            )

        return R2[0, 0]

    def bootstrap(self, degree: int, samples: int = 100):
        X_train = self.X_train[:, : ((degree + 1) * (degree + 2)) // 2 - 1]
        X_test = self.X_test[:, : ((degree + 1) * (degree + 2)) // 2 - 1]
        z_pred = np.zeros((X_test.shape[0], samples))
        for i in range(samples):
            X_bootstrap, z_bootstrap = self.data.create_bootstrap_sampling(
                X_train, self.z_train
            )
            opt_beta = self.fit_model_on_data(X_bootstrap, z_bootstrap)
            pred = X_test @ opt_beta
            z_pred[:, i] = pred.ravel()

        mse = np.mean(
            np.mean(
                (self.z_test - z_pred) ** 2,
                axis=1,
                keepdims=True,
            )
        )
        bias = np.mean((self.z_test - np.mean(z_pred, axis=1, keepdims=True)) ** 2)
        variance = np.mean(np.var(z_pred, axis=1, keepdims=True))

        return mse, bias, variance

    def bootstrap_mult_degs(
        self, min_deg: int = 1, max_deg: int = 12, samples: int = 100
    ):
        deg_list = []
        error_list = []
        bias_list = []
        variance_list = []

        self.get_preprocessed_data(degree=max_deg)

        for deg in range(min_deg, max_deg + 1):
            deg_list.append(deg)
            err, bias, var = self.bootstrap(deg, samples)
            error_list.append(err)
            bias_list.append(bias)
            variance_list.append(var)

        return deg_list, error_list, bias_list, variance_list

    def cross_validation(
        self, kfolds: int, degree: int, ridge_lambda: float = 0, lasso_lambda: float = 0
    ):
        cv_data = self.data.create_cross_validation(degree=degree, kfolds=kfolds)

        mse_train = []
        mse_test = []
        for i, (X_train, X_test, z_train, z_test) in enumerate(cv_data):
            X_bootstrap, z_bootstrap = self.data.create_bootstrap_sampling(
                X_train, self.z_train
            )
            opt_beta = self.fit_model_on_data(
                X_bootstrap, z_bootstrap, ridge_lambda, lasso_lambda
            )
            pred_train = X_train @ opt_beta
            pred_test = X_test @ opt_beta

            mse_train.append(np.mean((z_train - pred_train) ** 2))
            mse_test.append(np.mean((z_test - pred_test) ** 2))

        return mse_train, mse_test

    def cv_mult_degs(self, min_deg: int = 1, max_deg: int = 12, kfolds: int = 5):
        deg_list = []
        err_list = []

        self.get_preprocessed_data(degree=max_deg)

        for deg in range(min_deg, max_deg + 1):
            deg_list.append(deg)
            err = self.cross_validation(deg, kfolds)
            err_list.append(err)

        return deg_list, err


class OLSModel(RegModel):
    def __init__(self, data: DataHandler) -> None:
        super().__init__(data)

    def fit_model(self, degree: int):
        self.get_preprocessed_data(degree=degree)
        self.fit_model_on_data(
            self.X_train,
            self.z_train,
        )

    def fit_model_on_data(
        self, X_train, z_train
    ):
        return super().fit_model_on_data(X_train, z_train)

    def cross_validation(self, kfolds: int, degree: int):
        return super().cross_validation(kfolds, degree, ridge_lambda=0, lasso_lambda=0)


class RidgeModel(RegModel):
    def __init__(self, data: DataHandler) -> None:
        super().__init__(data)

    def fit_model(self, degree: int, lmbda: float):
        self.get_preprocessed_data(degree=degree)
        self.fit_model_on_data(
            self.X_train,
            self.z_train,
            lmbda=lmbda
        )

    def cross_validation(self, kfolds: int, degree: int, lmbda: float):
        return super().cross_validation(
            kfolds, degree, ridge_lambda=lmbda, lasso_lambda=0
        )
    
    def fit_model_on_data(
        self, X_train, z_train, lmbda: float
    ):
        return super().fit_model_on_data(X_train, z_train, ridge_lambda=lmbda)


class LassoModel(RegModel):
    def __init__(self, data: DataHandler) -> None:
        super().__init__(data)

    def fit_model(self, degree: int, lmbda: float):
        self.get_preprocessed_data(degree=degree)
        self.fit_model_on_data(
            self.X_train,
            self.z_train,
            lmbda=lmbda
        )

    def cross_validation(self, kfolds: int, degree: int, lmbda: float):
        return super().cross_validation(
            kfolds, degree, ridge_lambda=0, lasso_lambda=lmbda
        )
    
    def fit_model_on_data(
        self, X_train, z_train, lmbda: float
    ):
        return super().fit_model_on_data(X_train, z_train, lasso_lambda=lmbda)


if __name__ == "__main__":
    data = FrankeDataGen(101)
    handler = DataHandler(data)
    ols = OLSModel(handler)
    ols.fit_model(5)
    print(f"Train MSE: {ols.MSE(True)}.")
    print(f"Test MSE: {ols.MSE(False)}.")
    print(f"Train R2: {ols.R2(True)}.")
    print(f"Test R2: {ols.R2(False)}.")

    err, bias, var = ols.bootstrap_deg(2)
    print(f"MSE: {err}")
    print(f"Bias: {bias}")
    print(f"Variance: {var}")
