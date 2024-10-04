from data_handling import DataHandler
from data_gen import SimpleTest
import numpy as np
from sklearn.linear_model import Lasso


class RegModel:
    def __init__(self, data: DataHandler) -> None:
        self.data = data

    def get_preprocessed_data(self, degree: int):
        X_train, X_test, z_train, z_test = self.data.preprocess(degree=degree)
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

    def fit_model_on_data(self, X_train, z_train, lmbda: float = 0):
        opt_beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train

        self.intercept = np.mean(z_train)

        self.opt_beta = opt_beta
        return opt_beta

    def MSE(self, z_tilde, z) -> float:
        MSE = np.mean((z - z_tilde) ** 2)
        return MSE

    def R2(self, z_tilde, z) -> float:
        R2 = 1 - (np.sum((z - z_tilde) ** 2)) / (np.sum((z - np.mean(z)) ** 2))
        return R2

    def fit_simple_model(self, deg: int, ridge_lambda: int = 0, lasso_lambda: int = 0):
        self.get_preprocessed_data(deg)
        if ridge_lambda != 0:
            self.fit_model_on_data(self.X_train, self.z_train, ridge_lambda)
        else:
            self.fit_model_on_data(self.X_train, self.z_train, lasso_lambda)

    def predict(self, X):
        z_tilde = X @ self.opt_beta + self.intercept
        return z_tilde

    def bootstrap_index(self, N: int):
        bootstrap_sample = np.random.choice(list(range(N)), size=N, replace=True)
        return bootstrap_sample

    def bootstrap(
        self,
        n_samples: int = 100,
        degree: int = 10,
        ridge_lambda: float = 0,
        lasso_lambda: float = 0,
    ):
        self.get_preprocessed_data(degree=degree)
        indices = [self.bootstrap_index(len(self.z_train)) for _ in range(n_samples)]

        preds = []
        for ind in indices:
            X_ = self.X_train[ind]
            z_ = self.z_train[ind]

            if ridge_lambda != 0:
                self.fit_model_on_data(X_, z_, ridge_lambda)
            else:
                self.fit_model_on_data(X_, z_, lasso_lambda)
            preds.append(self.predict(self.X_test))

        return preds

    def MSE_bootstrap(
        self,
        n_samples: int = 100,
        degree: int = 10,
        ridge_lambda: float = 0,
        lasso_lambda: float = 0,
    ):
        preds = self.bootstrap(n_samples, degree, ridge_lambda, lasso_lambda)
        z_tilde = np.mean(preds, axis=0)
        MSE = self.MSE(z_tilde, self.z_test)

        return MSE

    def R2_bootstrap(
        self,
        n_samples: int = 100,
        degree: int = 10,
        ridge_lambda: float = 0,
        lasso_lambda: float = 0,
    ):
        preds = self.bootstrap(n_samples, degree, ridge_lambda, lasso_lambda)
        z_tilde = np.mean(preds, axis=0)
        R2 = self.R2(z_tilde, self.z_test)

        return R2

    def bias_var_bootstrap(self, n_samples: int = 100, degree: int = 10):
        preds = self.bootstrap(n_samples, degree)

        MSE = np.mean(
            np.mean(
                (self.z_test - preds) ** 2,
                axis=1,
                keepdims=True,
            )
        )
        bias = np.mean((self.z_test - np.mean(preds, axis=0, keepdims=True)) ** 2)
        variance = np.mean(np.var(preds, axis=1, keepdims=True))

        return MSE, bias, variance

    def make_cross_val_split(self, kfolds: int):
        # We have this as a sperate function so we can use the exact same
        # cross validation split multiple times when testing a model for
        # different compexities or lambdas
        self.data.create_cross_validation(kfolds=kfolds)

    def MSE_cross_validation(
        self, degree: int, ridge_lambda: float = 0, lasso_lambda: float = 0
    ):
        data_splits = self.data.preprocess_cross_val(degree=degree)

        MSE_train_list = []
        MSE_test_list = []
        for X_train, X_test, z_train, z_test in data_splits:
            if lasso_lambda > 0:
                self.fit_model_on_data(X_train, z_train, lasso_lambda)
            else:
                self.fit_model_on_data(X_train, z_train, ridge_lambda)
            z_train_tilde = self.predict(X_train)
            MSE_train_list.append(self.MSE(z_train_tilde, z_train))

            z_test_tilde = self.predict(X_test)
            MSE_test_list.append(self.MSE(z_test_tilde, z_test))

        MSE_train = np.mean(MSE_train_list)
        MSE_test = np.mean(MSE_test_list)

        return MSE_train, MSE_test


class OLSModel(RegModel):
    def fit_simple_model(self, deg: int, lmbda: float = 0):
        return super().fit_simple_model(deg, 0, 0)

    def fit_model_on_data(self, X, z, lmbda: float = 0):
        return super().fit_model_on_data(X, z, 0)


class RidgeModel(RegModel):
    def fit_simple_model(self, deg: int, lmbda: float = 0):
        return super().fit_simple_model(deg, lmbda, 0)

    def fit_model_on_data(self, X_train, z_train, lmbda: float = 0):
        opt_beta = (
            np.linalg.inv(X_train.T @ X_train + lmbda * np.eye(len(X_train[0])))
            @ X_train.T
            @ z_train
        )

        self.intercept = np.mean(z_train)

        self.opt_beta = opt_beta
        return opt_beta


class LassoModel(RegModel):
    def fit_simple_model(self, deg: int, lmbda: float = 0):
        return super().fit_simple_model(deg, 0, lmbda)

    def fit_model_on_data(self, X, z, lmbda: float = 0):
        model = Lasso(alpha=lmbda, max_iter=1000, fit_intercept=False)
        model.fit(X, z)

        self.model = model
        self.intercept = np.mean(z)

        self.opt_beta = np.array([[x] for x in model.coef_])
        return self.opt_beta

    def predict(self, X):
        z_tilde = self.model.predict(X) + self.intercept
        z_tilde = np.array([[z] for z in z_tilde])
        return z_tilde


if __name__ == "__main__":
    data = SimpleTest(data_points=21)
    handler = DataHandler(data, test_size=0.25)
    ols = OLSModel(handler)
    for deg in range(1, 12):
        ols.fit_simple_model(deg=deg)
        print(f"-------\nDeg: {deg}")
        print(f"Train MSE: {ols.MSE(ols.X_train, ols.z_train)}")
        print(f"Test MSE: {ols.MSE(ols.X_test, ols.z_test)}")
    ols.bootstrap()
