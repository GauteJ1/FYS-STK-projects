from data_gen import DataGen, FrankeDataGen
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
import numpy as np


class DataHandler:
    def __init__(self, data: DataGen) -> None:
        self.x = data.x.ravel()
        n = len(self.x)
        self.y = data.y.ravel()
        self.z = data.get_data().ravel().reshape(n, 1)
        self.test_size = 0.2
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.z_train,
            self.z_test,
        ) = train_test_split(self.x, self.y, self.z, test_size=self.test_size)

    def __make_X(self, x, y, degree: int) -> np.ndarray:
        X = np.zeros((len(x), ((degree + 1) * (degree + 2)) // 2 - 1))
        index = 0
        for i in range(degree):
            deg = i + 1
            for y_deg in range(0, deg + 1):
                X[:, index] = x ** (deg - y_deg) * y**y_deg
                index += 1
        return X

    def __scale(self, X_train, X_test, scaling: str = "None"):
        if scaling == "Mean":
            means = [np.mean(X_train[:, i]) for i in range(len(X_train[0]))]
            for i in range(len(X_train[0])):
                X_train[:, i] -= means[i]
                X_test[:, i] -= means[i]
            return X_train, X_test
        else:
            return X_train, X_test

    def preprocess(
        self, test_size: float = 0.2, scaling: str = "None", degree: int = 2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if test_size != self.test_size:
            self.test_size = test_size
            (
                self.x_train,
                self.x_test,
                self.y_train,
                self.y_test,
                self.z_train,
                self.z_test,
            ) = train_test_split(self.x, self.y, self.z, test_size=self.test_size)

        X_train = self.__make_X(self.x_train, self.y_train, degree)
        X_test = self.__make_X(self.x_test, self.y_test, degree)

        X_train, X_test = self.__scale(X_train, X_test, scaling)

        self.X_train = X_train
        self.X_test = X_test

        return X_train, X_test, self.z_train, self.z_test

    def create_bootstrap_sampling(self, X_train, z_train):

        X_bootstrap, z_bootstrap = resample(X_train, z_train)

        return X_bootstrap, z_bootstrap

    def create_cross_validation(self, degree: int, kfolds: int = 5):
        z = self.z
        X = self.__make_X(self.x, self.y, degree)

        kf = KFold(n_splits=kfolds, shuffle=True)

        data_splits = []
        splits = kf.split(X, z)
        for train, test in splits:
            data_splits.append(
                (
                    X[train],
                    X[test],
                    z[train],
                    z[test],
                )
            )

        return data_splits

class DataHandler1D:
    def __init__(self, data: DataGen) -> None:
        self.x = data.x.ravel()
        n = len(self.x)
        self.z = data.get_data().ravel().reshape(n, 1)
        self.test_size = 0.2
        (
            self.x_train,
            self.x_test,
            self.z_train,
            self.z_test,
        ) = train_test_split(self.x, self.z, test_size=self.test_size)

    def __make_X(self, x, degree: int) -> np.ndarray:
        X = np.zeros((len(x), degree))
        for i in range(degree):
            deg = i + 1
            X[:, i] = x ** deg
        return X

    def __scale(self, X_train, X_test, scaling: str = "None"):
        if scaling == "Mean":
            means = [np.mean(X_train[:, i]) for i in range(len(X_train[0]))]
            for i in range(len(X_train[0])):
                X_train[:, i] -= means[i]
                X_test[:, i] -= means[i]
            return X_train, X_test
        else:
            return X_train, X_test

    def preprocess(
        self, test_size: float = 0.2, scaling: str = "None", degree: int = 2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if test_size != self.test_size:
            self.test_size = test_size
            (
                self.x_train,
                self.x_test,
                self.z_train,
                self.z_test,
            ) = train_test_split(self.x, self.z, test_size=self.test_size)

        X_train = self.__make_X(self.x_train, degree)
        X_test = self.__make_X(self.x_test, degree)

        X_train, X_test = self.__scale(X_train, X_test, scaling)

        self.X_train = X_train
        self.X_test = X_test

        return X_train, X_test, self.z_train, self.z_test

    def create_bootstrap_sampling(self, X_train, z_train):

        X_bootstrap, z_bootstrap = resample(X_train, z_train)

        return X_bootstrap, z_bootstrap

    def create_cross_validation(self, degree: int, kfolds: int = 5):
        z = self.z
        X = self.__make_X(self.x, self.y, degree)

        kf = KFold(n_splits=kfolds, shuffle=True)

        data_splits = []
        splits = kf.split(X, z)
        for train, test in splits:
            data_splits.append(
                (
                    X[train],
                    X[test],
                    z[train],
                    z[test],
                )
            )

        return data_splits

if __name__ == "__main__":
    data = FrankeDataGen(10)
    handler = DataHandler(data)

    ### Test preprocess:
    handler.preprocess(test_size=0.2, degree=2, scaling="Mean")
    assert len(handler.X_train[0]) == 5
    assert len(handler.X_train) == 80
    assert np.mean(handler.X_train[0:, 1]) < 1e-9  # Should be scaled by mean

    handler.create_cross_validation(degree=2, kfolds=4)
