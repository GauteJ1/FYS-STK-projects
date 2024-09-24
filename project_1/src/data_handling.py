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

    def __make_X(self, degree: int) -> np.ndarray:
        X = np.zeros((len(self.x), ((degree + 1) * (degree + 2)) // 2 - 1))
        index = 0
        for i in range(degree):
            deg = i + 1
            for y_deg in range(0, deg + 1):
                X[:, index] = self.x ** (deg - y_deg) * self.y**y_deg
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
        X = self.__make_X(degree)

        X_train, X_test, z_train, z_test = train_test_split(
            X, self.z, test_size=test_size
        )

        X_train, X_test = self.__scale(X_train, X_test, scaling)

        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

        return X_train, X_test, z_train, z_test

    def create_bootstrap_sampling(self, X_train, z_train):

        X_bootstrap, z_bootstrap = resample(X_train, z_train)

        return X_bootstrap, z_bootstrap

    def create_cross_validation(self, degree: int, kfolds: int):
        z = self.z
        X = self.__make_X(degree)

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
