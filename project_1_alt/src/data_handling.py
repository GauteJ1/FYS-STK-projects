from data_gen import DataGen, FrankeDataGen
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np


class DataHandler:
    def __init__(self, data: DataGen, test_size: float = 0.2) -> None:
        # Ravel x and y data
        self.x = data.x.ravel()
        self.y = data.y.ravel()

        # Reshape z data
        n = len(self.x)
        self.z = data.get_data().ravel().reshape(n, 1)

        # Make train/test split
        self.test_size = test_size
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

    def __scale(self, X_train, X_test):
        # Scale data
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save the scaler in the class
        self.scaler = scaler

        return X_train_scaled, X_test_scaled

    def preprocess(
        self, degree: int = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train = self.__make_X(self.x_train, self.y_train, degree)
        X_test = self.__make_X(self.x_test, self.y_test, degree)

        X_train, X_test = self.__scale(X_train, X_test)

        self.X_train = X_train
        self.X_test = X_test

        return X_train, X_test, self.z_train, self.z_test

    def create_cross_validation(self, kfolds: int = 5):
        x = self.x
        y = self.y
        z = self.z

        kf = KFold(n_splits=kfolds, shuffle=True)

        data_splits = []
        splits = kf.split(x, z)
        for train, test in splits:
            data_splits.append(
                (
                    x[train],
                    x[test],
                    y[train],
                    y[test],
                    z[train],
                    z[test],
                )
            )

        self.data_splits = data_splits
    
    def preprocess_cross_val(self, degree: int = 10):
        data_splits = []
        for x_train, x_test, y_train, y_test, z_train, z_test in self.data_splits:
            X_train = self.__make_X(x_train, y_train, degree)
            X_test = self.__make_X(x_test, y_test, degree)

            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            data_splits.append((X_train_scaled, X_test_scaled, z_train, z_test))

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
