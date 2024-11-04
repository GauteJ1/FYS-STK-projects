import numpy as np
import kagglehub
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import os
os.environ['KAGGLE_USERNAME'] = "miamer"
os.environ['KAGGLE_KEY'] = "7b87a5fea36501c8f5ca5a35d5e88c8d"

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

np.random.seed(4155)


class DataGen: 
    def __init__(self, data_points: int) -> None:
        self.data_points = data_points


class Poly1D2Deg(DataGen):
    def __init__(self, data_points: int) -> None:
        self.x = np.random.rand(data_points, 1)
        super().__init__(data_points)
        self.__calc_y()

    def __calc_y(self):
        # These should be a list (due to generalization)
        a = 3
        b = -2
        c = 5
        noise = 0.2

        self.y = (
            a + b * self.x + c * self.x**2 
            + noise * np.random.randn(self.data_points, 1)
        )

class FrankeDataGen(DataGen):
    def __init__(self, data_points: int = 101, noise: bool = False) -> None:
        super().__init__(data_points)
        self.noise = noise
        self.__generate_data()

    def __generate_data(self) -> None:
        x = np.linspace(0, 1, self.data_points)
        y = np.linspace(0, 1, self.data_points)
        self.x, self.y = np.meshgrid(x, y)  # Create a 2D grid for x and y

        term1 = 0.75 * np.exp(
            -(0.25 * (9 * self.x - 2) ** 2) - 0.25 * ((9 * self.y - 2) ** 2)
        )
        term2 = 0.75 * np.exp(-((9 * self.x + 1) ** 2) / 49.0 - 0.1 * (9 * self.y + 1))
        term3 = 0.5 * np.exp(
            -((9 * self.x - 7) ** 2) / 4.0 - 0.25 * ((9 * self.y - 3) ** 2)
        )
        term4 = -0.2 * np.exp(-((9 * self.x - 4) ** 2) - (9 * self.y - 7) ** 2)

        if self.noise:
            self.z = (
                term1
                + term2
                + term3
                + term4
                + 0.3 * np.random.normal(0, 1, self.x.shape)
            )
        else:
            self.z = term1 + term2 + term3 + term4

class CancerData(DataGen):
    def __init__(self, data_points: int = 569) -> None:
        super().__init__(data_points)
        self.__get_data()

    def __get_data(self):
        api.dataset_download_files('uciml/breast-cancer-wisconsin-data', path='../data', unzip=True)

        data = pd.read_csv("../data/data.csv")
        self.x = data.drop(["diagnosis", "id", "Unnamed: 32"], axis=1)
        self.y = data["diagnosis"].map({'B': 0, 'M': 1})

        self.scale_data()

    def scale_data(self):
        scaler = MinMaxScaler()
        self.x = scaler.fit_transform(self.x)


if __name__ == "__main__":
    print(Poly1D2Deg(6).y)
