import numpy as np
import kagglehub
import pandas as pd


import os
os.environ['KAGGLE_USERNAME'] = "miamer"
os.environ['KAGGLE_KEY'] = "7b87a5fea36501c8f5ca5a35d5e88c8d"

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()


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
        self.a = 3
        self.b = -2
        self.c = 5
        noise = 0.2

        self.y = (
            self.a
            + self.b * self.x
            + self.c * self.x**2 
            + noise * np.random.randn(self.data_points, 1)
        )

class CancerData(DataGen):
    def __init__(self, data_points: int = 569) -> None:
        super().__init__(data_points)
        self.__get_data()

    def __get_data(self):
        api.dataset_download_files('uciml/breast-cancer-wisconsin-data', path='data', unzip=True)

        data = pd.read_csv("data/data.csv")
        self.x = data.drop("diagnosis", axis=1)
        self.y = data["diagnosis"].map({'B': 0, 'M': 1})


if __name__ == "__main__":
    print(Poly1D2Deg(6).y)
