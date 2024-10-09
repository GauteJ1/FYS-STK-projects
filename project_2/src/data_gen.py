import numpy as np


class DataGen:
    def __init__(self, data_points: int) -> None:
        self.data_points = data_points


class Poly1D2Deg(DataGen):
    def __init__(self, data_points: int) -> None:
        self.x = np.random.rand(data_points, 1)
        super().__init__(data_points)
        self.__calc_y()

    def __calc_y(self):
        self.y = (
            3
            + 2 * self.x
            + 5 * self.x**2  # + 0.2 * np.random.randn(self.data_points, 1)
        )


if __name__ == "__main__":
    print(Poly1D2Deg(6).y)
