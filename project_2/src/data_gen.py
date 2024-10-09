import numpy as np

class DataGen():
    def __init__(self, data_points: int) -> None:
        self.data_points = data_points


class Poly1D3Deg(DataGen):
    def __init__(self, data_points: int) -> None:
        self.x = np.linspace(0, 1, data_points)
        super().__init__(data_points)
        self.__calc_y()


    def __calc_y(self):
        self.y = 3 + 2*self.x + 5*self.x**2



if __name__ == "__main__":
    print(Poly1D3Deg(6).y)