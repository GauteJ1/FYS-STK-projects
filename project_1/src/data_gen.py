import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


class DataGen:
    def __init__(
        self, step_size: float
    ) -> None:  # Will probably have to set other start/end points for the axes as well
        x = np.arange(0, 1, step_size)
        y = np.arange(0, 1, step_size)
        x, y = np.meshgrid(x, y)
        self.x = x
        self.y = y

    def plot_data(
        self, z: np.ndarray[float], z_lim: tuple[float, float] = (-1, 1)
    ) -> None:
        x = self.x
        y = self.y

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the surface.
        surf = ax.plot_surface(
            x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )

        # Customize the z axis.
        ax.set_zlim(z_lim[0], z_lim[1])
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()


class FrankeDataGen(DataGen):
    def __init__(self, step_size: float) -> None:
        super().__init__(step_size)
        self.__generate_data()

    def __generate_data(self) -> None:
        term1 = 0.75 * np.exp(
            -(0.25 * (9 * self.x - 2) ** 2) - 0.25 * ((9 * self.y - 2) ** 2)
        )
        term2 = 0.75 * np.exp(-((9 * self.x + 1) ** 2) / 49.0 - 0.1 * (9 * self.y + 1))
        term3 = 0.5 * np.exp(
            -((9 * self.x - 7) ** 2) / 4.0 - 0.25 * ((9 * self.y - 3) ** 2)
        )
        term4 = -0.2 * np.exp(-((9 * self.x - 4) ** 2) - (9 * self.y - 7) ** 2)

        self.z = term1 + term2 + term3 + term4

    def get_data(self) -> np.ndarray[float]:
        return self.z

    def plot_data(self) -> None:
        z = self.z
        z_lim = (-0.10, 1.40)
        super().plot_data(z, z_lim)


class TerrainDataGen(DataGen):
    def __init__(self, step_size: float) -> None:
        super().__init__(step_size)
        self.__generate_data()

    def __generate_data(self) -> None:
        pass  # TODO: Implement generation of terrain data

    def get_data(self) -> np.ndarray[float]:
        return self.z

    def plot_data(self) -> None:
        z = self.z
        z_lim = (-10, 10)  # TODO: set terrain data z_lim if needed
        super().plot_data(z, z_lim)
