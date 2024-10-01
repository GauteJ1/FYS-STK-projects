import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


class DataGen:
    def __init__(
        self, data_points: int = 101
    ) -> None:  # Will probably have to set other start/end points for the axes as well
        self.data_points = data_points
        x = np.linspace(0, 1, data_points)
        y = np.linspace(0, 1, data_points)
        self.x, self.y = np.meshgrid(x, y)

    def plot_data(
        self,
        z: np.ndarray[float],
        z_lim: tuple[float, float] = (-1, 1),
        title: str = "Data",
        save_path: str = "None",
    ) -> None:
        x = self.x
        y = self.y

        plt.style.use("plot_settings.mplstyle")

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

        ax.set_title(title)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.tight_layout()
        if save_path != "None":
            plt.savefig(save_path)
        plt.show()

    def get_data(self) -> np.ndarray[float]:
        raise NotImplementedError


class FrankeDataGen(DataGen):
    def __init__(self, data_points: int = 101, noise: bool = False) -> None:
        super().__init__(data_points)
        self.noise = noise
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

        if self.noise:
            self.z = (
                term1
                + term2
                + term3
                + term4
                + 0.3 * np.random.normal(0, 1, (term1.shape))
            )

        else:
            self.z = term1 + term2 + term3 + term4

    def get_data(self) -> np.ndarray[float]:
        self.__generate_data()
        return self.z

    def plot_data(self, save_path: str = "None") -> None:
        z = self.z
        z_lim = (-0.10, 1.40)
        if self.noise:
            super().plot_data(z, z_lim, "Franke Function with noise", save_path)
        else:
            super().plot_data(z, z_lim, "Franke Function", save_path)


class TerrainDataGen(DataGen):
    def __init__(self, data_points: int = 101) -> None:
        super().__init__(data_points)
        self.__generate_data()

    def __generate_data(self) -> None:
        pass  # TODO: Implement generation of terrain data

    def get_data(self) -> np.ndarray[float]:
        return self.z

    def plot_data(self, save_path: str = "None") -> None:
        z = self.z
        z_lim = (-10, 10)  # TODO: set terrain data z_lim if needed
        super().plot_data(z, z_lim, "Terrain", save_path)


class SimpleTest(DataGen):
    def __init__(self, data_points: int = 101) -> None:
        super().__init__(data_points)
        self.__generate_data()

    def __generate_data(self) -> None:
        self.z = (
            2 * self.x**2
            - 4 * self.x
            + 4
            + 0.2 * np.random.normal(0, 1, (self.x.shape))
        )

    def get_data(self) -> np.ndarray[float]:
        return self.z

    def plot_data(self, save_path: str = "None") -> None:
        z = self.z
        z_lim = (2, 5)
        super().plot_data(z, z_lim, "Simple", save_path)


if __name__ == "__main__":
    franke = FrankeDataGen(data_points=101)
    franke.plot_data()

    terrain = TerrainDataGen(data_points=101)
    terrain.plot_data()
