import numpy as np
import torch 

np.random.seed(4155)  # for FYS-STK4155


class DataGen:
    """
    Base class for data generation
    """

    def __init__(self, Nx: int, Nt: int) -> None:
        self.Nx = Nx
        self.Nt = Nt


class RodDataGen(DataGen):
    """
    Class for generating data for the rod problem
    """

    def __init__(self, Nx: int, Nt: int, L: float = 1.0, T: float = 1.0) -> None:
        super().__init__(Nx, Nt)
        self.L = L
        self.T = T
        self.__generate_data()

    def __generate_data(self) -> None:
        self.x = torch.linspace(0, self.L, self.Nx) 
        self.t = torch.linspace(0, self.T, self.Nt) 

        self.xx, self.tt = torch.meshgrid(self.x, self.t, indexing="ij")

        self.x = self.xx.flatten().reshape(-1, 1)
        self.t = self.tt.flatten().reshape(-1, 1)

