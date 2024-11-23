import autograd.numpy as np

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
        super().__init__(Nx,Nt)
        self.L = L
        self.T = T
        self.__generate_data()

    def __generate_data(self) -> None:
        self.x = np.linspace(0, self.L, self.Nx).reshape(-1, 1)
        self.t = np.linspace(0, self.T, self.Nt).reshape(-1, 1)
        

