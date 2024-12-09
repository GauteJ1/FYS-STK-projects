import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from analytic import exact_sol


class FiniteDifference:
    def __init__(self, N: int, dt: float | None = None) -> None:

        """
        Parameters
        ----------
        N : int
            Number of spatial points
        dt : float, optional
            Time step, by default None  

        Raises
        ------
        Warning
            If the chosen values for N and dt are not within the stability criterion
        """

        self.N = N
        self.dx = 1 / N

        stable_dt = self.make_dt()
        if dt is None:
            dt = stable_dt
        elif dt > stable_dt:
            raise Warning(
                "Chosen values for N and dt are not within the stability criterion"
            )

        self.dt = dt
        self.c = dt / self.dx**2

    def make_dt(self):
        """
        Placeholder method for finding the dt that satisfies the stability criterion for the specific scheme
        Actual implementation should be done in the child class

        Raises
        ------
        RuntimeError
        """
        raise RuntimeError

    def D2(self):
        """
        Create the tridiagonal matrix with -2 on the diagonal and 1 on the off-diagonals

        Returns
        -------
        D2 : scipy.sparse.lil_matrix
            Tridiagonal matrix
        """

        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), "lil")
        return D2

    def set_init(self, init):
        """
        Set the initial condition

        Parameters
        ----------
        init : function
            Function for the initial condition
        """
        self.init = init

    def apply_bcs(self, u):
        """
        Apply Dirichlet boundary conditions

        Parameters
        ----------
        u : np.ndarray
            Solution at the current time step

        Returns
        -------
        u : np.ndarray
            Solution at the current time step with Dirichlet boundary conditions applied, i.e. u[0] = u[-1] = 0
        """

        u[0] = 0
        u[-1] = 0

        return u

    def __call__(self):
        """
        Placeholder method for solving the PDE
        Actual implementation should be done in the child class

        Raises
        ------
        RuntimeError
        """

        raise RuntimeError

    def plot_heatmap(self, data):

        """
        Plot heatmap of the solution

        Parameters
        ----------
        data : dict
            Dictionary containing the time steps and the solution
        """

        X = np.linspace(0, 1, self.N + 1)
        Y = data["time_steps"]
        X, Y = np.meshgrid(X, Y)
        Z = data["values"]

        plt.contourf(X, Y, Z, cmap="hot", levels=500, vmin=0, vmax=1)
        
        plt.colorbar()
        plt.show()

    def total_mse(self, data, exact):
        """
        Compute the total mean squared error of the solution compared to the exact solution

        Parameters
        ----------
        data : dict
            Dictionary containing the time steps and the solution
        exact : function
            Function for the exact solution

        Returns
        -------
        mse : float
            Total mean squared error
        """
        computed = data["values"]

        xs = np.linspace(0, 1, self.N+1)
        ts = data["time_steps"]
        xs, ts = np.meshgrid(xs, ts)
        expected = exact(xs, ts)

        mse = np.mean((computed-expected)**2)

        return mse

    def mse_at_time_step(self, data, time_step, exact):

        """
        Compute the mean squared error of the solution at a specific time step compared to the exact solution

        Parameters
        ----------
        data : dict
            Dictionary containing the time steps and the solution
        time_step : int
            Time step to compute the mean squared error
        exact : function
            Function for the exact solution

        Returns
        -------
        t : float
            Time at the specific time step
        mse : float
            Mean squared error at the specific time step
        """

        computed = data["values"][time_step]

        xs = np.linspace(0, 1, self.N+1)
        t = data["time_steps"][time_step]
        expected = exact(xs, t)

        mse = np.mean((computed-expected)**2)

        return t, mse
        


class ForwardEuler(FiniteDifference):
    
    def make_dt(self):
        """
        Use the largest dt possible within the stability criterion.

        Returns
        -------
        dt : float
            Largest dt possible within the stability criterion
        """

        dt = self.dx**2 / 2
        
        return dt

    def __call__(self, end_time: int):

        """
        Solve the heat equation using the Forward Euler scheme

        Parameters
        ----------
        end_time : int
            End time of the simulation

        Returns
        -------
        data : dict
            Dictionary containing the time steps and the solution
        """

        D = self.D2()
        cfl = self.c

        Nt = int(end_time / self.dt)

        # Initialize
        un = self.init(np.linspace(0, 1, self.N + 1))
        unp1 = np.zeros_like(un)
        values = [un.copy()]
        time_steps = [0]

        # Loop
        for t in tqdm(range(1, Nt + 1)):
            unp1[:] = un + cfl * (D @ un)
            unp1[:] = self.apply_bcs(unp1)

            un[:] = unp1

            values.append(un.copy())
            time_steps.append(t * self.dt)

        data = {"time_steps": time_steps, "values": values}
        self.data = data

        return data


if __name__ == "__main__":

    init = lambda x: np.sin(np.pi * x)
    model = ForwardEuler(10, 0.001)
    model.set_init(init)
    data = model(0.5)
    model.plot_heatmap(data)

    print(model.total_mse(data, exact_sol))