import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from tqdm import tqdm


def exact_sol(x, t):
    # Probably move to separate file, as it will be used other places
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


class FiniteDifference:
    def __init__(self, N: int, dt: float | None = None) -> None:
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
        raise RuntimeError

    def D2(self):
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), "lil")
        return D2

    def set_init(self, init):
        self.init = init

    def apply_bcs(self, u):
        # Apply Dirichlet boundary conidtions
        u[0] = 0
        u[-1] = 0

        return u

    def __call__(self):
        raise RuntimeError

    def plot_heatmap(self, data):
        X = np.linspace(0, 1, self.N + 1)
        Y = data["time_steps"]
        X, Y = np.meshgrid(X, Y)
        Z = data["values"]

        plt.contourf(X, Y, Z, cmap="hot", levels=500, vmin=0, vmax=1)
        
        plt.colorbar()
        plt.savefig("../plots/heat_map_finite_difference.png")
        plt.show()

    def total_mse(self, data, exact):
        computed = data["values"]

        xs = np.linspace(0, 1, self.N+1)
        ts = data["time_steps"]
        xs, ts = np.meshgrid(xs, ts)
        expected = exact(xs, ts)

        print(np.allclose(computed, expected, 1e-4))

        return np.mean((computed-expected)**2)

    def mse_at_time_step(self, data, time_step, exact):
        computed = data["values"][time_step]

        xs = np.linspace(0, 1, self.N+1)
        t = data["time_steps"][time_step]
        expected = exact(xs, t)

        return t, np.mean((computed-expected)**2)
        


class ForwardEuler(FiniteDifference):
    def make_dt(self):
        """
        Use the largest dt possible within the stability criterion.
        """
        return self.dx**2 / 2

    def __call__(self, end_time: int):
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
    print(model.mse_at_time_step(data, 5, exact_sol))
    print(model.mse_at_time_step(data, 20, exact_sol))
