import numpy as np
from tqdm import tqdm
from data_gen import DataGen, Poly1D2Deg
from learn_rate import Update_Beta

# We will have to revisit structuring, this i just an initial idea based on the first couple of tasks in the weekly assignment
# Possibly, the main class should not be OLS/Ridge, and they would rather be possible function options for the gradient or something

# Have not tuned the learning rate yet

class OLS:
    def __init__(self, data: DataGen) -> None:
        self.x = data.x
        self.y = data.y
        self.n = data.data_points


    def makeX(self, deg: int):
        # Add dimension 2 option to this if we want to use Franke/Terrain

        # Design matrix including the intercept
        # No scaling of data of and all data used for training (for now)
        X = np.zeros((self.n, deg))
        for d in range(deg):
            X[:,d] = self.x[:,0]**d
        return X
    
    def set_update(self, type: str = "Constant", eta: float = 0, gamma: float = 0):
        update = Update_Beta()
        if type == "Constant":
            update.constant(eta)
        elif type == "Momentum":
            update.momentum_based(eta, gamma)

        return update

    def gradient_descent(self):
        X = self.makeX(3)

        # Learning rate and number of iterations
        eta = 0.05
        gamma = 0.05
        update = self.set_update("Momentum", eta, gamma)
        Niterations = 20000

        # Gradient descent with OLS
        beta_OLS = np.random.randn(3,1)
        prev_beta_OLS = np.zeros_like(beta_OLS)
        gradient = np.zeros(3)

        # Iterations (separate jit-compilable function?)
        iter = 0
        tolerance = 1e-5

        pbar = tqdm(total=Niterations)
        while iter < Niterations and not np.allclose(prev_beta_OLS, beta_OLS, tolerance):
            prev_beta_OLS = beta_OLS.copy()
            iter += 1
            pbar.update(1)

            gradient = (2.0/self.n)*X.T @ (X @ beta_OLS-self.y)
            beta_OLS = update(beta_OLS, gradient)
        
        pbar.close()

        print('Parameters for OLS using gradient descent')    
        print(beta_OLS)
        print(f'After {iter} iterations')


class Ridge(OLS):
    def gradient_descent(self):
        X = self.makeX(3)

        # Learning rate and number of iterations
        eta = 0.05
        Niterations = 20000

        #Ridge parameter Lambda
        Lambda  = 0.01

        # Gradient descent with  Ridge
        beta_Ridge = np.random.randn(3,1)
        prev_beta_ridge = np.zeros_like(beta_Ridge)
        gradient = np.zeros(3)

        # Iterations (separate jit-compilable function?)
        iter = 0
        tolerance = 1e-5

        pbar = tqdm(total=Niterations)
        while iter < Niterations and not np.allclose(prev_beta_ridge, beta_Ridge, tolerance):
            prev_beta_ridge = beta_Ridge.copy()
            iter += 1
            pbar.update(1)

            gradient = 2.0/self.n*X.T @ (X @ beta_Ridge-self.y)+2*Lambda*beta_Ridge
            beta_Ridge -= eta*gradient

        pbar.close()

        print('Parameters for Ridge using gradient descent')    
        print(beta_Ridge)
        print(f'After {iter} iterations')


if __name__ == "__main__":
    np.random.seed(42)

    data = Poly1D2Deg(100)

    ols_model = OLS(data)
    ols_model.gradient_descent()

    ridge_model = Ridge(data)
    ridge_model.gradient_descent()
