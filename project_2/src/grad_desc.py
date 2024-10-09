import numpy as np
from tqdm import tqdm
from data_gen import DataGen, Poly1D2Deg
from learn_rate import Update_Beta

# We will have to revisit structuring, this i just an initial idea based on the first couple of tasks in the weekly assignment
# Possibly, the main class should not be OLS/Ridge, and they would rather be possible function options for the gradient or something

# Have not tuned the learning rate yet

class Model: 

    def __init__(self, data: DataGen, model_type: str) -> None:
        self.x = data.x
        self.y = data.y
        self.n = data.data_points
        self.model_type = model_type

        self.a = data.a
        self.b = data.b
        self.c = data.c


    def makeX(self, deg: int):
        # Add dimension 2 option to this if we want to use Franke/Terrain

        # Design matrix including the intercept
        # No scaling of data of and all data used for training (for now)
        X = np.zeros((self.n, deg))
        for d in range(deg):
            X[:,d] = self.x[:,0]**d
        return X
    
    def set_update(self, tpe, eta, gamma):
        update = Update_Beta()
        if tpe == "Constant":
            update.constant(eta)
        elif tpe == "Momentum":
            update.momentum_based(eta, gamma)
        elif tpe == "Adagrad":
            update.adagrad(eta)

        return update
    
    def gradient(self, X, y, beta, Lambda = 0.1):
        if self.model_type == "OLS":
            return 2.0/self.n*X.T @ (X @ beta-y)
        elif self.model_type == "Ridge":
            return 2.0/self.n*X.T @ (X @ beta-y)+2*Lambda*beta

    def gradient_descent(self, tpe: str = "Constant", eta: float = 0.05, gamma: float = 0.05, epochs: int = 10000, batch_size: int = 0):

        if batch_size == 0:
            batch_size = self.n

        self.epoch_list = []
        self.MSE_list = []
        self.beta = np.random.randn(3,1)

        X = self.makeX(3)

        # Learning rate and number of iterations
        update = self.set_update(tpe, eta, gamma)

        # Iterations (separate jit-compilable function?)
        iter = 0
        tolerance = 1e-5

        #pbar = tqdm(total = epochs * (self.n // batch_size))
        for epoch in range(epochs):

            prev_beta_ridge = self.beta.copy()
            
            indices = np.random.permutation(self.n)  # must check if it should be random or not, have not checked Morten's notes
            X_shuffled = X[indices]
            y_shuffled = self.y[indices]

            for i in range(0, self.n, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.beta = update(self.beta, self.gradient(X_batch, y_batch, self.beta))

                iter += 1
                #pbar.update(1)

            preds = X @ self.beta
            error = np.mean((self.y - preds)**2)
            self.MSE_list.append(error)
            self.epoch_list.append(epoch)
            
            if np.allclose(prev_beta_ridge, self.beta, tolerance):
                    #print(f'Converged after {epoch} epochs')
                    break
        
        #pbar.close()

        #print('Parameters for OLS using gradient descent')    
        #print(self.beta)
        #print(f'After {iter} iterations')