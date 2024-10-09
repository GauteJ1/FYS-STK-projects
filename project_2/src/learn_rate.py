import numpy as np

class Update_Beta():
    def __init__(self) -> None:
        self.rate_type = "" 

    def constant(self, eta: float):
        self.eta = eta
        self.rate_type = "Constant"

    def momentum_based(self, eta: float, gamma: float):
        self.eta = eta
        self.gamma = gamma
        self.rate_type = "Momentum"
        self.prev_v = None

    def adagrad(self, eta: float):
        self.eta = eta
        self.rate_type = "Adagrad"
            
    def __call__(self, beta, gradients):
        if self.rate_type == "Constant":
            return beta - self.eta*gradients
        
        elif self.rate_type == "Momentum":
            if self.prev_v is None:
                self.prev_v = np.zeros_like(gradients)
            v = self.gamma*self.prev_v + self.eta*gradients
            self.prev_v = v
            return beta - v
        
        """
        elif self.rate_type == "Adagrad":
            delta = 1e-8
            G = np.zeros_like(gradients)
            G += gradients**2
            return beta - self.eta*gradients/(np.sqrt(G) + delta)"""
        