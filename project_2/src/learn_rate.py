import numpy as np


class Update_Beta:
    def __init__(self) -> None:
        self.rate_type = ""

    def constant(self, eta: float) -> None:
        self.eta = eta
        self.rate_type = "Constant"

    def momentum_based(self, eta: float, gamma: float) -> None:
        self.eta = eta
        self.gamma = gamma
        self.rate_type = "Momentum"
        self.prev_v = None

    def adagrad(self, eta: float, delta: float = 1e-8) -> None:
        self.eta = eta
        self.delta = delta
        self.rate_type = "Adagrad"

    def adam(
        self,
        eta: float = 1e-3,
        epsilon: float = 1e-8,
        b1: float = 0.9,
        b2: float = 0.999
    ) -> None:
        self.eta = eta
        self.epsilon = epsilon
        self.b1 = b1
        self.b2 = b2
        self.rate_type = "Adam"
        self.m_prev = None
        self.s_prev = None

    def rmsprop(self, eta: float = 1e-3, epsilon: float = 1e-8, b: float = 0.9) -> None:
        self.eta = eta
        self.epsilon = epsilon
        self.b = b
        self.rate_type = "RMSprop"
        self.s_prev = None

    def __call__(self, beta: np.ndarray, gradients: np.ndarray, iter: int = 1) -> np.ndarray:
        if self.rate_type == "Constant":
            return beta - self.eta * gradients

        elif self.rate_type == "Momentum":
            if self.prev_v is None:
                self.prev_v = np.zeros_like(gradients)
            v = self.gamma * self.prev_v + self.eta * gradients
            self.prev_v = v
            return beta - v

        elif self.rate_type == "Adagrad":
            G = gradients**2
            return beta - self.eta * gradients / (np.sqrt(G) + self.delta)

        elif self.rate_type == "Adam":
            if self.m_prev is None:
                self.m_prev = np.zeros_like(gradients)
            if self.s_prev is None:
                self.s_prev = np.zeros_like(gradients)

            m = self.b1 * self.m_prev + (1 - self.b1) * gradients
            s = self.b2 * self.s_prev + (1 - self.b2) * gradients**2

            self.m_prev = m
            self.s_prev = s

            m = m / (1 - self.b1**iter)
            s = s / (1 - self.b2**iter)
            
            beta = beta - self.eta * m / (np.sqrt(s) + self.epsilon)

            return beta

        elif self.rate_type == "RMSprop":
            if self.s_prev is None:
                self.s_prev = np.zeros_like(gradients)
            s = self.b * self.s_prev + (1 - self.b) * gradients**2
            beta = beta - self.eta * gradients / (np.sqrt(s) + self.epsilon)

            self.s_prev = s
            return beta

        else:
            raise RuntimeError("Learning rate type not set")
