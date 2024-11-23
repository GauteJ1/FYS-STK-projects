import autograd.numpy as np

class Update_Beta:
    """
    Class for updating the beta values in the gradient descent (optimizers)
    """

    def __init__(self) -> None:
        """
        Initializes rate type and iteration count
        """
        self.rate_type = ""
        self.iter_count = 1

    def reset(self):
        """
        Resets the iteration count and the previous values
        """
        self.prev_v = None
        self.G = None
        self.m_prev = None
        self.s_prev = None
        self.iter_count = 1

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
        self.G = None

    def adagrad_momentum(self, eta: float, gamma: float, delta: float = 1e-8) -> None:
        self.eta = eta
        self.gamma = gamma
        self.delta = delta
        self.rate_type = "Adagrad_Momentum"
        self.G = None
        self.prev_v = None

    def adam(
        self,
        eta: float = 1e-3,
        epsilon: float = 1e-8,
        b1: float = 0.9,
        b2: float = 0.999,
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

    def __call__(self, beta: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Updates the beta values based on the learning rate type
        """
        if self.rate_type == "Constant":
            return beta - self.eta * gradients

        elif self.rate_type == "Momentum":
            if self.prev_v is None or self.prev_v.shape != gradients.shape:
                self.prev_v = np.zeros_like(gradients)
            v = self.gamma * self.prev_v + self.eta * gradients
            self.prev_v = v
            return beta - v

        elif self.rate_type == "Adagrad":
            if self.G is None or self.G.shape != gradients.shape:
                self.G = np.zeros_like(gradients)
            self.G += gradients**2
            return beta - self.eta * gradients / (np.sqrt(self.G) + self.delta)

        elif self.rate_type == "Adagrad_Momentum":
            if self.prev_v is None or self.prev_v.shape != gradients.shape:
                self.prev_v = np.zeros_like(gradients)
            if self.G is None or self.G.shape != gradients.shape:
                self.G = np.zeros_like(gradients)
            self.G += gradients**2
            v = self.gamma * self.prev_v + self.eta * gradients / (np.sqrt(self.G) + self.delta)
            self.prev_v = v
            return beta - v

        elif self.rate_type == "Adam":
            if self.m_prev is None or self.m_prev.shape != gradients.shape:
                self.m_prev = np.zeros_like(gradients)
            if self.s_prev is None or self.s_prev.shape != gradients.shape:
                self.s_prev = np.zeros_like(gradients)
            beta, self.m_prev, self.s_prev = adam(
                beta,
                gradients,
                self.eta,
                self.b1,
                self.b2,
                self.epsilon,
                self.iter_count,
                self.m_prev,
                self.s_prev,
            )
            self.iter_count += 1
            return beta

        elif self.rate_type == "RMSprop":
            if self.s_prev is None or self.s_prev.shape != gradients.shape:
                self.s_prev = np.zeros_like(gradients)
            s = self.b * self.s_prev + (1 - self.b) * gradients**2
            beta = beta - self.eta * gradients / (np.sqrt(s) + self.epsilon)
            self.s_prev = s
            return beta

        else:
            raise RuntimeError("Learning rate type not set")

def adam(
    beta: np.ndarray,
    gradients: np.ndarray,
    eta: float,
    b1: float,
    b2: float,
    epsilon: float,
    iter: int,
    m_prev: np.ndarray,
    s_prev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = b1 * m_prev + (1 - b1) * gradients
    s = b2 * s_prev + (1 - b2) * gradients**2
    m_prev = m
    s_prev = s
    m = m / (1 - b1**iter)
    s = s / (1 - b2**iter)
    beta = beta - eta * m / (np.sqrt(s) + epsilon)
    return beta, m_prev, s_prev
