import jax.numpy as jnp
import jax
from jax._src.typing import Array

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
        self.G = None

    def adagrad_momentum(self, eta: float, gamma: float, delta: float = 1e-8) -> None:
        self.eta= eta
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

    def __call__(self, beta: Array, gradients: Array, iter: int = 1) -> Array:

        if self.rate_type == "Constant":
            return beta - self.eta * gradients

        elif self.rate_type == "Momentum":
            if self.prev_v is None or self.prev_v.shape != gradients.shape:
                self.prev_v = jnp.zeros_like(gradients)
            v = self.gamma * self.prev_v + self.eta * gradients
            self.prev_v = v
            return beta - v

        elif self.rate_type == "Adagrad":
            if self.G is None or self.G.shape != gradients.shape:
                self.G = jnp.zeros_like(gradients)
            self.G += gradients**2

            return beta - self.eta * gradients / (jnp.sqrt(self.G) + self.delta)
        
        elif self.rate_type == "Adagrad_Momentum":
            if self.prev_v is None or self.prev_v.shape != gradients.shape:
                self.prev_v = jnp.zeros_like(gradients)
            if self.G is None:
                self.G = jnp.zeros_like(gradients)

            self.G = gradients**2
            v = self.gamma * self.prev_v + self.eta * gradients / (jnp.sqrt(self.G) + self.delta)
            self.prev_v = v
            return beta - v
        
        elif self.rate_type == "Adam":
            if self.m_prev is None or self.m_prev.shape != gradients.shape:
                self.m_prev = jnp.zeros_like(gradients)
            if self.s_prev is None or self.s_prev.shape != gradients.shape:
                self.s_prev = jnp.zeros_like(gradients)
            
            beta, self.m_prev, self.s_prev = adam(beta, gradients, self.eta, self.b1, self.b2, self.epsilon, iter, self.m_prev, self.s_prev)
                                                  
            return beta

        elif self.rate_type == "RMSprop":
            if self.s_prev is None or self.s_prev.shape != gradients.shape:
                self.s_prev = jnp.zeros_like(gradients)
            s = self.b * self.s_prev + (1 - self.b) * gradients**2
            beta = beta - self.eta * gradients / (jnp.sqrt(s) + self.epsilon)

            self.s_prev = s
            return beta

        else:
            raise RuntimeError("Learning rate type not set")

@jax.jit
def adam(beta: Array, gradients: Array, eta: float, b1: float, b2: float, epsilon: float, iter: int, m_prev: Array, s_prev: Array) -> tuple[Array, Array, Array]:
    m = b1 * m_prev + (1 - b1) * gradients
    s = b2 * s_prev + (1 - b2) * gradients**2

    m_prev = m
    s_prev = s

    m = m / (1 - b1**iter)
    s = s / (1 - b2**iter)
    
    beta = beta - eta * m / (jnp.sqrt(s) + epsilon)

    return beta, m_prev, s_prev
