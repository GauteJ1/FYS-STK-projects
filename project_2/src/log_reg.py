import numpy as np
import jax

from methods import sigmoid, binary_cross_entropy


class LogReg:
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        l2_reg_param: float,
        update_strategy: str,
        # train_test_split: bool = True,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.activation = sigmoid
        self.l2_reg_param = l2_reg_param

        weights = np.random.rand(output_shape, input_shape)
        bias = np.random.rand(output_shape)
        self.model = (weights, bias)

        self.eps = 1e-10 #Small constant to avoid division by zero errors due to machine precision

    def predict(self, x: np.ndarray) -> np.ndarray:
        weights, bias = self.model
        z = weights @ x + bias
        y = self.activation(z)

        return y

    def cost(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        preds = self.predict(inputs)

        
        preds = np.clip(preds, self.eps, 1 - self.eps)

        cost = 0
        for p, y in zip(preds, targets):
            if y == 1:
                cost -= np.log(p)
            else:
                cost -= np.log(1 - p)

        weights = self.model[0]
        cost += self.l2_reg_param * np.mean(weights**2)

        return cost

    def gradients(self):

        def jax_cost(model, inputs, targets):
            weights, bias = model
            z = weights @ inputs + bias
            y = self.activation(z)

            preds = np.clip(y, self.eps, 1 - self.eps)
            
            cost = 0
            for p, y in zip(preds, targets):
                if y == 1:
                    cost -= np.log(p)
                else:
                    cost -= np.log(1 - p)

            cost += self.l2_reg_param * np.mean(weights**2)

            return cost
        
        gradients = jax.grad(jax_cost, 0)
        return gradients
        

    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        max_epochs: int,
        learning_rate: float,
        error_tol: float = 1e-5,
        batch_size: int = 100,
        save_stats: bool = False,
        train_test: bool = True
    ):
        if self.train_test_split:
            train_inputs, test_inputs = inputs
            train_targets, test_targets = targets
        else:
            train_inputs = inputs
            train_targets = targets
            test_inputs = None
            test_targets = None

        if save_stats:
            self.loss = []
            self.accuracy = []
            self.test_loss = []
            self.test_accuracy = []

        num_samples = train_inputs.shape[0]
        if batch_size == 0:
            m = num_samples
        else:
            m = int(num_samples / batch_size)

        for epoch in range(max_epochs):

            random_idx = batch_size * np.random.randint(m)
            batch_inputs = train_inputs[random_idx : random_idx + batch_size]
            batch_targets = train_targets[random_idx : random_idx + batch_size]
        
            gradients = self.gradients()(self.model, )