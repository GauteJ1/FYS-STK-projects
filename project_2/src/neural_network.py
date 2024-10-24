import jax.numpy as jnp
import jax
import numpy as np

from jax import grad, jit


class NeuralNetwork:
    def __init__(
        self,
        network_shape,
        activation_funcs,
        cost_fun,
    ):
        self.cost_fun = cost_fun
        self.cost_fun_der = grad(cost_fun, 0)  # Double check this
        self.activation_funcs = activation_funcs
        self.activation_funcs_der = [grad(func) for func in activation_funcs]
        self.layers = self.create_layers(network_shape)

    def create_layers(self, network_shape):
        layers = []
        i_size = network_shape[0]

        for layer_output_size in network_shape[1:]:
            W = np.random.randn(layer_output_size, i_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size

        return layers

    def predict(self, inputs):
        # Simple feed forward pass
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = W @ a + b
            a = activation_func(z)
        return a

    def cost(self, inputs, targets):
        predict = self.predict(inputs)
        return self.cost_fun(predict, targets)
    
    def train_network(self, epochs, tol, batch, use_jax=True):
        # Train neural network
        pass

    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = W @ a + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a

    def manual_gradient(self, inputs, targets):
        pass  # Fyll inn fra weekly exercise 43
        # Må være brukbar på batches også

    def update_weights(self, layer_grads):
        # Include learning rate class?
        pass

    def jaxgrad_gradient(self, inputs, targets):
        # Function calculating jax gradient using a separate jax_grad_cost function
        def jax_grad_predict(layers, inputs):
            a = inputs
            for (W, b), activation_func in zip(layers, self.activation_funcs):
                z = W @ a + b
                a = activation_func(z)
            return a

        def jax_grad_cost(layers, inputs, targets):
            # Mimicing cost, but with layers as argument to use jax.grad on it
            predict = jax_grad_predict(layers, inputs)
            return self.cost_fun(inputs, targets)

        gradients = grad(jax_grad_cost, 0)

        return gradients
