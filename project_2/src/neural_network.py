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
        manual_gradients = False,
    ):
        self.cost_fun = cost_fun
        self.cost_fun_der = grad(cost_fun, 0) #Double check this
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

    def compute_gradient(self, inputs, targets):
        pass

    def update_weights(self, layer_grads):
        pass

    # Unsure if we need these when using jax
    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, layers, inputs):
        pass

    def autograd_gradient(self, inputs, targets):
        pass
