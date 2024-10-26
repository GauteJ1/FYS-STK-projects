import jax.numpy as jnp
import jax
import numpy as np
import json as json
from methods import *
from tqdm import tqdm
from jax import grad, jit, vmap


class NeuralNetwork:
    def __init__(
        self,
        network_shape,
        activation_funcs,
        cost_fun,
    ):
        self.cost_fun = cost_fun
        self.cost_fun_der = grad(cost_fun, 0)  # Double check this
        self.activation_funcs = [globals()[func] for func in activation_funcs]
        self.activation_funcs_der = [globals()[func + '_der'] for func in activation_funcs]
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
            z = jnp.dot(a, W.T) + b
            a = activation_func(z)
        return a
    
    def feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = np.dot(a, W.T) + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a

    def cost(self, inputs, targets):
        predict = self.predict(inputs)
        return self.cost_fun(predict, targets)

    
    def backpropagation_batch(self, inputs, target):
        layer_inputs, zs, predict = self.feed_forward_saver(inputs)

        layer_grads = [() for layer in self.layers]

        # We loop over the self.layers, from the last to the first
        for i in reversed(range(len(self.layers))):

            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_funcs_der[i]

            if i == len(self.layers) - 1:
                dC_da = self.cost_fun_der(predict, target)
            else:
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W
            
            dC_dz = dC_da * activation_der(z) 
            dC_dW = jnp.dot(dC_dz.T, layer_input) 
            dC_db = jnp.mean(dC_dz, axis=0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads
    
    def train_network(self, inputs, targets, epochs, learning_rate, use_jax=True):
        accuracy_list = []
        loss_list = []

        for i in tqdm(range(epochs)):
            layers_grad = self.backpropagation_batch(inputs, targets)
            
            for idx, ((W, b), (W_g, b_g)) in enumerate(zip(self.layers, layers_grad)):
                self.layers[idx] = (W - learning_rate * W_g, b - learning_rate * b_g)

            predictions = self.predict(inputs)
            accuracy_score = r_2(predictions, targets)
            accuracy_list.append(accuracy_score)
            loss = self.cost_fun(predictions, targets)
            loss_list.append(loss)

        return self.layers, accuracy_list, loss_list, predictions

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

    def save_network(self, file_name):
        # Write network info to json file

        # save layers 
        json.dump(self.layers, file_name)
        # save activation functions
        json.dump(self.activation_funcs, file_name)
        # save cost function
        json.dump(self.cost_fun, file_name)
        # save loss/accurracy

        pass

    def load_network(self, file_name):
        # Load network from json file
        pass 