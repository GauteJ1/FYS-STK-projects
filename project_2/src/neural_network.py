import jax.numpy as jnp
import jax
import numpy as np
import json as json
from methods import *
from tqdm import tqdm
from jax import grad, jit, vmap

from learn_rate import Update_Beta


class NeuralNetwork:
    def __init__(
        self,
        network_shape,
        activation_funcs,
        cost_fun,
        type_of_network,
        update_strategy,
    ):
        self.cost_fun = cost_fun
        self.cost_fun_der = grad(cost_fun, 0)  # Double check this
        self.activation_funcs = [globals()[func] for func in activation_funcs]
        self.activation_funcs_der = [globals()[func + '_der'] for func in activation_funcs]
        self.layers = self.create_layers(network_shape)
        self.type_of_network = type_of_network
        self.update_strategy = update_strategy
        self.update_beta = Update_Beta()

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
            z = jnp.dot(a, W.T) + b
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
            #print(f"layer_input: {layer_input[0]}, z: {z[0]}, activation_der: {activation_der(z)[0]}")
            if i == len(self.layers) - 1:
                dC_da = self.cost_fun_der(predict, target)
                #print(f" for i = {i} dC_da: {dC_da[0]}")
            else:
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W
                #print(f" for i = {i} dC_da: {dC_da[0]}")

            dC_da = jnp.clip(dC_da, -1e3, 1e3)
            dC_dz = dC_da * activation_der(z) 
            dC_dz = jnp.clip(dC_dz, -1e3, 1e3)

            #print(f" for i = {i} dC_dz: {dC_dz[0]} with dC_da: {dC_da[0]} and activation_der: {activation_der(z)[0]}")
            dC_dW = jnp.dot(dC_dz.T, layer_input) 
            dC_db = jnp.mean(dC_dz, axis=0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads
    
    def train_network(self, inputs, targets, epochs, learning_rate, batch_size=10, manuel_grads=True):
        
        # Set accuracy function based on network type
        if self.type_of_network == "classification":
            self.accuracy_func = accuracy_one_hot
        elif self.type_of_network == "regression":
            self.accuracy_func = r_2
        else:
            raise ValueError("Invalid type of network")
        

        if self.update_strategy == "Constant":
            self.update_beta.constant(learning_rate)
        elif self.update_strategy == "Momentum":
            self.update_beta.momentum_based(learning_rate, gamma=0.9)
        elif self.update_strategy == "Adagrad":
            self.update_beta.adagrad(learning_rate)
        elif self.update_strategy == "Adagrad_Momentum":
            self.update_beta.adagrad_momentum(learning_rate, gamma=0.9)
        elif self.update_strategy == "Adam":
            self.update_beta.adam(learning_rate)
        elif self.update_strategy == "RMSprop":
            self.update_beta.rmsprop(learning_rate)
        else:
            raise ValueError("Unsupported update strategy")
        
        accuracy_list = []
        loss_list = []
        num_samples = inputs.shape[0]

        if manuel_grads:
            for i in tqdm(range(epochs)):
                # Shuffle data at the beginning of each epoch
                permutation = np.random.permutation(num_samples)
                inputs_shuffled = inputs[permutation]
                targets_shuffled = targets[permutation]

                for start in range(0, num_samples, batch_size):
                    end = start + batch_size
                    batch_inputs = inputs_shuffled[start:end]
                    batch_targets = targets_shuffled[start:end]

                    # Compute gradients for the batch
                    layers_grad = self.backpropagation_batch(batch_inputs, batch_targets)
                    # Update weights
                    for idx, ((W, b), (W_g, b_g)) in enumerate(zip(self.layers, layers_grad)):
                        # Update W and b using the chosen strategy in Update_Beta
                        W_updated = self.update_beta(W, W_g, iter=i+1)  # Update weights
                        b_updated = self.update_beta(b, b_g,  iter=i+1)  # Update biases
                        self.layers[idx] = (W_updated, b_updated)

                # Calculate metrics after each epoch
                predictions = self.predict(inputs)
                accuracy_score = self.accuracy_func(predictions, targets)
                accuracy_list.append(accuracy_score)
                loss = self.cost_fun(predictions, targets)
                loss_list.append(loss)

        else: 
            for i in tqdm(range(epochs)):
                inputs = np.array(inputs)
                targets = np.array(targets)

                # Shuffle data at the beginning of each epoch
                permutation = np.random.permutation(num_samples)
                inputs_shuffled = inputs[permutation]
                targets_shuffled = targets[permutation]

                for start in range(0, num_samples, batch_size):
                    end = start + batch_size
                    batch_inputs = inputs_shuffled[start:end]
                    batch_targets = targets_shuffled[start:end]

                    # Compute gradients for the batch
                    layers_grad = self.jaxgrad_gradient(batch_inputs, batch_targets)
                    # Update weights
                    for idx, ((W, b), (W_g, b_g)) in enumerate(zip(self.layers, layers_grad)):
                        # Update W and b using the chosen strategy in Update_Beta
                        W_updated = self.update_beta(W, W_g)
                        b_updated = self.update_beta(b, b_g)  

                        self.layers[idx] = (W_updated, b_updated)

                # Calculate metrics after each epoch
                predictions = self.predict(inputs)
                accuracy_score = self.accuracy_func(predictions, targets)
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
                z = jnp.dot(a, W.T) + b
                a = activation_func(z)
            return a

        def jax_grad_cost(layers, inputs, targets):
            # Mimicing cost, but with layers as argument to use jax.grad on it
            predictions = jax_grad_predict(layers, inputs)
            return self.cost_fun(predictions, targets)

        gradients = grad(jax_grad_cost, argnums=0)(self.layers, inputs, targets)

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