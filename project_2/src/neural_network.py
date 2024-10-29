import jax.numpy as jnp
import jax
import numpy as np
import json as json
from methods import *
from tqdm import tqdm
from jax import grad, jit, vmap
from sklearn.model_selection import train_test_split


from learn_rate import Update_Beta


class NeuralNetwork:
    def __init__(
        self,
        network_shape: list[int],
        activation_funcs: list[str],    
        cost_func: str,
        type_of_network: str,
        update_strategy: str,
        manual_gradients: bool = False,
        train_test_split: bool = True,
    ):
        self.cost_func = cost_func

        self.activation_funcs = [globals()[func] for func in activation_funcs]
        self.activation_funcs_der = [globals()[func + '_der'] for func in activation_funcs]
        
        self.layers = self.create_layers(network_shape)
        self.type_of_network = type_of_network
        
        self.update_strategy = update_strategy
        self.update_beta = Update_Beta() # import from learn_rate.py
        self.manual_gradients = manual_gradients
        self.train_test_split = train_test_split

        self.epochs = 0  

    def set_update_strategy(self):

        if self.update_strategy == "Constant":
            self.update_beta.constant(self.learning_rate)
        elif self.update_strategy == "Momentum":
            self.update_beta.momentum_based(self.learning_rate, gamma=0.9)
        elif self.update_strategy == "Adagrad":
            self.update_beta.adagrad(self.learning_rate)
        elif self.update_strategy == "Adagrad_Momentum":
            self.update_beta.adagrad_momentum(self.learning_rate, gamma=0.9)
        elif self.update_strategy == "Adam":
            self.update_beta.adam(self.learning_rate)
        elif self.update_strategy == "RMSprop":
            self.update_beta.rmsprop(self.learning_rate)
        else:
            raise ValueError("Unsupported update strategy")
        
    def set_accuracy_function(self):
    
        if self.type_of_network == "classification":
            self.accuracy_func = recall
        elif self.type_of_network == "continuous":
            self.accuracy_func = r_2
        else:
            raise ValueError("Invalid type of network")
        
    def set_grads(self):
                
        if self.manual_gradients:
            self.gradient = self.manual_gradient
        else:
            self.gradient = self.jaxgrad_gradient
        

    def set_cost_function(self) -> None:
        if self.cost_func == "MSE":
            self.cost_fun = mse
        elif self.cost_func == "CrossEntropy":
            self.cost_fun = cross_entropy
        elif self.cost_func == "BinaryCrossEntropy":
            self.cost_fun = binary_cross_entropy
        else:
            raise ValueError("Unsupported cost function")

        self.cost_fun_der = grad(self.cost_fun, 0) 

        

    def create_layers(self, network_shape: list[int]) -> list:

        layers = []
        i_size = network_shape[0]
        
        np.random.seed(4155)
        
        for layer_output_size in network_shape[1:]:
            W = np.random.randn(layer_output_size, i_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size

        return layers

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        # Simple feed forward pass
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):  
            z = jnp.dot(a, W.T) + b
            a = activation_func(z)
        return a
    
    def feed_forward_saver(self, inputs: np.ndarray) -> tuple:
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = jnp.dot(a, W.T) + b
            a = activation_func(z)

            zs.append(z)
        
        return layer_inputs, zs, a
    
    def train_network(self, 
                      inputs: np.ndarray, 
                      targets: np.ndarray, 
                      epochs: int, 
                      learning_rate: float, 
                      batch_size: int = 64,
                      test_size: float = 0.3
        ) -> None:
        

        if self.train_test_split:

            train_inputs, test_inputs = inputs
            train_targets, test_targets = targets

        else:
            train_inputs = inputs
            train_targets = targets
            test_inputs = None
            test_targets = None

         # Initialize lists only if they are empty (for first training session)
        if not hasattr(self, 'loss') or not self.loss:
            self.loss = []
        if not hasattr(self, 'accuracy') or not self.accuracy:
            self.accuracy = []
        if not hasattr(self, 'test_loss') or not self.test_loss:
            self.test_loss = []
        if not hasattr(self, 'test_accuracy') or not self.test_accuracy:
            self.test_accuracy = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs += epochs

        self.set_accuracy_function()
        self.set_update_strategy()
        self.set_grads()
        self.set_cost_function()

        num_samples = train_inputs.shape[0]

        for epoch in range(epochs):

            permutation = np.random.permutation(num_samples)
            train_inputs = train_inputs[permutation]
            train_targets = train_targets[permutation]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_inputs = train_inputs[start:end]
                batch_targets = train_targets[start:end]

                layers_grad = self.gradient(batch_inputs, batch_targets)

                for idx, ((W, b), (W_g, b_g)) in enumerate(zip(self.layers, layers_grad)):
                    W_updated = self.update_beta(W, W_g)
                    b_updated = self.update_beta(b, b_g)  

                    self.layers[idx] = (W_updated, b_updated)


            train_predictions = self.predict(train_inputs)
            self.loss.append(self.cost_fun(train_predictions, train_targets))
            self.accuracy.append(self.accuracy_func(train_predictions, train_targets))

            if test_inputs is not None:

                test_predictions = self.predict(test_inputs)
                self.test_loss.append(self.cost_fun(test_predictions, test_targets))
                self.test_accuracy.append(self.accuracy_func(test_predictions, test_targets))
    

    def manual_gradient(self, inputs: np.ndarray, target: np.ndarray) -> list[float]:

        layer_inputs, zs, predict = self.feed_forward_saver(inputs)

        layer_grads = [() for layer in self.layers]

        # We loop over the self.layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_funcs_der[i]

            if i == len(self.layers) - 1:
                dC_da = self.cost_fun_der(predict, target)
            else:
                (W, b) = self.layers[i + 1]
                dC_da = (dC_dz @ W) 

            dC_dz = dC_da * activation_der(z)  
            dC_dW = jnp.dot(dC_dz.T, layer_input)
            dC_db = jnp.mean(dC_dz, axis=0)

            dC_db *= inputs.shape[0] ## MIA: check out this!!! Makes manuel == jaxgrad

            layer_grads[i] = (dC_dW, dC_db) 


        clipped_layer_grads = [(jnp.clip(W_grad, -1e3, 1e3), jnp.clip(b_grad, -1e3, 1e3)) for W_grad, b_grad in layer_grads]
        
        return clipped_layer_grads
    
    def jaxgrad_gradient(self, inputs: np.ndarray, targets: np.ndarray):
        # Function calculating jax gradient using a separate jax_grad_cost function

        def jax_grad_predict(layers: list[float], inputs: np.ndarray) -> np.ndarray:
            
            a = inputs
            for (W, b), activation_func in zip(layers, self.activation_funcs):
                z = jnp.dot(a, W.T) + b
                a = activation_func(z)
            return a

        def jax_grad_cost(layers: list[float], inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
            # Mimicing cost, but with layers as argument to use jax.grad on it
            predictions = jax_grad_predict(layers, inputs)
            return self.cost_fun(predictions, targets)

        gradients = grad(jax_grad_cost, argnums=0)(self.layers, inputs, targets)

        clipped_gradients = [(jnp.clip(W_grad, -1e3, 1e3), jnp.clip(b_grad, -1e3, 1e3)) for W_grad, b_grad in gradients]

        return clipped_gradients



    def save_network(self, file_name: str) -> None:
        # Convert JAX arrays to lists if necessary
        network_info = {
            "network_shape": [layer[0].shape[1] for layer in self.layers],
            "activation_funcs": [func.__name__ for func in self.activation_funcs],
            "cost_func": self.cost_func,
            "type_of_network": self.type_of_network,
            "update_strategy": self.update_strategy,
            "manual_gradients": self.manual_gradients,
            "layers": [(np.array(W).tolist(), np.array(b).tolist()) for W, b in self.layers],
            "loss_history": [float(loss) for loss in self.loss],
            "accuracy_history": [float(acc) for acc in self.accuracy],
            "loss_history_test": [float(loss) for loss in self.test_loss],
            "accuracy_history_test": [float(acc) for acc in self.test_accuracy],
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs
        }

        with open(file_name, 'w') as file:
            json.dump(network_info, file, indent=4)

    @classmethod
    def load_network(cls, file_name: str) -> "NeuralNetwork":
        # Load network info from JSON and initialize a new instance with it
        with open(file_name, 'r') as file:
            network_info = json.load(file)

        # Create a new instance with the loaded parameters
        model = cls(
            network_info["network_shape"],
            network_info["activation_funcs"],
            network_info["cost_func"],
            network_info["type_of_network"],
            network_info["update_strategy"],
            network_info["manual_gradients"],
        )

        # Set additional attributes
        model.layers = [(np.array(W), np.array(b)) for W, b in network_info["layers"]]
        model.loss = network_info.get("loss_history", [])
        model.accuracy = network_info.get("accuracy_history", [])
        model.test_loss = network_info.get("loss_history_test", [])
        model.test_accuracy = network_info.get("accuracy_history_test", [])
        model.learning_rate = network_info.get("learning_rate")
        model.batch_size = network_info.get("batch_size")
        model.epochs = network_info.get("epochs")

        return model

