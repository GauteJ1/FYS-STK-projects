import json as json
from methods import *
from autograd import jacobian, hessian, grad
import autograd.numpy as np
from learn_rate import Update_Beta
from initialisation import initialize
from tqdm import tqdm

seed = 4155

class NeuralNetwork:
    """
    A class for creating and training a neural network for solving the 1D heat equation.
    """

    def __init__(self, network_shape: list[int], 
                 activation_funcs: list[str], 
                 optimizer: str, 
                 initialization: str) -> None:
        
        self.activation_funcs = [globals()[func] for func in activation_funcs]
        self.activation_funcs_der = [globals()[func + "_der"] for func in activation_funcs]

        self.network_shape = network_shape

        if len(activation_funcs) != len(network_shape) - 1:
            raise ValueError("Number of activation functions must be equal to the number of hidden layers")

        self.init = initialization
        self.layers = self.create_layers(network_shape)
        
        self.update_strategy = optimizer
        
        self.update_beta = Update_Beta()  # import from learn_rate.py
        
        self.epochs = 0

    def set_update_strategy(self) -> None:
        
        """
        Set the update strategy for the neural network

        Raises
        ------
        ValueError
            If the update strategy is not supported
        """

        if self.update_strategy == "Constant":
            self.update_beta.constant(self.learning_rate)
        elif self.update_strategy == "Momentum":
            self.update_beta.momentum_based(self.learning_rate, gamma=0.95)
        elif self.update_strategy == "Adagrad":
            self.update_beta.adagrad(self.learning_rate)
        elif self.update_strategy == "Adagrad_Momentum":
            self.update_beta.adagrad_momentum(self.learning_rate, gamma=0.95)
        elif self.update_strategy == "Adam":
            self.update_beta.adam(self.learning_rate)
        elif self.update_strategy == "RMSprop":
            self.update_beta.rmsprop(self.learning_rate)
        
        else:
            raise ValueError("Unsupported update strategy")
        
    def f(self, point):
        """
        Calculate the right-hand side of the differential equation

        """
        return 0.

    def cost_function(self, x, t):

        """
        Calculate the cost function for the neural network

        Parameters
        ----------
        x : np.ndarray
            The x-values to evaluate the cost function at
        t : np.ndarray
            The t-values to evaluate the cost function at

        Returns
        -------
        final_cost : float
            The final cost of the neural network
        """
        cost_sum = 0
        
        g_t_jacobian_func = jacobian(self.g_trial)
        g_t_hessian_func = hessian(self.g_trial)
        
        for x_ in x:
            for t_ in t:
                point = np.array([x_, t_])
        
                g_t_jacobian = g_t_jacobian_func(point)
                g_t_hessian = g_t_hessian_func(point)
        
                g_t_dt = g_t_jacobian[0, 0, 1, 0]
                g_t_d2x = g_t_hessian[0, 0, 0, 0, 1, 0]
        
                func = self.f(point)
        
                err_sqr = ((g_t_dt - g_t_d2x) - func) ** 2
                cost_sum += err_sqr

        final_cost = cost_sum / (np.size(x) * np.size(t))

        return final_cost
    
    def cost_function_with_params(self, params, x, t):

        """
        Calculate the cost function for the neural network with given parameters

        Parameters
        ----------
        params : np.ndarray
            The parameters to evaluate the cost function with
        x : np.ndarray
            The x-values to evaluate the cost function at
        t : np.ndarray
            The t-values to evaluate the cost function at

        Returns
        -------
        cost : float
            The cost of the neural network with the given parameters
        """

        self.layers = self.reshape_layers(params)
        cost = self.cost_function(x, t)
        
        return cost

    def create_layers(self, network_shape: list[int]) -> list:

        """
        Initialization of the layers of the neural network

        Parameters
        ----------
        network_shape : list[int]
            The shape of the neural network

        Returns
        -------
        list
            A list of tuples containing the weight matrices and bias vectors for each layer in the network.
            Each tuple is in the form (W, b), where W is the weight matrix and b is the bias vector for a layer.

        Raises
        ------
        ValueError
            If an unsupported initialization method is specified

        """

        if self.init ==  "Standard":
            layers = initialize(network_shape, "Standard")
        elif self.init == "Xavier":
            layers = initialize(network_shape, "Xavier")
        elif self.init == "He":
            layers = initialize(network_shape, "He")
        else:
            raise ValueError("Unsupported initialization method")
        
        return layers
        
        # layers = []
        # i_size = network_shape[0]
        # np.random.seed(seed)
        
        # for layer_output_size in network_shape[1:]:
        #     W = np.random.randn(layer_output_size, i_size)
        #     b = np.random.randn(layer_output_size)
            
        #     layers.append((W, b))
        #     i_size = layer_output_size
        
        # return layers

    def ravel_layers(self, layers: list) -> np.ndarray:

        """
        Ravel the layers of the neural network

        Returns
        -------
        np.ndarray
            A raveled array of the layers
        """


        theta = np.array([])
        
        for W, b in layers:
            theta = np.append(theta, np.ravel(W))
            theta = np.append(theta, np.ravel(b))
        
        return theta.ravel()

    # def unravel_layers(self, theta: np.ndarray) -> list:
        
    #     network_shape = self.network_shape
    #     layers = []
    #     i_size = network_shape[0]
    #     index = 0
        
    #     for layer_output_size in network_shape[1:]:
    #         W_shape = (layer_output_size, i_size)
    #         W_size = np.prod(W_shape)
    #         W = theta[index:index + W_size].reshape(W_shape)
    #         index += W_size
    #         b_shape = (layer_output_size,)
    #         b = theta[index:index + layer_output_size].reshape(b_shape)
    #         index += layer_output_size
    #         layers.append((W, b))
    #         i_size = layer_output_size
    #     return layers

    def reshape_layers(self, theta: np.ndarray) -> list:
        """
        Reshape the layers of the neural network

        Returns
        -------
        list
            A list of tuples containing the weights and biases for each layer
        """

        network_shape = self.network_shape

        layers = []
        i_size = network_shape[0]

        index = 0

        np.random.seed(4155)

        for layer_output_size in network_shape[1:]:
            W = np.reshape(
                theta[index : index + (layer_output_size * i_size)],
                (layer_output_size, i_size),
            )
            index += layer_output_size * i_size
            b = np.reshape(theta[index : index + layer_output_size], layer_output_size)
            index += layer_output_size
            layers.append((W, b))
            i_size = layer_output_size

        return layers

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        
        """
        Predict the output of the neural network

        Parameters
        ----------
        inputs : np.ndarray
            The input data

        Returns
        -------
        a : np.ndarray
            The predicted output
        """

        if inputs.ndim == 3:
            inputs = inputs.reshape(inputs.shape[0], -1)
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = np.dot(a, W.T) + b
            a = activation_func(z)
        return a

    def g_trial(self, point) -> np.ndarray:

        """
        Trial solution with initial condition included.

        Returns
        -------
        gtrial : np.ndarray
            The trial solution
        """

        x, t = point

        point_array = np.array([x, t]).reshape(1, -1)
        
        g1 = (1 - t) * np.sin(np.pi * x) + x * (1 - x) * t * self.predict(point_array)
        g2 = np.sin(np.pi * x) * np.exp(-np.pi**2 * t) + (1 - x ) * x * self.predict(point_array)
        
        return g2

    def train_network(self, 
                      x: np.ndarray, 
                      t: np.ndarray,
                      epochs: int, 
                      learning_rate: float, 
                      batch_size: int = 100) -> None:
        
        """
        Train the neural network

        Parameters
        ----------
        x : np.ndarray
            The input data in the x-direction
        t : np.ndarray
            The input data in the t-direction
        epochs : int
            The number of epochs to train the network
        learning_rate : float
            The learning rate
        batch_size : int, optional
            The batch size, by default 100
        """
        
        self.loss = []
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs += epochs
        self.set_update_strategy()
        
        cost_function_grad = grad(self.cost_function_with_params, 0)
        
        theta_initial = self.ravel_layers(self.layers)

        for epoch in tqdm(range(epochs)):
            
            cost = self.cost_function(x, t)
            self.loss.append(cost)
            
            theta_grad = cost_function_grad(theta_initial, x, t)
            theta_updated = self.update_beta(theta_initial, theta_grad)
            self.layers = self.reshape_layers(theta_updated)
            theta_initial = theta_updated # updating the weights and biases

        # final cost
        cost = self.cost_function(x, t)
        self.loss.append(cost)

        x_t_combinations = np.array([[x_, t_] for x_ in x for t_ in t])
        self.final_preds = self.predict(x_t_combinations).reshape(len(x), len(t))












##### PINNS from ChatGPT #####


# import json as json
# from methods import *
# from autograd import jacobian, hessian, grad
# import autograd.numpy as np
# from learn_rate import Update_Beta
# from tqdm import tqdm

# class NeuralNetwork:
#     """
#     A Physics-Informed Neural Network (PINN) class for solving differential equations.
#     """

#     def __init__(self, network_shape: list[int], activation_funcs: list[str], optimizer: str) -> None:
#         self.activation_funcs = [globals()[func] for func in activation_funcs]
#         self.network_shape = network_shape
#         self.layers = self.create_layers(network_shape)
#         self.update_strategy = optimizer
#         self.update_beta = Update_Beta()  # import from learn_rate.py
#         self.epochs = 0

#     def set_update_strategy(self) -> None:
#         if self.update_strategy == "Constant":
#             self.update_beta.constant(self.learning_rate)
#         elif self.update_strategy == "Momentum":
#             self.update_beta.momentum_based(self.learning_rate, gamma=0.95)
#         elif self.update_strategy == "Adagrad":
#             self.update_beta.adagrad(self.learning_rate)
#         elif self.update_strategy == "Adagrad_Momentum":
#             self.update_beta.adagrad_momentum(self.learning_rate, gamma=0.95)
#         elif self.update_strategy == "Adam":
#             self.update_beta.adam(self.learning_rate)
#         elif self.update_strategy == "RMSprop":
#             self.update_beta.rmsprop(self.learning_rate)
#         else:
#             raise ValueError("Unsupported update strategy")
    
#     def create_layers(self, network_shape: list[int]) -> list:
#         layers = []
#         i_size = network_shape[0]
#         np.random.seed(4155)
#         for layer_output_size in network_shape[1:]:
#             W = np.random.randn(layer_output_size, i_size) * np.sqrt(2 / i_size)
#             b = np.zeros(layer_output_size)
#             layers.append((W, b))
#             i_size = layer_output_size
#         return layers

#     def ravel_layers(self, layers: list) -> np.ndarray:
#         theta = np.array([])
#         for W, b in layers:
#             theta = np.append(theta, np.ravel(W))
#             theta = np.append(theta, np.ravel(b))
#         return theta.ravel()

#     def unravel_layers(self, theta: np.ndarray) -> list:
#         network_shape = self.network_shape
#         layers = []
#         i_size = network_shape[0]
#         index = 0
#         for layer_output_size in network_shape[1:]:
#             W_shape = (layer_output_size, i_size)
#             W_size = np.prod(W_shape)
#             W = theta[index:index + W_size].reshape(W_shape)
#             index += W_size
#             b_shape = (layer_output_size,)
#             b = theta[index:index + layer_output_size].reshape(b_shape)
#             index += layer_output_size
#             layers.append((W, b))
#             i_size = layer_output_size
#         return layers

#     def predict(self, inputs: np.ndarray) -> np.ndarray:
#         # Flatten inputs if they have additional dimensions
#         if inputs.ndim > 2:
#             inputs = inputs.reshape(inputs.shape[0], -1)
            
#         a = inputs
#         for (W, b), activation_func in zip(self.layers, self.activation_funcs):
#             z = np.dot(a, W.T) + b
#             a = activation_func(z)
#         return a


#     def g_trial(self, x, t) -> np.ndarray:
#         """
#         Trial solution with initial condition included.
#         """
#         return (1 - t) * np.sin(np.pi * x) + x * (1 - x) * t * self.predict(np.hstack((x, t)))

#     def physics_based_cost_function(self, params, x, t):
#         """
#         Calculate the physics-based loss by enforcing the differential equation.
#         """
#         self.layers = self.unravel_layers(params)  # Update network with current parameters

#         g_t_jacobian_func = jacobian(self.g_trial, 1)  # derivative wrt time
#         g_t_hessian_func = hessian(self.g_trial, 0)    # second derivative wrt space

#         cost_sum = 0
#         for x_ in x:
#             for t_ in t:
#                 point = np.array([[x_], [t_]])

#                 g_t = self.g_trial(x_, t_)
#                 g_t_dt = g_t_jacobian_func(x_, t_)
#                 g_t_d2x = g_t_hessian_func(x_, t_)

#                 # Physics loss: (∂g/∂t - ∂²g/∂x²) should be zero
#                 physics_residual = g_t_dt - g_t_d2x
#                 cost_sum += physics_residual**2

#         return cost_sum / (np.size(x) * np.size(t))

#     def train_network(self, x: np.ndarray, t: np.ndarray, epochs: int, learning_rate: float) -> None:
#         self.loss = []
#         self.learning_rate = learning_rate
#         self.set_update_strategy()
#         cost_function_grad = grad(self.physics_based_cost_function, 0)
#         theta_initial = self.ravel_layers(self.layers)

#         for epoch in tqdm(range(epochs)):
#             cost = self.physics_based_cost_function(theta_initial, x, t)
#             self.loss.append(cost[0][0][0])
#             theta_grad = cost_function_grad(theta_initial, x, t)
#             theta_updated = self.update_beta(theta_initial, theta_grad)
#             self.layers = self.unravel_layers(theta_updated)
#             theta_initial = theta_updated

#         # Generate final predictions for all combinations of x and t
#         x_t_combinations = np.array([[x_, t_] for x_ in x for t_ in t]).reshape(-1, 2)
#         self.final_preds = self.predict(x_t_combinations).reshape(len(x), len(t))
