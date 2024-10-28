import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import jax.numpy as jnp
import numpy as np
import optax
from jax import nn as jax_nn
from jax import grad
import jax as jax
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from neural_network import NeuralNetwork
from methods import *
from data_gen import FrankeDataGen, CancerData
from learn_rate import Update_Beta

np.random.seed(4155)
torch.manual_seed(4155)

class TestNeuralNetworkComparison(unittest.TestCase):

    def setUp(self):
        data = FrankeDataGen(noise=False)
        self.inputs = jnp.column_stack((data.x.flatten(), data.y.flatten()))  
        self.targets = data.z.ravel().reshape(-1, 1)  

        self.network_shape = [2, 8, 1]
        self.activation_funcs = ["ReLU", "identity"]
        self.cost_func = "MSE"

        self.type_of_network = "regression"
        self.update_strategy = "Constant"
        self.manual_gradients = False

        self.train_test_split = False

        self.custom_model = NeuralNetwork(
            self.network_shape, 
            self.activation_funcs, 
            self.cost_func, 
            self.type_of_network,
            self.update_strategy,
            self.manual_gradients,
            self.train_test_split
        )

    def train_pytorch_model(self, inputs, targets, epochs, learning_rate):

        inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.float32)
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)


        class PyTorchModel(nn.Module):
            def __init__(self):
                super(PyTorchModel, self).__init__()
                self.fc1 = nn.Linear(2, 8)
                self.fc2 = nn.Linear(8, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
                self.identity = nn.Identity()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.identity(self.fc2(x))
                return x


        model = PyTorchModel()

        np.random.seed(4155)

        # ensuring exact same initialization
        network_shape = [2, 8, 1]
        layers = []
        i_size = network_shape[0]

        for layer_output_size in network_shape[1:]:
            W = np.random.randn(layer_output_size, i_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size

        with torch.no_grad():
            model.fc1.weight.copy_(torch.tensor(layers[0][0], dtype=torch.float32))
            model.fc1.bias.copy_(torch.tensor(layers[0][1], dtype=torch.float32))
            model.fc2.weight.copy_(torch.tensor(layers[1][0], dtype=torch.float32))
            model.fc2.bias.copy_(torch.tensor(layers[1][1], dtype=torch.float32))


        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs_tensor)
            loss = criterion(outputs, targets_tensor)
            loss.backward()
            optimizer.step()

        return loss.item()


    def test_custom_vs_pytorch(self):

        n_epochs = 100
        learning_rate = 0.001
        batch_size = 10201 # Full batch, i.e. like no batching

        self.custom_model.train_network(
            self.inputs, 
            self.targets, 
            epochs=n_epochs, 
            learning_rate=learning_rate,
            batch_size=batch_size, 
        )

        custom_loss = self.custom_model.loss
        
        pytorch_loss = self.train_pytorch_model(self.inputs, self.targets, epochs=n_epochs, learning_rate=learning_rate)

        acceptable_loss_diff = 0.05
        self.assertAlmostEqual(custom_loss[-1], pytorch_loss, delta=acceptable_loss_diff, 
                            msg="Final loss differs too much from PyTorch model in the regression case.")
        

    # comparing manual_gradients=False and manual_gradients=True for the custom model
    def test_manual_vs_jax_gradients(self):
        model = NeuralNetwork(
            network_shape=[2, 8, 1], 
            activation_funcs=["ReLU", "sigmoid"], 
            cost_func="MSE", 
            type_of_network="regression",
            update_strategy="Constant",
            manual_gradients=False, # has nothing to say as we call manual_gradient and jaxgrad_gradient directly 
            train_test_split=False
        )

        # Define test case with mock data
        test_inputs = np.random.rand(800, 2)
        test_targets = np.random.rand(800, 1)
        model.set_cost_function()

        # Compute gradients with both methods
        manual_grads = model.manual_gradient(test_inputs, test_targets)
        jax_grads = model.jaxgrad_gradient(test_inputs, test_targets)

        # Assert numerical equivalence within tolerance
        for (manual, jax) in zip(manual_grads, jax_grads):
            for m_grad, j_grad in zip(manual, jax):
                self.assertTrue(
                    np.allclose(m_grad, j_grad, atol=1e-5),
                    msg="Gradients differ at a layer"
                )

class TestNeuralNetworkBinaryClassificationComparison(unittest.TestCase):

    def setUp(self):
        # Load CancerData
        data = CancerData()
        self.inputs = np.array(data.x)
        self.targets = np.array(data.y).reshape(-1, 1)

        # Define network structure
        input_size = 30
        output_size = 1
        hidden_layers = [32, 16]
        self.network_shape = [input_size] + hidden_layers + [output_size]
        self.activation_funcs = ["ReLU", "ReLU", "sigmoid"]
        self.cost_func = "BinaryCrossEntropy"

        # Classification setup
        self.type_of_network = "classification"
        self.update_strategy = "Constant"
        self.manual_gradients = False
        self.train_test_split = False

        # Initialize custom model
        self.custom_model = NeuralNetwork(
            self.network_shape, 
            self.activation_funcs, 
            self.cost_func, 
            self.type_of_network,
            self.update_strategy,
            self.manual_gradients,
            self.train_test_split
        )

    def train_pytorch_model(self, inputs, targets, epochs, learning_rate):
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        class PyTorchModel(nn.Module):
            def __init__(self):
                super(PyTorchModel, self).__init__()
                self.fc1 = nn.Linear(30, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 1)
                self.sigmoid = nn.Sigmoid()
                self.relu = nn.ReLU()   

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        model = PyTorchModel()

        # ensuring exact same initialization
        np.random.seed(4155)
        network_shape = [30, 32, 16, 1]
        layers = []
        i_size = network_shape[0]

        for layer_output_size in network_shape[1:]:
            W = np.random.randn(layer_output_size, i_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size

        with torch.no_grad():
            model.fc1.weight.copy_(torch.tensor(layers[0][0], dtype=torch.float32))
            model.fc1.bias.copy_(torch.tensor(layers[0][1], dtype=torch.float32))
            model.fc2.weight.copy_(torch.tensor(layers[1][0], dtype=torch.float32))
            model.fc2.bias.copy_(torch.tensor(layers[1][1], dtype=torch.float32))
            model.fc3.weight.copy_(torch.tensor(layers[2][0], dtype=torch.float32))
            model.fc3.bias.copy_(torch.tensor(layers[2][1], dtype=torch.float32))


        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs_tensor)
            loss = criterion(outputs, targets_tensor)
            loss.backward()
            optimizer.step()

        return loss.item()

    def test_custom_vs_pytorch(self):
        # Test parameters
        epochs = 100
        learning_rate = 0.001
        batch_size = 569 # Full batch, i.e. like no batching

        # Train custom model
        self.custom_model.train_network(
            self.inputs, 
            self.targets, 
            epochs=epochs, 
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        custom_loss = self.custom_model.loss

        # Train PyTorch model
        pytorch_loss = self.train_pytorch_model(self.inputs, self.targets, epochs=epochs, learning_rate=learning_rate)

        # Compare final losses with an acceptable tolerance
        acceptable_loss_diff = 0.05
        self.assertAlmostEqual(custom_loss[-1], pytorch_loss, delta=acceptable_loss_diff, 
                               msg="Final loss differs too much from PyTorch model in the classification case.")
        
        



class TestUpdateBetaWithOptax(unittest.TestCase):
    
    def setUp(self):
        self.beta = jnp.array([1.0, 2.0])
        self.gradients = jnp.array([0.1, 0.2])
        self.iter = 1
        self.learning_rate = 0.01

    def test_constant_update(self):
        updater = Update_Beta()
        updater.constant(eta=self.learning_rate)
        new_beta = updater(self.beta, self.gradients)

        # Using SGD with a fixed learning rate in Optax for comparison
        optax_optimizer = optax.sgd(learning_rate=self.learning_rate)
        optax_state = optax_optimizer.init(self.beta)
        updates, _ = optax_optimizer.update(self.gradients, optax_state, self.beta)
        optax_beta = self.beta + updates

        self.assertTrue(jnp.allclose(new_beta, optax_beta), "Constant update does not match Optax")

    def test_momentum_update(self):
        updater = Update_Beta()
        updater.momentum_based(eta=self.learning_rate, gamma=0.9)
        new_beta = updater(self.beta, self.gradients)

        # Using Momentum in Optax for comparison
        optax_optimizer = optax.sgd(learning_rate=self.learning_rate, momentum=0.9)
        optax_state = optax_optimizer.init(self.beta)
        updates, _ = optax_optimizer.update(self.gradients, optax_state, self.beta)
        optax_beta = self.beta + updates

        self.assertTrue(jnp.allclose(new_beta, optax_beta), "Momentum update does not match Optax")

    def test_adagrad_update(self):
        updater = Update_Beta()
        updater.adagrad(eta=self.learning_rate)
        # Perform a single update with the custom Adagrad
        new_beta = updater(self.beta, self.gradients)

        # Using Adagrad in Optax for comparison
        optax_optimizer = optax.adagrad(learning_rate=self.learning_rate)
        optax_state = optax_optimizer.init(self.beta)
        updates, _ = optax_optimizer.update(self.gradients, optax_state, self.beta)
        optax_beta = self.beta + updates

        # Assert numerical equivalence between custom and Optax implementations
        self.assertTrue(jnp.allclose(new_beta, optax_beta, atol=1e-2), "Adagrad update does not match Optax ")
        ### MIA: check this, have a loose tolerance for now


    def test_adam_update(self):
        updater = Update_Beta()
        updater.adam(eta=self.learning_rate, epsilon=1e-8, b1=0.9, b2=0.999)
        new_beta = updater(self.beta, self.gradients, self.iter)

        # Using Adam in Optax for comparison
        optax_optimizer = optax.adam(learning_rate=self.learning_rate)
        optax_state = optax_optimizer.init(self.beta)
        updates, _ = optax_optimizer.update(self.gradients, optax_state, self.beta)
        optax_beta = self.beta + updates

        self.assertTrue(jnp.allclose(new_beta, optax_beta, atol=1e-6), "Adam update does not match Optax")

    def test_rmsprop_update(self):
        updater = Update_Beta()
        updater.rmsprop(eta=self.learning_rate, epsilon=1e-8, b=0.9)
        new_beta = updater(self.beta, self.gradients)

        # Using RMSProp in Optax for comparison
        optax_optimizer = optax.rmsprop(learning_rate=self.learning_rate, decay=0.9, eps=1e-8)
        optax_state = optax_optimizer.init(self.beta)
        updates, _ = optax_optimizer.update(self.gradients, optax_state, self.beta)
        optax_beta = self.beta + updates

        self.assertTrue(jnp.allclose(new_beta, optax_beta), "RMSprop update does not match Optax")

class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.z = jnp.array([1.0, -1.0, 2.0, -2.0])
        self.predictions = jnp.array([[0.3, 0.7], [0.8, 0.2]])
        self.targets = jnp.array([[0, 1], [1, 0]])
    
    def test_relu(self):
        # ReLU test
        expected = jax_nn.relu(self.z)
        actual = ReLU(self.z)
        self.assertTrue(jnp.allclose(actual, expected), "ReLU function mismatch")

    def test_relu_derivative(self):
        # ReLU derivative test with element-wise grad computation
        expected = jax.vmap(grad(jax_nn.relu))(self.z)
        actual = ReLU_der(self.z)
        self.assertTrue(jnp.allclose(actual, expected), "ReLU derivative mismatch")

    def test_leaky_relu(self):
        # Leaky ReLU test against PyTorch's LeakyReLU
        torch_leaky_relu = torch.nn.LeakyReLU(0.01)
        expected = torch_leaky_relu(torch.tensor(self.z)).numpy()
        actual = leaky_ReLU(self.z)
        self.assertTrue(jnp.allclose(actual, expected), "Leaky ReLU function mismatch")

    def test_leaky_relu(self):
        # Convert self.z to a NumPy array for PyTorch compatibility
        torch_leaky_relu = torch.nn.LeakyReLU(0.01)
        expected = torch_leaky_relu(torch.tensor(np.array(self.z))).numpy()
        actual = leaky_ReLU(self.z)
        self.assertTrue(jnp.allclose(actual, expected), "Leaky ReLU function mismatch")

    def test_sigmoid(self):
        # Sigmoid test
        expected = jax_nn.sigmoid(self.z)
        actual = sigmoid(self.z)
        self.assertTrue(jnp.allclose(actual, expected), "Sigmoid function mismatch")

    def test_sigmoid_derivative(self):
        # Sigmoid derivative test
        expected = jax.vmap(grad(jax_nn.sigmoid))(self.z)
        actual = sigmoid_der(self.z)
        self.assertTrue(jnp.allclose(actual, expected), "Sigmoid derivative mismatch")

    def test_softmax(self):
        # Softmax test for multi-dimensional input
        logits = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        expected = jax_nn.softmax(logits, axis=1)
        actual = softmax(logits)
        self.assertTrue(jnp.allclose(actual, expected), "Softmax function mismatch")

    def test_accuracy(self):
        # Binary accuracy comparison
        predictions = jnp.array([1, 0, 1, 1])
        targets = jnp.array([1, 0, 0, 1])
        expected = accuracy_score(targets, predictions)
        actual = accuracy(predictions, targets)
        self.assertAlmostEqual(actual, expected, msg="Binary accuracy mismatch")

    def test_accuracy_one_hot(self):
        # One-hot accuracy comparison
        predictions = jnp.array([[0.1, 0.9], [0.8, 0.2]])
        targets = jnp.array([[0, 1], [1, 0]])
        expected = accuracy_score(jnp.argmax(targets, axis=1), jnp.argmax(predictions, axis=1))
        actual = accuracy_one_hot(predictions, targets)
        self.assertAlmostEqual(actual, expected, msg="One-hot accuracy mismatch")

    def test_r2_score(self):
        predictions = jnp.array([3.0, -0.5, 2.0, 7.0])
        targets = jnp.array([2.5, 0.0, 2.0, 8.0])
        expected = r2_score(targets, predictions)
        actual = r_2(predictions, targets)
        self.assertAlmostEqual(actual, expected, msg="R2 score mismatch")

    def test_mse(self):
        # Mean squared error comparison
        predictions = jnp.array([3.0, -0.5, 2.0, 7.0])
        targets = jnp.array([2.5, 0.0, 2.0, 8.0])
        expected = mean_squared_error(targets, predictions)
        actual = mse(targets, predictions)
        self.assertAlmostEqual(actual, expected, msg="MSE mismatch")

    def test_cross_entropy_loss(self):
        # Cross-entropy loss comparison
        logits = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        targets = jnp.array([[0, 0, 1], [1, 0, 0]])

        # Using log_softmax for stability in cross-entropy computation
        softmax_preds = jax_nn.softmax(logits, axis=1)
        expected = -jnp.mean(jnp.sum(targets * jnp.log(softmax_preds + 1e-15), axis=1))
        actual = cross_entropy(logits, targets)

        self.assertTrue(jnp.allclose(actual, expected), "Cross-entropy loss mismatch")

def test_binary_cross_entropy(self):

    predictions = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32)
    target_tensor = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    criterion = nn.BCELoss()
    expected = criterion(predictions, target_tensor).item()
    actual = binary_cross_entropy(predictions.detach().numpy(), target_tensor.detach().numpy())
    self.assertTrue(np.allclose(actual, expected), "Binary cross-entropy loss mismatch")




if __name__ == "__main__":
    unittest.main()
