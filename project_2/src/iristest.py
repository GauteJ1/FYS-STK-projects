from sklearn import datasets
import numpy as np


from neural_network import NeuralNetwork
from methods import *

def test_iris():

    np.random.seed(4155) # FYS-STK4155 

    iris = datasets.load_iris()

    inputs = iris.data
    targets = np.zeros((len(iris.data), 3))
    for i, t in enumerate(iris.target):
        targets[i, t] = 1

    input_size = 4 # number of input features
    output_size = 3 # number of output features
    network_shape = [input_size, 8, output_size] 
    activation_funcs = ["sigmoid", "ReLU"]
    cost_func = mse

    model = NeuralNetwork(network_shape, activation_funcs, mse)

    _, acc, loss = model.train_network(inputs, targets, epochs=50, learning_rate=0.05, batch=10)

    assert loss[-1] < 0.3, "Iris test failed"

    print("Iris test passed")