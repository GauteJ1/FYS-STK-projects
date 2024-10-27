import jax.numpy as jnp
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

### Activation functions and their derivatives ###


def ReLU(z):
    return jnp.where(z > 0, z, 1e-15)

def ReLU_der(z):
    return jnp.where(z > 0, 1, 1e-15)  

def sigmoid(z):
    z = jnp.clip(z, -30, 30)  
    return 1 / (1 + jnp.exp(-z))

def sigmoid_der(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def softmax(z):
    e_z = jnp.exp(z - jnp.max(z, axis=0, keepdims=True))  # for numerical stability
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    e_z = jnp.exp(z - jnp.max(z))  # for numerical stability
    return e_z / jnp.sum(e_z)

def softmax_der(z):
    return softmax(z) * (1 - softmax(z))

def identity(z):
    return z

def identity_der(z):
    return 1


### Accuracy functions ###

def accuracy(predictions, targets):
    return accuracy_score(predictions, targets)  # Use only for final evaluations, not within JAX-traced functions

def accuracy_one_hot(predictions, targets):
    # Convert predictions to class labels
    predicted_labels = jnp.argmax(predictions, axis=1)
    
    # If targets are in one-hot encoding, convert them to class labels too
    if targets.ndim > 1:
        target_labels = jnp.argmax(targets, axis=1)
    else:
        target_labels = targets
    
    return accuracy_score(predicted_labels, target_labels)

### Cost functions ###

def r_2(predictions, targets):
    return r2_score(predictions.flatten(), targets.flatten())

def mse(targets, predictions):
    return jnp.mean((predictions - targets) ** 2)