import jax.numpy as jnp
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


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

# derivation of softmax 
def softmax_der(z):
    return softmax(z) * (1 - softmax(z))

def accuracy(y, y_pred):
    return accuracy_score(y, y_pred)  # Use only for final evaluations, not within JAX-traced functions

def accuracy_one_hot(predictions, targets):
    # Convert predictions to class labels
    predicted_labels = jnp.argmax(predictions, axis=1)
    
    # If targets are in one-hot encoding, convert them to class labels too
    if targets.ndim > 1:
        target_labels = jnp.argmax(targets, axis=1)
    else:
        target_labels = targets
    
    return accuracy_score(predicted_labels, target_labels)

def r_2(y, y_pred):
    return r2_score(y.flatten(), y_pred.flatten())

def cross_entropy(y_pred, y):
    # Average to ensure scalar output
    return -jnp.mean(jnp.sum(y * jnp.log(y_pred), axis=1))

def mse(y_pred, y):
    return jnp.mean((y - y_pred) ** 2)

def mse_der(predict, target):
    return (2 / len(predict)) * (predict - target)

def identity(z):
    return z

def identity_der(z):
    return 1

def binary_cross_entropy(y_true, y_pred):
    y_pred = jnp.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0) by clipping
    bce = -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))
    return bce