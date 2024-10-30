import jax.numpy as jnp
import numpy as np  
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score



def ReLU(z):
    return jnp.where(z > 0, z, 1e-15)

def ReLU_der(z):
    return jnp.where(z > 0, 1, 1e-15)  

def leaky_ReLU(z):
    return jnp.where(z > 0, z, 0.01 * z)

def leaky_ReLU_der(z):
    return jnp.where(z > 0, 1, 0.01)

def sigmoid(z):
    z = jnp.clip(z, -30, 30)  
    return 1 / (1 + jnp.exp(-z))

def sigmoid_der(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def softmax(z):
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))  # Adjust axis to 1 for row-wise stability
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

def accuracy(predictions, targets):
    predictions = jnp.where(predictions >= 0.5, 1, 0)
    return accuracy_score(predictions, targets) # Use only for final evaluations, not within JAX-traced functions

def accuracy_one_hot(predictions, targets):
    # Convert predictions to class labels
    predicted_labels = jnp.argmax(predictions, axis=1)
    
    # If targets are in one-hot encoding, convert them to class labels too
    if targets.ndim > 1:
        target_labels = jnp.argmax(targets, axis=1)
    else:
        target_labels = targets
    
    return accuracy_score(predicted_labels, target_labels)


def r_2(predictions, targets):

    target_mean = jnp.mean(targets)
    ss_total = jnp.sum((targets - target_mean) ** 2)
    ss_res = jnp.sum((targets - predictions) ** 2)

    return 1 - ss_res / ss_total

def mse(targets, predictions):
    mse_value = np.mean((predictions - targets) ** 2)
    return mse_value

def cross_entropy(logits, targets):

    exp_logits = jnp.exp(logits - jnp.max(logits, axis=1, keepdims=True))
    softmax_preds = exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)

    return -jnp.mean(jnp.sum(targets * jnp.log(softmax_preds + 1e-15), axis=1))

def binary_cross_entropy(predictions, targets):
    epsilon = 1e-10  # Small constant to avoid log(0)
    predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)  # Ensure stability
    bce = -jnp.mean(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions))
    return bce

def recall(predictions, targets):
    # turn to binary values
    predictions = jnp.where(predictions >= 0.5, 1, 0)
    true_positives = jnp.sum(predictions * targets)
    actual_positives = jnp.sum(targets)
    recall_val = true_positives / (actual_positives + 1e-15)
    # check for values outside the range [0, 1]
    if recall_val < 0 or recall_val > 1:
        raise ValueError(f"Recall value outside the range [0, 1]: {recall_val}")
    return recall_val

def precision(predictions, targets):
    true_positives = jnp.sum(predictions * targets)
    predicted_positives = jnp.sum(predictions)
    return true_positives / (predicted_positives + 1e-15)