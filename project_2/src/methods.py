import jax.numpy as jnp
import numpy as np
from sklearn.metrics import accuracy_score


def ReLU(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(z > 0, z, 1e-15)


def ReLU_der(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(z > 0, 1, 1e-15)


def leaky_ReLU(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(z > 0, z, 0.01 * z)


def leaky_ReLU_der(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(z > 0, 1, 0.01)


def sigmoid(z: jnp.ndarray) -> jnp.ndarray:
    z = jnp.clip(z, -30, 30)
    return 1 / (1 + jnp.exp(-z))


def sigmoid_der(z: jnp.ndarray) -> jnp.ndarray:
    sig = sigmoid(z)
    return sig * (1 - sig)


def softmax(z: jnp.ndarray) -> jnp.ndarray:
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)


def softmax_vec(z: jnp.ndarray) -> jnp.ndarray:
    e_z = jnp.exp(z - jnp.max(z))
    return e_z / jnp.sum(e_z)


def softmax_der(z: jnp.ndarray) -> jnp.ndarray:
    return softmax(z) * (1 - softmax(z))


def identity(z: jnp.ndarray) -> jnp.ndarray:
    return z


def identity_der(z: jnp.ndarray) -> int:
    return 1


def accuracy(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    predictions = jnp.where(predictions >= 0.5, 1, 0)
    return accuracy_score(
        predictions, targets
    )  # Use only for final evaluations, not within JAX-traced functions


def accuracy_one_hot(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:

    predicted_labels = jnp.argmax(predictions, axis=1)

    if targets.ndim > 1:
        target_labels = jnp.argmax(targets, axis=1)
    else:
        target_labels = targets

    return accuracy_score(predicted_labels, target_labels)


def r_2(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:

    target_mean = jnp.mean(targets)
    ss_total = jnp.sum((targets - target_mean) ** 2)
    ss_res = jnp.sum((targets - predictions) ** 2)

    return 1 - ss_res / ss_total


def mse(targets: jnp.ndarray, predictions: jnp.ndarray) -> float:
    mse_value = np.mean((predictions - targets) ** 2)
    return mse_value


def cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> float:

    exp_logits = jnp.exp(logits - jnp.max(logits, axis=1, keepdims=True))
    softmax_preds = exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)

    return -jnp.mean(jnp.sum(targets * jnp.log(softmax_preds + 1e-15), axis=1))


def binary_cross_entropy(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:

    epsilon = 1e-6
    predictions = jnp.clip(predictions, epsilon, 1 - epsilon)  # to avoid nan

    bce = -jnp.mean(
        targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions)
    )

    if jnp.isnan(bce):
        raise ValueError("NaN encountered in binary cross-entropy")

    return bce


def recall(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:

    predictions = jnp.where(predictions >= 0.5, 1, 0)
    true_positives = jnp.sum(predictions * targets)
    false_negatives = jnp.sum((1 - predictions) * targets)

    recall_val = true_positives / (true_positives + false_negatives + 1e-15)
    if recall_val < 0 or recall_val > 1:
        raise ValueError(f"Recall value outside the range [0, 1]: {recall_val}")

    return recall_val


def precision(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:

    predictions = jnp.where(predictions >= 0.5, 1, 0)
    true_positives = jnp.sum(predictions * targets)
    false_positives = jnp.sum(predictions * (1 - targets))

    precision_val = true_positives / (true_positives + false_positives + 1e-15)

    if precision_val < 0 or precision_val > 1:
        raise ValueError(f"Precision value outside the range [0, 1]: {precision_val}")

    return precision_val


def f1score(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    precision_val = precision(predictions, targets)
    recall_val = recall(predictions, targets)

    if not (0 <= precision_val <= 1):
        raise ValueError(f"Precision value outside the range [0, 1]: {precision_val}")
    if not (0 <= recall_val <= 1):
        raise ValueError(f"Recall value outside the range [0, 1]: {recall_val}")

    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-15)

    if not (0 <= f1 <= 1):
        raise ValueError(f"F1 score value outside the range [0, 1]: {f1}")

    return f1
