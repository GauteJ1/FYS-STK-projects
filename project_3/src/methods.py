from sklearn.metrics import accuracy_score
import autograd.numpy as np  # Ensure autograd.numpy is used

# Activation Functions
def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def leaky_ReLU(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, 0.01 * z)

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))

def softmax(z: np.ndarray) -> np.ndarray:
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z: np.ndarray) -> np.ndarray:
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def identity(z: np.ndarray) -> np.ndarray:
    return z


# Derivatives of Activation Functions
def ReLU_der(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)

def leaky_ReLU_der(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64) + 0.01 * (z <= 0).astype(np.float64)

def sigmoid_der(z: np.ndarray) -> np.ndarray:
    sig = sigmoid(z)
    return sig * (1 - sig)

def softmax_der(z: np.ndarray) -> np.ndarray:
    return softmax(z) * (1 - softmax(z))

def tanh_der(z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z) ** 2

def identity_der(z: np.ndarray) -> int:
    return 1


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.where(predictions >= 0.5, 1, 0)
    return accuracy_score(
        predictions, targets
    )  # Use only for final evaluations, not within JAX-traced functions


def accuracy_one_hot(predictions: np.ndarray, targets: np.ndarray) -> float:

    predicted_labels = np.argmax(predictions, axis=1)

    if targets.ndim > 1:
        target_labels = np.argmax(targets, axis=1)
    else:
        target_labels = targets

    return accuracy_score(predicted_labels, target_labels)


def r_2(predictions: np.ndarray, targets: np.ndarray) -> float:

    target_mean = np.mean(targets)
    ss_total = np.sum((targets - target_mean) ** 2)
    ss_res = np.sum((targets - predictions) ** 2)

    return 1 - ss_res / ss_total


def mse(targets: np.ndarray, predictions: np.ndarray) -> float:
    mse_value = np.mean((predictions - targets) ** 2)
    return mse_value


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_preds = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return -np.mean(np.sum(targets * np.log(softmax_preds + 1e-15), axis=1))


def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> float:

    epsilon = 1e-6
    predictions = np.clip(predictions, epsilon, 1 - epsilon)  # to avoid nan

    bce = -np.mean(
        targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)
    )

    if np.isnan(bce):
        raise ValueError("NaN encountered in binary cross-entropy")

    return bce


def recall(predictions: np.ndarray, targets: np.ndarray) -> float:

    predictions = np.where(predictions >= 0.5, 1, 0)
    true_positives = np.sum(predictions * targets)
    false_negatives = np.sum((1 - predictions) * targets)

    recall_val = true_positives / (true_positives + false_negatives + 1e-15)
    if recall_val < 0 or recall_val > 1:
        raise ValueError(f"Recall value outside the range [0, 1]: {recall_val}")

    return recall_val


def precision(predictions: np.ndarray, targets: np.ndarray) -> float:

    predictions = np.where(predictions >= 0.5, 1, 0)
    true_positives = np.sum(predictions * targets)
    false_positives = np.sum(predictions * (1 - targets))

    precision_val = true_positives / (true_positives + false_positives + 1e-15)

    if precision_val < 0 or precision_val > 1:
        raise ValueError(f"Precision value outside the range [0, 1]: {precision_val}")

    return precision_val


def f1score(predictions: np.ndarray, targets: np.ndarray) -> float:
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
