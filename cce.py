import numpy as np
from keras.losses import CategoricalCrossentropy

y_true = np.array([[0, 1, 0], [0, 0, 1]])
# y_true = np.array([0, 1, 0])
y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
# y_pred = np.array([0.05, 0.95, 0])
# Using 'auto'/'sum_over_batch_size' reduction type.
cce = CategoricalCrossentropy(reduction='none')
cce(y_true, y_pred).numpy()


def cross_E(y_true, y_pred):
    if not isinstance(y_true[0], np.ndarray):
        y_true = np.array([y_true])
    if not isinstance(y_pred[0], np.ndarray):
        y_pred = np.array([y_pred])
    return np.divide(-np.sum(y_true * np.log(y_pred + 10**-100)), y_true.shape[0])

cross_E(y_true, y_pred)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def compute_stable_bce_cost(Y, Z):
    """
    This function computes the "Stable" Binary Cross-Entropy(stable_bce) Cost and returns the Cost and its
    derivative w.r.t Z_last(the last linear node) .
    The Stable Binary Cross-Entropy Cost is defined as:
    => (1/m) * np.sum(max(Z,0) - ZY + log(1+exp(-|Z|)))
    Args:
        Y: labels of data
        Z: Values from the last linear node
    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    m = Y.shape[1]

    cost = (1/m) * np.sum(np.maximum(Z, 0) - Z*Y + np.log(1+ np.exp(- np.abs(Z))))
    dZ_last = (1/m) * ((1/(1+np.exp(- Z))) - Y)  # from Z computes the Sigmoid so P_hat - Y, where P_hat = sigma(Z)

    return cost, dZ_last