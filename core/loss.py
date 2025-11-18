import numpy as np


def bce(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_derivative(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.size
