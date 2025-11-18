import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    x = np.clip(x, -500, 500)
    s = sigmoid(x)
    return s * (1 - s)
