import numpy as np

from core.xavier import xavier


class Linear:

    def __init__(self, in_features, out_features):
        self.W = xavier(in_features, out_features)
        self.b = np.zeros((1, out_features))
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.grad_W = self.input.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.W.T
        return grad_input
