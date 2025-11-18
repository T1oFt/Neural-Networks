import numpy as np

from core.sigmoid import sigmoid, sigmoid_derivative
from core.linear import Linear
from core.loss import bce_derivative


class MLP:
    def __init__(self, input_size, hidden1, hidden2):
        self.fc1 = Linear(input_size, hidden1)
        self.fc2 = Linear(hidden1, hidden2)
        self.fc3 = Linear(hidden2, 1)

    def forward(self, x):
        self.z1 = self.fc1.forward(x)
        a1 = sigmoid(self.z1)
        self.z2 = self.fc2.forward(a1)
        a2 = sigmoid(self.z2)
        self.z3 = self.fc3.forward(a2)
        y_pred = sigmoid(self.z3)
        return y_pred

    def backward(self, y_true, y_pred):
        dL_dz3 = bce_derivative(y_pred, y_true) * sigmoid_derivative(self.z3)

        dL_da2 = self.fc3.backward(dL_dz3)
        dL_dz2 = dL_da2 * sigmoid_derivative(self.z2)
        dL_da1 = self.fc2.backward(dL_dz2) 
        dL_dz1 = dL_da1 * sigmoid_derivative(self.z1)
        self.fc1.backward(dL_dz1)

    def update(self, lr):
        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b

    def zero_grad(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.grad_W = np.zeros_like(layer.W)
            layer.grad_b = np.zeros_like(layer.b)
