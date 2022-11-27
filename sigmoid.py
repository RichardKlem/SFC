import numpy as np


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        x = np.clip(x, a_min=None, a_max=709)
        self.y = np.exp(x) / (1. + np.exp(x))
        return self.y

    def backward(self, grad):
        return grad * self.y * (1. - self.y)
