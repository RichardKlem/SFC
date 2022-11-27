import numpy as np


class Softmax:
    def __init__(self):
        self.y = None

    def forward(self, x):
        x = np.clip(x, a_min=None, a_max=709)
        self.y = np.exp(x) / np.expand_dims(np.exp(x).sum(axis=1), axis=1)
        return self.y

    def backward(self, grad):
        return self.y * (grad - np.expand_dims((grad * self.y).sum(axis=1), axis=1))
