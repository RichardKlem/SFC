import numpy as np


class Softmax:
    def __init__(self):
        self.old_y = None

    def forward(self, x):
        self.old_y = np.exp(x) / np.exp(x).sum(axis=1)[:, None]
        return self.old_y

    def backward(self, grad):
        return self.old_y * (grad - (grad * self.old_y).sum(axis=1)[:, None])
