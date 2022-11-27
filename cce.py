import numpy as np


class CrossEntropy:
    def __init__(self):
        self.y = None
        self.x = None

    def forward(self, x, y):
        self.x = np.maximum(x, 1e-8)
        self.y = y
        return (y * -np.log(self.x)).sum(axis=1)

    def backward(self):
        return -self.y / self.x
