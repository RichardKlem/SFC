import numpy as np


class CrossEntropy:
    def __init__(self):
        self.old_y = None
        self.old_x = None

    def forward(self, x, y):
        self.old_x = np.maximum(x, 1e-8)
        self.old_y = y
        return (y * -np.log(self.old_x)).sum(axis=1)

    def backward(self):
        return -self.old_y / self.old_x
