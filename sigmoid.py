import numpy as np  # import numpy library


class SigmoidLayer:
    def __init__(self, shape=None):
        self.dx = None
        self.old_y = np.zeros(shape) if shape else None

    def forward(self, x):
        self.old_y = np.exp(x) / (1. + np.exp(x))
        return self.old_y

    def backward(self, upstream_grad):
        self.dx = upstream_grad * self.old_y * (1. - self.old_y)
        return self.dx
