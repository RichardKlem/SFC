import numpy as np  # import numpy library


class SigmoidLayer:
    def __init__(self, shape):
        self.A = np.zeros(shape)  # create space for the resultant activations

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))  # compute activations

    def backward(self, upstream_grad):
        self.dZ = upstream_grad * self.A * (1 - self.A)
