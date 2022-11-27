import numpy as np


class Linear:
    def __init__(self, n_in, n_out):
        self.grad_w = None
        self.grad_b = None
        self.y = None
        self.weights = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        self.biases = np.zeros(n_out)
        self.opt = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 10**-8
        self.velocity1 = 0.
        self.velocity2 = 0.
        self.momentum = np.zeros((n_in, n_out))

    def forward(self, x):
        self.y = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad):
        self.grad_b = grad.mean(axis=0)
        self.grad_w = (
            np.matmul(np.expand_dims(self.y, axis=2), np.expand_dims(grad, axis=1))).mean(axis=0)
        return np.dot(grad, self.weights.transpose())

    def update_params(self, alpha):
        if self.opt in ["amsgrad", "adam"]:
            self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * self.grad_w
            self.velocity1 = self.beta2 * self.velocity1 + (1 - self.beta2) * (self.grad_w**2)
            if self.opt == "amsgrad":
                self.velocity2 = np.maximum(self.velocity2, self.velocity1)
            else:
                self.velocity2 = self.velocity1 / (1 - self.beta2)
            self.weights -= alpha * (1 / (np.sqrt(self.velocity2) + self.eps)) * self.momentum
        else:
            self.weights -= alpha * self.grad_w

        self.biases -= alpha * self.grad_b
