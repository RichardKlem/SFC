from cce import CrossEntropy
from linear import Linear
from sigmoid import SigmoidLayer
from softmax import Softmax

# pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# To download the dataset for torchvision
import torchvision
from torchvision import datasets, transforms
# For plots
import matplotlib.pyplot as plt
import numpy as np


class Model:
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        return self.cost.forward(self.forward(x), y)

    def backward(self):
        grad = self.cost.backward()
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)


net = Model([Linear(784, 100), SigmoidLayer(), Linear(100, 10), Softmax()], CrossEntropy())


def train(model, lr, nb_epoch, data):
    for epoch in range(nb_epoch):
        running_loss = 0.
        num_inputs = 0
        for mini_batch in data:
            inputs, targets = mini_batch
            num_inputs += inputs.shape[0]
            # Forward pass + compute loss
            running_loss += model.loss(inputs, targets).sum()
            # Back propagation
            model.backward()
            # Update of the parameters
            for layer in model.layers:
                if type(layer) == Linear:
                    layer.weights -= lr * layer.grad_w
                    layer.biases -= lr * layer.grad_b
        print(f'Epoch {epoch + 1}/{nb_epoch}: loss = {running_loss / num_inputs}')


def load_minibatches(batch_size=64):
    tsfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trn_set = datasets.MNIST('.', train=True, download=True, transform=tsfms)
    trn_loader = torch.utils.data.DataLoader(trn_set, batch_size=batch_size, shuffle=True,
                                             num_workers=0)
    data = []
    it = 0
    for mb in trn_loader:
        if np.mod(it, 100) == 0:
            print(it)
        it += 1
        inputs_t, targets_t = mb
        inputs = np.zeros((inputs_t.size(0), 784))
        targets = np.zeros((inputs_t.size(0), 10))
        for i in range(0, inputs_t.size(0)):
            targets[i, targets_t[i]] = 1.
            for j in range(0, 28):
                for k in range(0, 28):
                    inputs[i, j * 28 + k] = inputs_t[i, 0, j, k]
        data.append((inputs, targets))
    return data


data = load_minibatches()

train(net, 1, 10, data)
