import os.path
import pickle

import numpy as np
# Only for DataLoader
from torch.utils.data.dataloader import DataLoader
# Only for MNIST dataset.
from torchvision import datasets, transforms

from linear import Linear


def train(model, lr, nb_epoch, data):
    losses = []
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
                    layer.update_params(lr)
        loss = running_loss / num_inputs
        losses.append(loss)
        print(f'Epoch {epoch + 1}/{nb_epoch}: loss = {loss}')
    return range(0, nb_epoch), losses


def validate(model, data):
    running_loss = 0.
    num_inputs = 0
    for mini_batch in data:
        inputs, targets = mini_batch
        num_inputs += inputs.shape[0]
        running_loss += model.loss(inputs, targets).sum()
    loss = running_loss / num_inputs
    print(f'Loss = {loss}')
    return loss


def load_minibatches(train=True, size="small", file_name=None, save=True, batch_size=1):
    tsfms = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('.', train=True, download=True, transform=tsfms)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_set = datasets.MNIST('.', train=False, download=True, transform=tsfms)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    loader = train_loader if train else test_loader

    if save and file_name is not None:
        if os.path.exists(file_name):
            with open(file_name, mode='rb') as data_file:
                data = pickle.load(data_file)
        else:
            data = get_data(loader, size)
            with open(file_name, mode='xb') as data_file:
                pickle.dump(data, data_file)
    else:
        data = get_data(loader, size)
    return data


def get_data(loader, size):
    data = []
    it = 0
    for mb in loader:
        if size == "small" and np.mod(it, 8) != 0:
            it += 1
            continue
        else:
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


def run(net, net_opt, data_train_file, data_test_file, alfa=0.001, test=False):
    val, val_opt = None, None
    data_train = load_minibatches(file_name=data_train_file, batch_size=8)
    data_test = load_minibatches(file_name=data_test_file, train=False, batch_size=8)
    _, y = train(net, alfa, 1, data_train)
    _, y_opt = train(net_opt, alfa, 1, data_train)

    if test:
        val = validate(net, data_test)
        val_opt = validate(net_opt, data_test)

    return y, y_opt, val, val_opt
