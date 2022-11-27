import os.path
import pickle

from linear import Linear
from model import Model


def train(model: Model, alpha, nb_epoch, data):
    losses = []
    loss = 0.
    num_inputs = 0
    for mini_batch in data:
        inputs, targets = mini_batch
        num_inputs += inputs.shape[0]
        loss += model.loss(inputs, targets).sum()
        model.backward()
        for layer in model.layers:
            # Only linear layers have trainable params.
            if type(layer) == Linear:
                layer.update_params(alpha)
    final_loss = loss / num_inputs
    losses.append(final_loss)
    print(
        f'{str(model.layers[0].opt).capitalize().ljust(7)}{" - Train Loss    = ".ljust(21)}{final_loss}')
    return range(0, nb_epoch), losses


def validate(model, data):
    loss = 0.
    num_inputs = 0
    for data_input in data:
        inputs, targets = data_input
        num_inputs += inputs.shape[0]
        loss += model.loss(inputs, targets).sum()
    final_loss = loss / num_inputs
    print(
        f'{str(model.layers[0].opt).capitalize().ljust(7)}{" - Evaluation Loss = ".ljust(21)}{final_loss}')
    return final_loss


def load_minibatches(file_name=None):
    if file_name is not None:
        if os.path.exists(file_name):
            with open(file_name, mode='rb') as data_file:
                data = pickle.load(data_file)
        else:
            print("ERROR: cannot find data file.")
            exit(1)
    else:
        print("ERROR: cannot find data file.")
        exit(1)
    return data


def run(net, net_opt, data_train_file, data_test_file, alfa=0.001, test=False):
    y, y_opt, val, val_opt = None, None, None, None
    if not test:
        data_train = load_minibatches(file_name=data_train_file)
        _, y = train(net, alfa, 1, data_train)
        _, y_opt = train(net_opt, alfa, 1, data_train)
    else:
        data_test = load_minibatches(file_name=data_test_file)
        val = validate(net, data_test)
        val_opt = validate(net_opt, data_test)
    return y, y_opt, val, val_opt
