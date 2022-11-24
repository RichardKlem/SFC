import numpy as np
from keras.losses import CategoricalCrossentropy

y_true = np.array([[0, 1, 0], [0, 0, 1]])
# y_true = np.array([0, 1, 0])
y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
# y_pred = np.array([0.05, 0.95, 0])
# Using 'auto'/'sum_over_batch_size' reduction type.
cce = CategoricalCrossentropy(reduction='none')
cce(y_true, y_pred).numpy()


class CrossEntropy:
    def __init__(self):
        self.old_y = None
        self.old_x = None

    def forward(self, x, y):
        self.old_x = x.clip(min=1e-8, max=None)
        self.old_y = y
        inner = np.where(y == 1, -np.log(self.old_x), 0)
        return (np.where(y == 1, -np.log(self.old_x), 0)).sum(axis=1)

    def backward(self):
        return np.where(self.old_y == 1, -1. / self.old_x, 0)
