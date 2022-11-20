import numpy as np
from linear import LinearLayer
from sigmoid import SigmoidLayer
from cce import compute_stable_bce_cost

# define training constants
learning_rate = 1
number_of_epochs = 5000

np.random.seed(48) # set seed value so that the results are reproduceable
                   # (weights will now be initailzaed to the same pseudo-random numbers, each time)

# Our network architecture has the shape:
#                       (input)--> [Linear->Sigmoid] -->(output)


#------ LAYER-1 ----- define output layer that takes in training data
Z1 = LinearLayer(input_shape=X_train.shape, n_out=1, ini_type='plain')
A1 = SigmoidLayer(Z1.Z.shape)

costs = []  # initially empty list, this will store all the costs after a certain number of epochs

# Start training
for epoch in range(number_of_epochs):
    # ------------------------- forward-prop -------------------------
    Z1.forward(X_train)
    A1.forward(Z1.Z)

    # ---------------------- Compute Cost ----------------------------
    cost, dZ1 = compute_stable_bce_cost(Y_train, Z1.Z)
    # print and store Costs every 100 iterations and of the last iteration.
    if (epoch % 100) == 0 or epoch == number_of_epochs - 1:
        print("Cost at epoch#{}: {}".format(epoch, cost))
        costs.append(cost)

    # ------------------------- back-prop ----------------------------
    Z1.backward(dZ1)

    # ----------------------- Update weights and bias ----------------
    Z1.update_params(learning_rate=learning_rate)


