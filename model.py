class Model:
    def __init__(self, layers, opt, cost):
        self.layers = layers
        self.cost = cost

        for layer in layers:
            layer.opt = opt

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        return self.cost.forward(self.forward(x), y)

    def backward(self):
        grad = self.cost.backward()
        for i in reversed(range(len(self.layers))):
            grad = self.layers[i].backward(grad)
