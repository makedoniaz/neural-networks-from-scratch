from .base_layer import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, prob):
        self.prob = prob
        self.mask = None

    def forward(self, X, training=True):
        if not training:
            return X

        self.mask = np.random.rand(*X.shape) > self.prob
        out = X * self.mask / (1 - self.prob)

        return out

    def backward(self, dout):
        return dout * self.mask / (1 - self.prob)