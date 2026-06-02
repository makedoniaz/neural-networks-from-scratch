import numpy as np
from .base_layer import Layer

class ReLU(Layer):

    def forward(self, X: np.ndarray):
        out = np.maximum(X, 0)
        self.cache = X

        return out

    def backward(self, dout: np.ndarray):
        X = self.cache

        return dout * (self.cache > 0)
    
class LeakyReLU(Layer):
    def __init__(self, slope):
        self.slope = slope
        super().__init__()
    
    def forward(self, X: np.ndarray):
        out = np.maximum(X, self.slope * X)
        self.cache = X

        return out

    def backward(self, dout: np.ndarray):
        X = self.cache

        grad = np.ones_like(X)
        grad[X <= 0] = self.slope

        return dout * grad

class Sigmoid(Layer):

    def forward(self, X: np.ndarray):
        out = 1 / (1 + np.exp(-X))
        self.cache = out

        return out

    def backward(self, dout: np.ndarray):
        activation = self.cache

        return dout * activation * (1 - activation)

class Tanh(Layer):

    def forward(self, X: np.ndarray):
        out = np.tanh(X)
        self.cache = out

        return out

    def backward(self, dout: np.ndarray):
        activation = self.cache

        return dout * (1 - activation * activation)