import numpy as np
from .base_layer import Layer

class ReLU(Layer):
    def __init__(self):
        self.cache = None

    def forward(self, X: np.ndarray):
        out = np.maximum(X, 0)
        self.cache = X

        return out

    def backward(self, dout: np.ndarray):
        X = self.cache

        return dout * (self.cache > 0)
    
    def __repr__(self):
        return "ReLU()"