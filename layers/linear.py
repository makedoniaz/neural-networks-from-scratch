import numpy as np
from .base_layer import Layer

class Linear(Layer):
    def __init__(self, input_size, output_size, std=1e-3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.W = std * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)

        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray):
        self.cache = X
        out = X.dot(self.W) + self.b

        return out

    def backward(self, dout: np.ndarray):
        X = self.cache

        self.dW = X.T.dot(dout)
        self.db = np.sum(dout, axis=0)
        dX = dout.dot(self.W.T)

        return dX
    
    def parameters(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_size} -> {self.output_size})"