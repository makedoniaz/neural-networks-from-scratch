import numpy as np
from .base_layer import Layer
from parameter import Parameter

class Linear(Layer):
    def __init__(self, input_size, output_size, std=1e-3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.W = std * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)

        self.W = Parameter(
            std * np.random.randn(input_size, output_size),
            weight_decay=True
        )

        self.b = Parameter(
            np.zeros(output_size),
            weight_decay=False
        )

    def forward(self, X: np.ndarray):
        self.cache = X
        out = X.dot(self.W.data) + self.b.data

        return out

    def backward(self, dout: np.ndarray):
        X = self.cache

        self.W.grad = X.T.dot(dout)
        self.b.grad = np.sum(dout, axis=0)

        dX = dout.dot(self.W.data.T)

        return dX
    
    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_size} -> {self.output_size})"