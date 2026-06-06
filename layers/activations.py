import numpy as np
from .base_layer import Layer

class ReLU(Layer):
    """Rectified Linear Unit activation layer.

    Applies ReLU: output = max(0, input).
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.maximum(X, 0)
        self.cache = X
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (self.cache > 0)
    
class LeakyReLU(Layer):
    """Leaky Rectified Linear Unit activation layer.

    Applies LeakyReLU: output = max(alpha * input, input).

    Args:
        slope: Negative slope for values less than zero. Typically 0.01 or 0.2.
    """

    def __init__(self, slope: float) -> None:
        self.slope = slope
        super().__init__()
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.maximum(X, self.slope * X)
        self.cache = X
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        X = self.cache
        grad = np.ones_like(X)
        grad[X <= 0] = self.slope
        return dout * grad

class Sigmoid(Layer):
    """Sigmoid activation layer.

    Applies sigmoid: output = 1 / (1 + exp(-input)).
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-X))
        self.cache = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        activation = self.cache
        return dout * activation * (1 - activation)

class Tanh(Layer):
    """Hyperbolic tangent activation layer.

    Applies tanh: output = tanh(input).
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.tanh(X)
        self.cache = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        activation = self.cache
        return dout * (1 - activation * activation)