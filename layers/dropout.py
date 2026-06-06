from .base_layer import Layer
import numpy as np

class Dropout(Layer):
    """Dropout regularization layer.

    Randomly deactivates neurons during training to prevent overfitting.
    Has no effect during inference.

    Args:
        prob: Dropout probability (fraction of units to drop). Typically 0.5.
    """

    def __init__(self, prob: float) -> None:
        super().__init__()
        self.prob = prob
        self.mask = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return X

        self.mask = np.random.rand(*X.shape) > self.prob
        out = X * self.mask / (1 - self.prob)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask / (1 - self.prob)