from .base_layer import Layer
import numpy as np

class Flatten(Layer):
    """Flattens multi-dimensional input into 2D (batch_size, features).

    Preserves batch dimension and flattens all other dimensions.
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.cache = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        original_shape = self.cache
        return dout.reshape(original_shape)