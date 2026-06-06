from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    """Abstract base class for neural network layers.

    All layers must implement forward and backward methods for
    forward and backward propagation.
    """

    def __init__(self) -> None:
        self.cache = None
        self.training = True

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the layer.

        Args:
            X: Input array.

        Returns:
            Output array.
        """
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass (gradient computation).

        Args:
            dout: Gradient of loss with respect to layer output.

        Returns:
            Gradient of loss with respect to layer input.
        """
        pass

    def parameters(self) -> list:
        """Return trainable parameters of the layer.

        Returns:
            List of Parameter objects (empty for non-parametric layers).
        """
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"