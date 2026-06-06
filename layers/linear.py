import numpy as np
from .base_layer import Layer
from network.parameter import Parameter

class Linear(Layer):
    """Fully connected (dense) layer.

    Applies linear transformation: output = input @ W + b.

    Args:
        input_size: Number of input features.
        output_size: Number of output features.
        initialization: Weight initialization strategy ('he', 'xavier', or default). Defaults to 'he'.
    """

    def __init__(
        self, input_size: int, output_size: int, initialization: str = "he"
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        if initialization == "he":
            std = np.sqrt(2 / input_size)
        elif initialization == "xavier":
            std = np.sqrt(1 / input_size)
        else:
            std = 1e-3

        self.W = Parameter(
            np.random.randn(input_size, output_size) * std,
            weight_decay=True
        )

        self.b = Parameter(
            np.zeros(output_size),
            weight_decay=False
        )

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.cache = X
        out = X.dot(self.W.data) + self.b.data
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        X = self.cache
        self.W.grad = X.T.dot(dout)
        self.b.grad = np.sum(dout, axis=0)
        dX = dout.dot(self.W.data.T)
        return dX
    
    def parameters(self) -> list:
        return [self.W, self.b]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.input_size} -> {self.output_size})"