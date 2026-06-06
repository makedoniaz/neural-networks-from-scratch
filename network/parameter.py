import numpy as np

class Parameter:
    """Trainable parameter container.

    Stores parameter data, gradients, and regularization settings.

    Args:
        data: Parameter values (typically initialized weights or biases).
        weight_decay: Whether to apply weight decay regularization. Defaults to True.
    """

    def __init__(self, data: np.ndarray, weight_decay: bool = True) -> None:
        self.data = data
        self.grad = None
        self.weight_decay = weight_decay