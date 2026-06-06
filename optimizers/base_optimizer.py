from abc import ABC, abstractmethod

class Optimizer(ABC):
    """Abstract base class for optimization algorithms.

    Manages parameter updates using gradients computed during backpropagation,
    with optional weight decay regularization.

    Args:
        parameters: List of Parameter objects to optimize.
        learning_rate: Step size for parameter updates.
        weight_decay: L2 regularization coefficient. Defaults to 0.0.
    """

    def __init__(self, parameters: list, learning_rate: float, weight_decay: float = 0.0) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def step(self) -> None:
        """Apply optimizer update to all parameters based on their gradients."""
        for param in self.parameters:
            if param.grad is None:
                continue

            grad = param.grad

            if self.weight_decay != 0 and param.weight_decay:
                grad = grad + self.weight_decay * param.data

            self._update(param, grad)

    @abstractmethod
    def _update(self, param, grad) -> None:
        """Update a single parameter using computed gradient.

        Args:
            param: Parameter object to update.
            grad: Gradient with respect to the parameter.
        """
        pass
