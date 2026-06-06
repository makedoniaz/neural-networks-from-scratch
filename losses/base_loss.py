from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    """Abstract base class for loss functions.

    Defines the interface for computing loss and gradients during training.

    Args:
        reduction: How to reduce loss over batch ('mean', 'sum', or 'none'). Defaults to 'mean'.
    """

    def __init__(self, reduction: str = 'mean') -> None:
        self.cache = None
        self.reduction = reduction
    
    @abstractmethod
    def forward(self, y_out: np.ndarray, y_truth: np.ndarray) -> float | np.ndarray:
        """Compute loss between predictions and targets.

        Args:
            y_out: Model predictions.
            y_truth: Ground truth targets.

        Returns:
            Scalar loss value (or array if reduction='none').
        """
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        """Compute gradient of loss with respect to predictions.

        Returns:
            Gradient array.
        """
        pass

    def reduce(self, loss: np.ndarray) -> float | np.ndarray:
        """Apply reduction operation to per-sample losses.

        Args:
            loss: Per-sample loss values.

        Returns:
            Reduced loss (scalar or array).
        """
        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        elif self.reduction == 'none':
            return loss
        
    def scale_grad(self, grad: np.ndarray) -> np.ndarray:
        """Scale gradient according to reduction strategy.

        Args:
            grad: Gradient array.

        Returns:
            Scaled gradient.
        """
        if self.reduction == 'mean':
            grad /= grad.size

        return grad
