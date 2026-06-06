from abc import ABC, abstractmethod
import numpy as np


class Transform(ABC):
    """Abstract base class for data transformations."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the transformation to input data.

        Args:
            x: Input array to transform.

        Returns:
            Transformed array.
        """
        pass

class Compose(Transform):
    """Chain multiple transformations to apply sequentially.

    Args:
        transforms: List of Transform objects to apply in order.
    """

    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply all transformations in sequence.

        Args:
            x: Input array.

        Returns:
            Transformed array after applying all transforms.
        """
        for transform in self.transforms:
            x = transform(x)
        return x

class ToFloat(Transform):
    """Convert input array to float32 dtype."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Convert array to float32.

        Args:
            x: Input array.

        Returns:
            Array cast to np.float32.
        """
        return x.astype(np.float32)

class Normalize(Transform):
    """Normalize array by subtracting mean and dividing by standard deviation.

    Args:
        mean: Mean value for normalization. Defaults to 0.0.
        std: Standard deviation for normalization. Defaults to 1.0.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Normalize array using (x - mean) / std.

        Args:
            x: Input array.

        Returns:
            Normalized array.
        """
        return (x - self.mean) / self.std

class Scale(Transform):
    """Scale array by dividing by a scale factor.

    Args:
        scale: Divisor for scaling. Defaults to 255.0 (typical for uint8 image data).
    """

    def __init__(self, scale: float = 255.0) -> None:
        self.scale = scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Scale array by dividing by scale factor.

        Args:
            x: Input array.

        Returns:
            Scaled array (x / scale).
        """
        return x / self.scale