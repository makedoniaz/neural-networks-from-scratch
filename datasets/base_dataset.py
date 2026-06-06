from abc import ABC, abstractmethod
from utils.subset import Subset
import numpy as np

class Dataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Retrieve a sample by index.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Dictionary containing sample data and label.
        """
        pass

    def subset(self, indices: np.ndarray) -> Subset:
        """Create a subset of this dataset.

        Args:
            indices: Array of indices to include in the subset.

        Returns:
            A Subset object containing only the selected samples.
        """
        return Subset(self, indices)

class DummyDataset(Dataset):
    """Dummy dataset for testing, generating random data and labels.

    Args:
        num_samples: Number of samples to generate. Defaults to 20.
        num_features: Number of features per sample. Defaults to 10.
        num_classes: Number of target classes. Defaults to 3.
        seed: Random seed for reproducibility. Defaults to 1.
    """

    def __init__(
        self,
        num_samples: int = 20,
        num_features: int = 10,
        num_classes: int = 3,
        seed: int = 1,
    ) -> None:
        rng = np.random.default_rng(seed)

        self.X = rng.standard_normal((num_samples, num_features))
        self.y = rng.integers(0, num_classes, size=num_samples)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.X)

    def __getitem__(self, index: int) -> dict:
        """Retrieve a single sample and its label.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Dictionary with keys 'data' and 'label'.
        """
        return {
            "data": self.X[index],
            "label": self.y[index],
        }
