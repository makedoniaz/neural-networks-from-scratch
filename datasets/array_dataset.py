from .base_dataset import Dataset
from .utils import Transform
import numpy as np

class ArrayDataset(Dataset):
    """Dataset wrapping arrays of data and labels with optional transformations.

    Args:
        data: Array-like containing input samples.
        labels: Array-like containing target labels.
        transform: Optional transformation to apply to data samples. Defaults to None.
        target_transform: Optional transformation to apply to labels. Defaults to None.

    Raises:
        ValueError: If data and labels have different lengths.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        transform: Transform | None = None,
        target_transform: Transform | None = None,
    ) -> None:
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

        if len(data) != len(labels):
            raise ValueError(
                "Data and labels must have the same length."
            )

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """Retrieve a single sample and its label.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Dictionary with keys 'data' (transformed if applicable) and 'label'.
        """
        x = self.data[index]
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)
        
        if self.target_transform:
            y = self.target_transform(y)

        return {
            "data": x,
            "label": y,
        }