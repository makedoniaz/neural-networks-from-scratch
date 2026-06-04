from abc import ABC, abstractmethod
from .subset import Subset
import numpy as np

class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def subset(self, indices):
        return Subset(self, indices)

class DummyDataset(Dataset):
    def __init__(
        self,
        num_samples=20,
        num_features=10,
        num_classes=3,
        seed=1,
    ):
        rng = np.random.default_rng(seed)

        self.X = rng.standard_normal((num_samples, num_features))
        self.y = rng.integers(0, num_classes, size=num_samples)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {
            "data": self.X[index],
            "label": self.y[index],
        }
