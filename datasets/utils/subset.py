from datasets.base_dataset import Dataset
from collections.abc import Sequence
import numpy as np

class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, float]:
        return self.dataset[self.indices[index]]