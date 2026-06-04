from .base_dataset import Dataset
import numpy as np

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]