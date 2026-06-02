from abc import ABC, abstractmethod
import numpy as np

class Dataset(ABC):
    def __init__(self, root):
        self.root = root
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, index):
        pass


class DummyDataset(Dataset):
    def __init__(self, root, num_samples = 10):
        super().__init__(root)
        self.data = np.linspace(1, 10, num_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"data": self.data[index].item()}
