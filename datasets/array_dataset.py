from .base_dataset import Dataset
from .utils import Transform

class ArrayDataset(Dataset):
    def __init__(self, data, labels, transform: Transform = None, target_transform: Transform = None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

        if len(data) != len(labels):
            raise ValueError(
                "Data and labels must have the same length."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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