from .array_dataset import ArrayDataset
from .transforms import Transform
import idx2numpy

class IDXDataset(ArrayDataset):
    def __init__(self, data_path, labels_path, transform: Transform = None, target_transform: Transform = None):
        data = idx2numpy.convert_from_file(data_path)
        labels = idx2numpy.convert_from_file(labels_path)

        super().__init__(
            data=data,
            labels=labels,
            transform=transform,
            target_transform=target_transform,
        )