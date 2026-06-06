from .array_dataset import ArrayDataset
from .utils import Transform
import idx2numpy

class IDXDataset(ArrayDataset):
    """Dataset for loading IDX format files (commonly used in machine learning datasets).

    IDX is a simple format for vectors and multidimensional matrices of various numerical types.

    Args:
        data_path: Path to the IDX-format data file.
        labels_path: Path to the IDX-format labels file.
        transform: Optional transformation to apply to data samples. Defaults to None.
        target_transform: Optional transformation to apply to labels. Defaults to None.
    """

    def __init__(
        self,
        data_path: str,
        labels_path: str,
        transform: Transform | None = None,
        target_transform: Transform | None = None,
    ) -> None:
        data = idx2numpy.convert_from_file(data_path)
        labels = idx2numpy.convert_from_file(labels_path)

        super().__init__(
            data=data,
            labels=labels,
            transform=transform,
            target_transform=target_transform,
        )