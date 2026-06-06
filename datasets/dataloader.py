from .base_dataset import Dataset
import numpy as np

class DataLoader:
    """Batches data from a Dataset for training and evaluation.

    Handles shuffling, batch formation, and the drop_last option.

    Args:
        dataset: Dataset instance providing samples.
        batch_size: Number of samples per batch.
        drop_last: If True, drop the last incomplete batch. Defaults to False.
        shuffle: If True, shuffle sample order each epoch. Defaults to True.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    
    def __iter__(self):
        """Iterate over batches in the dataset.

        Yields:
            Dictionary with 'data' and 'label' keys, each containing batched arrays.
        """
        def collate_batch(batch: list) -> dict:
            return {
                key: np.array([sample[key] for sample in batch])
                for key in batch[0]
            }

        batch = []
        indices = (
            np.random.permutation(len(self.dataset))
            if self.shuffle
            else range(len(self.dataset))
        )

        for index in indices:
            batch.append(self.dataset[index])

            if len(batch) == self.batch_size:
                yield collate_batch(batch)
                batch = []
        
        if batch and not self.drop_last:
            yield collate_batch(batch)
