from datasets import Dataset
import numpy as np

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size, drop_last, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    
    def __iter__(self):
        def collate_batch(batch):
            return {
                key: np.array([sample[key] for sample in batch])
                for key in batch[0]
            }

        batch = []

        indices = np.random.permutation(len(self.dataset)) \
            if self.shuffle else range(len(self.dataset))

        for index in indices:
            batch.append(self.dataset[index])

            if len(batch) == self.batch_size:
                yield collate_batch(batch)
                batch = []
        
        if batch and not self.drop_last:
            yield collate_batch(batch)