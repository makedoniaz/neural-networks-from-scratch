import numpy as np

class Dataloader:
    def __init__(self, dataset, batch_size, drop_last, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    
    def __iter__(self):
        batch = []

        indices = np.random.permutation(len(self.dataset)) \
            if self.shuffle else range(len(self.dataset))

        for index in indices:
            batch.append(self.dataset[index])

            if len(batch) == self.batch_size:
                yield np.array(batch)
                batch = []
        
        if batch and not self.drop_last:
            yield np.array(batch)