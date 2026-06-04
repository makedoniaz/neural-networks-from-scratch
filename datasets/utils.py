import numpy as np

def random_split(dataset, splits, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))

    total_ratio = sum(splits.values())

    if not np.isclose(total_ratio, 1.0):
        raise ValueError("Split percentages must sum to 1.0")

    total_size = len(dataset)

    lengths = {
        name: int(ratio * total_size)
        for name, ratio in splits.items()
    }

    remainder = total_size - sum(lengths.values())
    last_name = list(splits.keys())[-1]
    lengths[last_name] += remainder

    subsets = {}
    start = 0

    for name, length in lengths.items():
        end = start + length
        subsets[name] = dataset.subset(indices[start:end])
        start = end

    return subsets