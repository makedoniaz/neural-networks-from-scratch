from abc import ABC, abstractmethod
import numpy as np

class Transform(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

class ToFloat(Transform):
    def __call__(self, x):
        return x.astype(np.float32)

class Normalize(Transform):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

class Scale(Transform):
    def __init__(self, scale=255.0):
        self.scale = scale

    def __call__(self, x):
        return x / self.scale

class Flatten(Transform):
    def __call__(self, x):
        return x.reshape(-1)