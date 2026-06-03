from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.cache = None

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    def parameters(self):
        return []

    def __repr__(self):
        return f"{self.__class__.__name__}()"