from abc import ABC, abstractmethod
from network import Network

class Optimizer(ABC):
    def __init__(self, model: Network, learning_rate):
        self.model = model
        self.learning_rate = learning_rate
    
    def step(self):
        for param, grad in self.model.parameters():
            if grad is not None:
                self._update(param, grad)

    @abstractmethod
    def _update(self, param, grad):
        pass
