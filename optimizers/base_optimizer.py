from abc import ABC, abstractmethod
from layers import Layer
from losses import Loss
from network import Network

class Optimizer(ABC):
    def __init__(self, model: Network, loss_func: Loss, learning_rate):
        self.model = model
        self.loss_func = loss_func
        self.learning_rate = learning_rate
    
    def backward(self):
        dout = self.loss_func.backward()
        self.model.backward(dout)
    
    def step(self):
        for layer in self.model.layers:
            for param, grad in layer.parameters():
                if grad is not None:
                    self._update(param, grad)

    @abstractmethod
    def _update(self, param, grad):
        pass
