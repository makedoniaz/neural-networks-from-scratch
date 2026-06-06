from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, parameters, learning_rate, weight_decay=0.0):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def step(self):
        for param in self.parameters:
            if param.grad is None:
                continue

            grad = param.grad

            if self.weight_decay != 0 and param.weight_decay:
                grad = grad + self.weight_decay * param.data

            self._update(param, grad)

    @abstractmethod
    def _update(self, param, grad):
        pass
