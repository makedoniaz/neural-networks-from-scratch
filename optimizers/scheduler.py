from abc import ABC, abstractmethod
from optimizers import Optimizer

class Scheduler(ABC):
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    @abstractmethod
    def step(self):
        pass

class StepLR(Scheduler):
    def __init__(self, optimizer: Optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)

        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        self.epoch += 1

        if self.epoch % self.step_size == 0:
            self.optimizer.learning_rate *= self.gamma