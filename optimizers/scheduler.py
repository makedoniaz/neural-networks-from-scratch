from abc import ABC, abstractmethod
from optimizers import Optimizer

class Scheduler(ABC):
    """Abstract base class for learning rate schedulers.

    Adjusts the optimizer's learning rate based on training progress.

    Args:
        optimizer: Optimizer instance to schedule.
    """

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
    
    @abstractmethod
    def step(self) -> None:
        """Update the learning rate."""
        pass

class StepLR(Scheduler):
    """Step-based learning rate scheduler.

    Multiplies learning rate by gamma every step_size epochs.

    Args:
        optimizer: Optimizer instance to schedule.
        step_size: Number of epochs before applying decay.
        gamma: Multiplicative decay factor. Defaults to 0.1.
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self) -> None:
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.learning_rate *= self.gamma