from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    def __init__(self, reduction='mean'):
        self.cache = None
        self.reduction = reduction
    
    @abstractmethod
    def forward(self, y_out, y_truth):
        pass

    @abstractmethod
    def backward(self):
        pass

    def reduce(self, loss):
        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        elif self.reduction == 'none':
            return loss
        
    def scale_grad(self, grad):
        if self.reduction == 'mean':
            grad /= grad.size

        return grad
