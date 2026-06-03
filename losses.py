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

class L1(Loss):
    def forward(self, y_out, y_truth):
        difference = y_out - y_truth
        self.cache = difference

        loss = np.abs(difference)
        return self.reduce(loss)
    
    def backward(self):
        grad = np.sign(self.cache)
        return self.scale_grad(grad)
    

class MSE(Loss):
    def forward(self, y_out, y_truth):
        difference = y_out - y_truth
        self.cache = difference

        loss = difference ** 2
        return self.reduce(loss)
    
    def backward(self):
        grad = 2 * self.cache
        return self.scale_grad(grad)
    
class BCE(Loss):
    def forward(self, y_out, y_truth):
        eps = 1e-12
        y_out = np.clip(y_out, eps, 1 - eps)

        self.cache = (y_truth, y_out)

        loss = -y_truth * np.log(y_out) - (1 - y_truth) * np.log(1 - y_out)
        
        return self.reduce(loss)
    
    def backward(self):
        y_truth, y_out = self.cache
        grad = - (y_truth / y_out) + (1 - y_truth) / (1 - y_out)

        return self.scale_grad(grad)
    
class SoftmaxCrossEntropy(Loss):
    def forward(self, logits, y_truth):
        N, _ = logits.shape

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True) # probs_ij, i - i-th object, j - j-th class

        loss = -np.log(probs[np.arange(N), y_truth]) # for each object i take the prob of its correct class y_i = j

        self.cache = (probs, y_truth)

        return self.reduce(loss)

    def backward(self):
        probs, y_truth = self.cache
        N, _ = probs.shape

        grad = probs.copy()
        grad[np.arange(N), y_truth] -= 1

        if self.reduction == 'mean':
            grad /= N

        return grad