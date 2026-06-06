from .base_loss import Loss
import numpy as np

class L1(Loss):
    """L1 (Mean Absolute Error) loss.

    Computes element-wise absolute differences: loss = |y_out - y_truth|.
    """

    def forward(self, y_out: np.ndarray, y_truth: np.ndarray) -> float | np.ndarray:
        difference = y_out - y_truth
        self.cache = difference
        loss = np.abs(difference)
        return self.reduce(loss)
    
    def backward(self) -> np.ndarray:
        grad = np.sign(self.cache)
        return self.scale_grad(grad)
    

class MSE(Loss):
    """Mean Squared Error loss.

    Computes element-wise squared differences: loss = (y_out - y_truth)^2.
    """

    def forward(self, y_out: np.ndarray, y_truth: np.ndarray) -> float | np.ndarray:
        difference = y_out - y_truth
        self.cache = difference
        loss = difference ** 2
        return self.reduce(loss)
    
    def backward(self) -> np.ndarray:
        grad = 2 * self.cache
        return self.scale_grad(grad)
    
class BCE(Loss):
    """Binary Cross Entropy loss.

    For binary classification: loss = -y_truth * log(y_out) - (1 - y_truth) * log(1 - y_out).
    """

    def forward(self, y_out: np.ndarray, y_truth: np.ndarray) -> float | np.ndarray:
        eps = 1e-12
        y_out = np.clip(y_out, eps, 1 - eps)
        self.cache = (y_truth, y_out)
        loss = -y_truth * np.log(y_out) - (1 - y_truth) * np.log(1 - y_out)
        return self.reduce(loss)
    
    def backward(self) -> np.ndarray:
        y_truth, y_out = self.cache
        grad = - (y_truth / y_out) + (1 - y_truth) / (1 - y_out)
        return self.scale_grad(grad)
    
class SoftmaxCrossEntropy(Loss):
    """Softmax Cross Entropy loss for multi-class classification.

    Combines softmax activation with cross-entropy:
    loss = -log(softmax(logits)[correct_class]).
    """

    def forward(self, logits: np.ndarray, y_truth: np.ndarray) -> float | np.ndarray:
        N, _ = logits.shape
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(N), y_truth])
        self.cache = (probs, y_truth)
        return self.reduce(loss)

    def backward(self) -> np.ndarray:
        probs, y_truth = self.cache
        N, _ = probs.shape
        grad = probs.copy()
        grad[np.arange(N), y_truth] -= 1
        if self.reduction == 'mean':
            grad /= N
        return grad