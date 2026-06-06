from .base_optimizer import Optimizer
import numpy as np

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Basic gradient descent with constant learning rate:
    param = param - learning_rate * grad.
    """

    def _update(self, param, grad) -> None:
        param.data -= self.learning_rate * grad

class SGDMomentum(Optimizer):
    """SGD with Momentum optimizer.

    Accumulates gradient momentum to accelerate convergence:
    v = momentum * v - learning_rate * grad
    param = param + v.

    Args:
        parameters: List of Parameter objects to optimize.
        learning_rate: Learning rate. Defaults to 1e-3.
        momentum: Momentum coefficient (typically 0.9). Defaults to 0.9.
        weight_decay: L2 regularization coefficient. Defaults to 0.0.
    """

    def __init__(
        self,
        parameters: list,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, learning_rate, weight_decay)
        self.momentum = momentum
        self.velocities = {}

    def _update(self, param, grad) -> None:
        param_id = id(param)

        if param_id not in self.velocities:
            self.velocities[param_id] = np.zeros_like(param.data)

        v = self.velocities[param_id]
        v = self.momentum * v - self.learning_rate * grad

        param.data += v

        self.velocities[param_id] = v

class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer.

    Combines momentum and adaptive learning rates for efficient optimization.
    Maintains estimates of first and second moments of gradients.

    Args:
        parameters: List of Parameter objects to optimize.
        learning_rate: Learning rate. Defaults to 1e-3.
        beta1: Exponential decay rate for first moment estimates. Defaults to 0.9.
        beta2: Exponential decay rate for second moment estimates. Defaults to 0.999.
        eps: Small constant for numerical stability. Defaults to 1e-8.
        weight_decay: L2 regularization coefficient. Defaults to 0.0.
    """

    def __init__(
        self,
        parameters: list,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self) -> None:
        self.t += 1
        super().step()

    def _update(self, param, grad) -> None:
        param_id = id(param)

        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param.data)
            self.v[param_id] = np.zeros_like(param.data)

        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

        param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)