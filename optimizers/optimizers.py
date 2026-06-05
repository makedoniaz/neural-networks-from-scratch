from .base_optimizer import Optimizer
import numpy as np

class SGD(Optimizer):
    def _update(self, param, grad):
        param.data -= self.learning_rate * grad

class SGDMomentum(Optimizer):
    def __init__(self, parameters, learning_rate=1e-3, momentum=0.9, weight_decay=0.0):
        super().__init__(parameters, learning_rate, weight_decay)
        self.momentum = momentum
        self.velocities = {}

    def _update(self, param, grad):
        param_id = id(param)

        if param_id not in self.velocities:
            self.velocities[param_id] = np.zeros_like(param.data)

        v = self.velocities[param_id]
        v = self.momentum * v - self.learning_rate * grad

        param.data += v

        self.velocities[param_id] = v

class Adam(Optimizer):
    def __init__(
        self,
        parameters,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(parameters, learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        super().step()

    def _update(self, param, grad):
        param_id = id(param)

        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param.data)
            self.v[param_id] = np.zeros_like(param.data)

        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

        param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)