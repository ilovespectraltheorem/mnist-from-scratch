"""Optimisers: turning gradients into parameter updates."""
import numpy as np


def is_weight(key):
    """True for weight matrices ("0.W"), False for biases ("0.b")."""
    return key.endswith(".W")


class Optimizer:
    def __init__(self, lr, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.state = {}         # per-parameter memory, keyed by the parameter's key
        self.t = 0              # number of steps taken so far

    def step(self, model):
        """One update of every parameter, after model.backward()."""
        self.t += 1
        for key, param, grad in model.parameters():
            if grad is None:
                continue
            grad = self.add_weight_decay(key, param, grad)
            self.update(key, param, grad)

    def add_weight_decay(self, key, param, grad):
        """L2 regularisation: adding (weight_decay / 2) * ||W||^2 to the loss adds
        weight_decay * W to the gradient. Applied to weights only, not biases."""
        if self.weight_decay and is_weight(key):
            return grad + self.weight_decay * param
        return grad

    def update(self, key, param, grad):
        raise NotImplementedError


class SGD(Optimizer):
    """Plain SGD, w <- w - lr * g, or with momentum:

        v <- momentum * v + g,    w <- w - lr * v

    Momentum averages past gradients, so the steady-state step is lr / (1 - momentum).
    """

    def __init__(self, lr, momentum=0.0, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.momentum = momentum

    def update(self, key, param, grad):
        # Updates are in place (-=): param is the array inside the layer.
        if self.momentum:
            v = self.state.get(key)
            if v is None:
                v = np.zeros_like(param)
            v = self.momentum * v + grad
            self.state[key] = v
            param -= self.lr * v
        else:
            param -= self.lr * grad


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015). With t the step count:

        m <- beta1 * m + (1 - beta1) * g            running mean of gradients
        v <- beta2 * v + (1 - beta2) * g^2          running mean of squared gradients
        m_hat = m / (1 - beta1^t),   v_hat = v / (1 - beta2^t)    bias correction
        w <- w - lr * m_hat / (sqrt(v_hat) + eps)

    Momentum plus a separate step size for every parameter. The bias correction
    stops the first steps being tiny, because m and v start at zero.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def update(self, key, param, grad):
        m, v = self.state.get(key, (np.zeros_like(param), np.zeros_like(param)))
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad ** 2
        self.state[key] = (m, v)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
