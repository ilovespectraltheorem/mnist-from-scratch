"""Layers: each one knows how to go forward and how to send gradients back.

Every layer is a small object with two methods:

    forward(x)          -> output     (and remembers whatever backward will need)
    backward(grad_out)  -> grad_in    (and fills self.grads for its parameters)

grad_out is dLoss/d(this layer's output) and grad_in is dLoss/d(this layer's
input). Passing grad_in to the previous layer, all the way back to the start,
is the chain rule: that is all backpropagation is.

Dense layers take x of shape (batch, features) and work for any batch size.
"""
import numpy as np


class Layer:
    """Base class. Parameters live in self.params and their gradients in
    self.grads (same keys, same shapes). self.training is set by the network."""

    def __init__(self):
        self.params = {}
        self.grads = {}
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError


def init_std(fan_in, fan_out, scheme):
    """Standard deviation for random initial weights.

        "lecun":  variance 1 / fan_in
        "he":     variance 2 / fan_in               (designed for ReLU)
        "xavier": variance 2 / (fan_in + fan_out)   (designed for tanh/sigmoid)

    Each scheme keeps the variance of activations roughly constant from layer
    to layer. He doubles it because ReLU zeroes about half of its inputs.
    """
    if scheme == "lecun":
        return np.sqrt(1.0 / fan_in)
    if scheme == "he":
        return np.sqrt(2.0 / fan_in)
    if scheme == "xavier":
        return np.sqrt(2.0 / (fan_in + fan_out))
    raise ValueError(f"unknown init scheme {scheme!r}")


class Linear(Layer):
    """Fully connected layer: out = x W + b, with W of shape (in, out)."""

    def __init__(self, in_features, out_features, rng, init="he"):
        super().__init__()
        std = init_std(in_features, out_features, init)
        self.params["W"] = rng.standard_normal((in_features, out_features)) * std
        self.params["b"] = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out):
        # grad_out already includes the loss's 1/N, so no division by the batch size here.
        self.grads["W"] = self.x.T @ grad_out          # (in, batch) @ (batch, out) -> (in, out)
        self.grads["b"] = grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T           # (batch, out) @ (out, in) -> (batch, in)


class ReLU(Layer):
    """max(0, x). The gradient is 1 where the input was positive and 0 elsewhere,
    so, unlike sigmoid, it doesn't shrink as it passes through many layers."""

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


class Sigmoid(Layer):
    """1 / (1 + e^-x), computed as (1 + tanh(x/2)) / 2, which never overflows.
    Its derivative s(1 - s) is at most 0.25, the source of vanishing gradients."""

    def forward(self, x):
        self.out = 0.5 * (1.0 + np.tanh(0.5 * x))
        return self.out

    def backward(self, grad_out):
        return grad_out * self.out * (1.0 - self.out)


class Dropout(Layer):
    """Inverted dropout: in training, zero each unit with probability p and scale
    the survivors by 1 / (1 - p), so the expected activation is unchanged and
    nothing needs to change at test time. In evaluation mode it does nothing."""

    def __init__(self, p, rng):
        super().__init__()
        self.p = p          # probability of DROPPING a unit
        self.rng = rng

    def forward(self, x):
        if not self.training or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad_out):
        if self.mask is None:
            return grad_out
        return grad_out * self.mask


class Flatten(Layer):
    """(N, C, H, W) -> (N, C*H*W). Used after the conv layers in the CNN."""

    def forward(self, x):
        self.in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_out):
        return grad_out.reshape(self.in_shape)


class Reshape(Layer):
    """(N, 784) -> (N, 1, 28, 28), so a CNN takes the same flat input as the MLP."""

    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)      # target shape without the batch axis

    def forward(self, x):
        self.in_shape = x.shape
        return x.reshape((x.shape[0],) + self.shape)

    def backward(self, grad_out):
        return grad_out.reshape(self.in_shape)
