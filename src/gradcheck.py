"""Gradient checking: verifying backpropagation numerically.

Backprop computes gradients analytically. A gradient check measures them
directly instead: nudge one number up and down by eps and see how the loss
changes. If the two agree to about 1e-9, the backward passes are correct.
"""
import numpy as np


def numerical_gradient(f, x, eps=1e-5):
    """Central-difference estimate of d f / d x for every element of x.

    f takes no arguments and recomputes the loss from whatever is currently in
    x, so each element is perturbed in place and then restored. The central
    difference (f(x + eps) - f(x - eps)) / 2 eps has error O(eps^2).
    """
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def relative_error(a, b):
    """||a - b|| / (||a|| + ||b||), or 0.0 if both are zero.

    Below 1e-7 is right; 1e-7 to 1e-4 is suspicious (it can happen at ReLU's
    kink); above 1e-4 is a bug.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def gradient_check(model, loss_fn, X, y, eps=1e-5):
    """Compare backprop with numerical gradients for every parameter and for the input.

    Runs in evaluation mode (dropout masks would change the loss between calls)
    and in float64 (float32 has too few digits for finite differences).
    Returns {parameter key: relative error, ..., "input": relative error}.
    """
    model.eval()
    X = X.astype(float).copy()
    loss_fn.forward(model.forward(X), y)
    grad_X = model.backward(loss_fn.backward())
    analytic = {key: g.copy() for key, _, g in model.parameters()}

    def f():
        return loss_fn.forward(model.forward(X), y)

    errors = {}
    for key, param, _ in model.parameters():
        errors[key] = relative_error(analytic[key], numerical_gradient(f, param, eps))
    errors["input"] = relative_error(grad_X, numerical_gradient(f, X, eps))
    return errors
