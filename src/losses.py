"""Softmax and cross-entropy."""
import numpy as np


def softmax(z):
    """Row-wise softmax of logits z (N, K); each row sums to 1.

    Subtracting each row's maximum leaves the result unchanged (the constant
    cancels top and bottom) and stops exp() overflowing for large logits.
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=1, keepdims=True)


def log_softmax(z):
    """log(softmax(z)) without ever taking log(0), via the log-sum-exp trick:

        log softmax(z)_k = (z_k - m) - log( sum_j exp(z_j - m) ),   m = row max
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


class SoftmaxCrossEntropy:
    """Softmax followed by cross-entropy, combined because together their
    gradient with respect to the logits is simply (p - one_hot(y)) / N."""

    def forward(self, logits, y):
        """Mean over the batch of -log p(correct class)."""
        self.probs = softmax(logits)
        self.y = y
        logp = log_softmax(logits)
        return float(-np.mean(logp[np.arange(len(y)), y]))

    def backward(self):
        """Gradient of the mean loss with respect to the logits."""
        n = len(self.y)
        grad = self.probs.copy()
        grad[np.arange(n), self.y] -= 1.0
        return grad / n
