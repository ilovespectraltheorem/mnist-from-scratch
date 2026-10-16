"""Convolution and max-pooling layers, written with NumPy.

Images have shape (N, C, H, W): batch, channels, height, width. Stride is 1;
padding adds `pad` rows/columns of zeros on every side.

im2col copies every k x k patch a filter looks at into one row of a matrix, so
the whole convolution becomes a single matrix multiply, and its backward pass
is Linear.backward on that matrix followed by col2im.
"""
import numpy as np
from src.layers import Layer, init_std


def conv_output_size(size, kernel, pad):
    """Output height (or width) of a stride-1 convolution."""
    return size + 2 * pad - kernel + 1


def im2col(x, kh, kw, pad):
    """(N, C, H, W) -> (N * OH * OW, C * kh * kw): one row per output position,
    holding the patch the filter sees there. Loops only over the kh * kw kernel
    offsets, copying one shifted slice of the padded input for all positions at once."""
    N, C, H, W = x.shape
    OH, OW = conv_output_size(H, kh, pad), conv_output_size(W, kw, pad)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((N, C, kh, kw, OH, OW), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = xp[:, :, i:i + OH, j:j + OW]
    # row order (n, oh, ow); column order (c, i, j), matching W.reshape(F, -1)
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, C * kh * kw)


def col2im(cols, x_shape, kh, kw, pad):
    """Adjoint of im2col: scatter each row back to its patch, adding where patches
    overlap (a pixel feeding several outputs receives the sum of their
    gradients), then remove the padding."""
    N, C, H, W = x_shape
    OH, OW = conv_output_size(H, kh, pad), conv_output_size(W, kw, pad)
    cols = cols.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i:i + OH, j:j + OW] += cols[:, :, i, j, :, :]
    return xp[:, :, pad:pad + H, pad:pad + W]


class Conv2D(Layer):
    """2-D convolution (cross-correlation, as in every deep-learning library).
    W has shape (out_channels, in_channels, k, k)."""

    def __init__(self, in_channels, out_channels, kernel_size, pad, rng, init="he"):
        super().__init__()
        k = kernel_size
        std = init_std(in_channels * k * k, out_channels * k * k, init)
        self.params["W"] = rng.standard_normal((out_channels, in_channels, k, k)) * std
        self.params["b"] = np.zeros(out_channels)
        self.k, self.pad = k, pad

    def forward(self, x):
        N, C, H, W = x.shape
        F = self.params["W"].shape[0]
        OH, OW = conv_output_size(H, self.k, self.pad), conv_output_size(W, self.k, self.pad)
        self.x_shape = x.shape
        self.cols = im2col(x, self.k, self.k, self.pad)
        out = self.cols @ self.params["W"].reshape(F, -1).T + self.params["b"]
        return out.reshape(N, OH, OW, F).transpose(0, 3, 1, 2)

    def backward(self, grad_out):
        F = self.params["W"].shape[0]
        g = grad_out.transpose(0, 2, 3, 1).reshape(-1, F)       # same layout as forward's matrix
        self.grads["W"] = (g.T @ self.cols).reshape(self.params["W"].shape)
        self.grads["b"] = g.sum(axis=0)
        dcols = g @ self.params["W"].reshape(F, -1)
        return col2im(dcols, self.x_shape, self.k, self.k, self.pad)


class MaxPool2D(Layer):
    """Non-overlapping size x size max-pooling. The gradient of each window goes
    entirely to the position that held the maximum (the first one, on ties)."""

    def __init__(self, size=2):
        super().__init__()
        self.size = size

    def _windows(self, x):
        N, C, H, W = x.shape
        s = self.size
        OH, OW = H // s, W // s
        w = x[:, :, :OH * s, :OW * s].reshape(N, C, OH, s, OW, s)
        return w.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, OH, OW, s * s)

    def forward(self, x):
        windows = self._windows(x)
        arg = np.argmax(windows, axis=-1)
        self.mask = np.zeros_like(windows)
        np.put_along_axis(self.mask, arg[..., None], 1.0, axis=-1)
        self.x_shape = x.shape
        return np.take_along_axis(windows, arg[..., None], axis=-1)[..., 0]

    def backward(self, grad_out):
        N, C, H, W = self.x_shape
        s = self.size
        OH, OW = H // s, W // s
        g = self.mask * grad_out[..., None]
        g = g.reshape(N, C, OH, OW, s, s).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, OH * s, OW * s)
        dx = np.zeros(self.x_shape, dtype=g.dtype)
        dx[:, :, :OH * s, :OW * s] = g
        return dx
