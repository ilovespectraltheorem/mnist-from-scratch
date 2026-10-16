"""Robustness to shifted digits, data augmentation, and real drawings.

An MLP has no built-in notion that a 3 moved two pixels to the right is still
a 3: every pixel is an unrelated input. This module measures that, fixes it
with augmentation, and prepares hand-drawn digits the way MNIST's were (scaled
into a 20x20 box and centred by centre of mass).
"""
import numpy as np
from PIL import Image


def shift_images(X, dx, dy):
    """Move every image dx pixels right and dy pixels down (negative = left / up).

    Vacated pixels become 0 and pixels pushed past the edge are dropped (no
    wrap-around). Accepts (N, 784) or (N, 28, 28) and returns the same shape.
    """
    flat = X.ndim == 2
    imgs = X.reshape(-1, 28, 28)
    out = np.zeros_like(imgs)
    H, W = 28, 28
    if abs(dx) < W and abs(dy) < H:
        src_y = slice(max(0, -dy), H - max(0, dy))
        dst_y = slice(max(0, dy), H - max(0, -dy))
        src_x = slice(max(0, -dx), W - max(0, dx))
        dst_x = slice(max(0, dx), W - max(0, -dx))
        out[:, dst_y, dst_x] = imgs[:, src_y, src_x]
    return out.reshape(len(imgs), -1) if flat else out


def random_shift(X, max_shift, rng):
    """Augmentation: each image gets its own random shift, dx and dy each an
    integer in [-max_shift, max_shift]. Images are grouped by shift so there is
    one shift_images call per distinct (dx, dy)."""
    n = len(X)
    dxs = rng.integers(-max_shift, max_shift + 1, n)
    dys = rng.integers(-max_shift, max_shift + 1, n)
    out = np.empty_like(X)
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            idx = np.where((dxs == dx) & (dys == dy))[0]
            if len(idx):
                out[idx] = shift_images(X[idx], dx, dy)
    return out


def accuracy_vs_shift(model, X, y, shifts):
    """Accuracy at each shift k: plain accuracy for k = 0, otherwise the average
    over copies shifted k pixels right, left, down and up."""
    accs = []
    for k in shifts:
        if k == 0:
            accs.append(float(np.mean(model.predict(X) == y)))
            continue
        directions = [(k, 0), (-k, 0), (0, k), (0, -k)]
        accs.append(float(np.mean([np.mean(model.predict(shift_images(X, dx, dy)) == y)
                                   for dx, dy in directions])))
    return np.array(accs)


def preprocess_drawing(img, threshold=0.1):
    """Turn a drawing (white digit on black, any size) into an MNIST-style 784-vector.

    1. Scale to [0, 1] (accepts 0-255 or 0-1 input).
    2. Crop to the bounding box of pixels brighter than threshold.
    3. Resize so the longer side is 20 pixels, keeping the aspect ratio.
    4. Paste into the middle of a 28x28 canvas.
    5. Shift so the intensity-weighted centre of mass sits at (14, 14).
    """
    img = np.asarray(img, dtype=float)
    if img.max() > 1.0:
        img = img / 255.0
    mask = img > threshold
    if not mask.any():
        return np.zeros(784, dtype=np.float32)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    crop = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    pil = Image.fromarray((crop * 255).astype(np.uint8))
    small = np.asarray(pil.resize((new_w, new_h), Image.LANCZOS), dtype=float) / 255.0   # PIL: (width, height)
    canvas = np.zeros((28, 28))
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small
    ys, xs = np.indices(canvas.shape)
    total = canvas.sum()
    cy, cx = (ys * canvas).sum() / total, (xs * canvas).sum() / total
    canvas = shift_images(canvas[None], int(round(14 - cx)), int(round(14 - cy)))[0]
    return np.clip(canvas, 0, 1).reshape(784).astype(np.float32)
