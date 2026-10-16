"""Loading MNIST and feeding it to the network.

THE DATA AT EVERY STAGE
-----------------------
1. On disk (IDX format, big-endian):
     image file:  [magic 2051][n][rows][cols] then n*rows*cols bytes, one per pixel (0-255)
     label file:  [magic 2049][n]             then n bytes, one per label (0-9)

2. read_idx_images -> uint8 array (n, 28, 28)    a stack of n little pictures
   read_idx_labels -> uint8 array (n,)           one digit per picture, same order

3. load_mnist -> X float32 (n, 784) in [0, 1]    each picture flattened into one row
                 y int64   (n,)                  labels as integers 0-9

4. get_data -> a dict of six arrays:
     X_train (55000, 784)  y_train (55000,)     used to learn the weights
     X_val    (5000, 784)  y_val    (5000,)     used to choose settings / early stopping
     X_test  (10000, 784)  y_test  (10000,)     used once, at the end

5. iterate_minibatches -> (X_batch (b, 784), y_batch (b,)) pairs, b = batch_size
   (the last batch of an epoch may be smaller)

Row i of X always belongs with entry i of y. Every function below keeps that true.
"""
import struct
from pathlib import Path
import numpy as np

IMAGE_MAGIC = 2051      # first 4 bytes of every MNIST image file
LABEL_MAGIC = 2049      # first 4 bytes of every MNIST label file
FILES = {
    "train_images": "train-images.idx3-ubyte",
    "train_labels": "train-labels.idx1-ubyte",
    "test_images": "t10k-images.idx3-ubyte",
    "test_labels": "t10k-labels.idx1-ubyte",
}


def read_idx_images(path):
    """Read an MNIST image file into a uint8 array of shape (n, rows, cols)."""
    with open(path, "rb") as f:                      # "rb" = read raw bytes, not text
        # The header is 16 bytes: four unsigned 32-bit integers.
        # ">" = big-endian (most significant byte first), "I" = unsigned 4-byte int.
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != IMAGE_MAGIC:
            raise ValueError(f"{path} is not an MNIST image file "
                             f"(magic number {magic}, expected {IMAGE_MAGIC})")
        # Everything after the header is pixels. frombuffer views the bytes as a
        # 1-D array of n*rows*cols unsigned 8-bit integers (0 = black, 255 = white).
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    if pixels.size != n * rows * cols:
        raise ValueError(f"{path}: header promises {n * rows * cols} pixels, "
                         f"file contains {pixels.size}")
    # Pixels are stored image after image, row by row, so this reshape cuts the
    # long 1-D run back into n pictures of rows x cols.
    return pixels.reshape(n, rows, cols)


def read_idx_labels(path):
    """Read an MNIST label file into a uint8 array of shape (n,)."""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))   # header is only 8 bytes here
        if magic != LABEL_MAGIC:
            raise ValueError(f"{path} is not an MNIST label file "
                             f"(magic number {magic}, expected {LABEL_MAGIC})")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.size != n:
        raise ValueError(f"{path}: header promises {n} labels, file contains {labels.size}")
    return labels


def load_mnist(data_dir):
    """Returns X_train (60000, 784) float32 in [0, 1], y_train int64,
    X_test (10000, 784), y_test."""
    data_dir = Path(data_dir)
    missing = [f for f in FILES.values() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing MNIST files in {data_dir}: {missing}. "
            "Put the four .idx-ubyte files in the data/ folder.")
    Xtr = read_idx_images(data_dir / FILES["train_images"])
    ytr = read_idx_labels(data_dir / FILES["train_labels"])
    Xte = read_idx_images(data_dir / FILES["test_images"])
    yte = read_idx_labels(data_dir / FILES["test_labels"])
    # (n, 28, 28) -> (n, 784): each picture becomes one row of 784 numbers, because
    # a dense layer takes one vector per example. -1 means "work this size out".
    # Then scale 0-255 to 0-1: small inputs keep the first layer's outputs in a
    # sensible range, so the initialisation schemes in layers.py work as intended.
    Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0
    Xte = Xte.reshape(len(Xte), -1).astype(np.float32) / 255.0
    # Labels become int64: uint8 arithmetic wraps around (e.g. 0 - 1 = 255), which
    # is a nasty silent bug if a label is ever used in a calculation.
    return Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64)


def train_val_split(X, y, val_size, rng):
    """Shuffle, then split off val_size examples as a validation set.
    Returns X_train, y_train, X_val, y_val."""
    # One random ordering of the row numbers 0..n-1, e.g. [41207, 3, 58122, ...].
    perm = rng.permutation(len(X))
    val_idx = perm[:val_size]           # the first val_size shuffled row numbers
    train_idx = perm[val_size:]         # all the rest: no overlap, nothing lost
    # Indexing X and y with the SAME index array keeps every image with its label.
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def iterate_minibatches(X, y, batch_size, rng=None):
    """Yield (X_batch, y_batch) pairs that cover every example exactly once.

    One call = one epoch. With an rng, the order is a fresh random permutation
    each call; without one, the original order is kept (for evaluation/tests).
    """
    n = len(X)
    order = rng.permutation(n) if rng is not None else np.arange(n)
    # Walk through the (shuffled) row numbers in steps of batch_size:
    # 0..99, 100..199, ... The last slice is just whatever is left over.
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]            # "yield" hands out one batch, then pauses here


def one_hot(y, num_classes=10):
    """Integer labels (n,) -> (n, num_classes) array, a 1 in each label's column.
    e.g. [2, 0] -> [[0,0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]]"""
    out = np.zeros((len(y), num_classes))
    # Pairs up row 0 with column y[0], row 1 with column y[1], ... and sets each to 1.
    out[np.arange(len(y)), y] = 1.0
    return out


def get_data(data_dir, val_size, split_seed):
    """Loads MNIST and splits off the validation set with a fixed seed, so every
    experiment uses exactly the same train / validation / test images."""
    X, y, X_test, y_test = load_mnist(data_dir)
    rng = np.random.default_rng(split_seed)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_size, rng)
    return {"X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test}
