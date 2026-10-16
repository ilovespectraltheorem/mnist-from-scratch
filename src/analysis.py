"""What does the model get wrong, and what has it learned?"""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes=10):
    """cm[i, j] = number of images whose true digit is i and predicted digit is j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    # np.add.at counts repeated (i, j) pairs; cm[y_true, y_pred] += 1 would count each pair once.
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision_recall(cm):
    """Per-class precision (diagonal / column total) and recall (diagonal / row
    total), with 0 where a total is 0."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    return precision, recall


def top_confusions(cm, k=5):
    """The k most common mistakes as (true, predicted, count), largest first,
    ties broken by (true, predicted)."""
    pairs = [(int(cm[i, j]), i, j) for i in range(cm.shape[0])
             for j in range(cm.shape[1]) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(i, j, c) for c, i, j in pairs[:k]]


def pca(X, k):
    """Principal component analysis via the SVD of the centred data.

    The rows of V^T are the principal directions, largest variance first; the
    variance along direction i is S_i^2 / (n - 1). This avoids forming the
    D x D covariance matrix.

    Returns (projected (n, k), components (k, D), explained_ratio (k,)).
    """
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]
    var = S ** 2 / (len(X) - 1)
    ratio = var[:k] / var.sum()
    return Xc @ components.T, components, ratio


def hidden_activations(model, X, n_layers, batch_size=1000):
    """Output of the first n_layers layers (evaluation mode, in batches).
    For the MLP, n_layers = 2 gives the hidden layer after its activation."""
    was_training = [layer.training for layer in model.layers]
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        h = X[i:i + batch_size]
        for layer in model.layers[:n_layers]:
            h = layer.forward(h)
        outs.append(h)
    for layer, flag in zip(model.layers, was_training):
        layer.training = flag
    return np.concatenate(outs, axis=0)
