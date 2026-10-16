"""The training loop, with validation tracking and early stopping."""
import numpy as np
from src.data import iterate_minibatches


def evaluate(model, loss_fn, X, y, batch_size=1000):
    """Mean loss and accuracy of the model on X, y (dropout off)."""
    logits = model.logits(X, batch_size)
    loss = loss_fn.forward(logits, y)
    acc = float(np.mean(np.argmax(logits, axis=1) == y))
    return loss, acc


def train(model, loss_fn, optimizer, X_train, y_train, X_val, y_val,
          epochs, batch_size, rng, patience=None, augment=None, verbose=True):
    """Train for up to `epochs` epochs and return a history dict:

        {"train_loss": [...], "val_loss": [...], "val_acc": [...], "best_epoch": None or int}

    Each epoch visits every training example once in a fresh random order. If
    `augment` is given, each training batch is replaced by augment(batch, rng).

    Early stopping (when patience is not None): an epoch improves if its
    validation loss is strictly the lowest so far; after `patience` epochs in a
    row without improvement, training stops and the best weights are restored.
    The test set is never used here.
    """
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "best_epoch": None}
    best_loss, best_state, since_best = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, rng):
            if augment is not None:
                xb = augment(xb, rng)
            logits = model.forward(xb)
            batch_losses.append(loss_fn.forward(logits, yb))
            model.backward(loss_fn.backward())
            optimizer.step(model)
        val_loss, val_acc = evaluate(model, loss_fn, X_val, y_val)
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(f"  epoch {epoch + 1:3d}  train loss {history['train_loss'][-1]:.4f}"
                  f"  val loss {val_loss:.4f}  val acc {val_acc:.4f}")
        if patience is not None:
            if val_loss < best_loss:
                best_loss, best_state, since_best = val_loss, model.state_dict(), 0
                history["best_epoch"] = epoch
            else:
                since_best += 1
                if since_best >= patience:
                    break
    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)
    return history
