"""Are the model's probabilities honest?

A model is calibrated if, among all the times it says "90% sure", it is right
90% of the time. Accuracy doesn't measure this, but it matters whenever a
probability drives a decision, such as how much to risk on a prediction.
"""
import numpy as np
from src.losses import log_softmax


def reliability_bins(probs, y, n_bins=15):
    """Compare confidence (the largest probability) with accuracy, bin by bin.

    Bin i covers (i/n, (i+1)/n]; the first bin also includes 0, so a confidence
    of exactly 1.0 lands in the last bin. Empty bins get NaN.

    Returns {"edges", "confidence", "accuracy", "count"}.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_conf = np.where(count > 0, conf_sum / count, np.nan)
        bin_acc = np.where(count > 0, acc_sum / count, np.nan)
    return {"edges": edges, "confidence": bin_conf, "accuracy": bin_acc, "count": count}


def expected_calibration_error(probs, y, n_bins=15):
    """ECE = sum over bins of (bin count / total) * |bin accuracy - bin confidence|."""
    bins = reliability_bins(probs, y, n_bins)
    weights = bins["count"] / bins["count"].sum()
    gap = np.abs(bins["accuracy"] - bins["confidence"])
    return float(np.nansum(weights * gap))


def nll_at_temperature(logits, y, T):
    """Mean cross-entropy of softmax(logits / T). T > 1 softens the probabilities."""
    logp = log_softmax(logits / T)
    return float(-np.mean(logp[np.arange(len(y)), y]))


def fit_temperature(logits, y):
    """The T > 0 minimising nll_at_temperature, by a log-scale grid search
    followed by a finer search around the best value.

    Dividing every logit by T never changes which class is largest, so accuracy
    is untouched and only the confidence changes (Guo et al., 2017).
    """
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
    losses = [nll_at_temperature(logits, y, T) for T in grid]
    best = grid[int(np.argmin(losses))]
    fine = np.linspace(best * 0.95, best * 1.05, 101)
    losses = [nll_at_temperature(logits, y, T) for T in fine]
    return float(fine[int(np.argmin(losses))])
