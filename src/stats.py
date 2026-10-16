"""Is an improvement real, or noise?

Accuracy on 10,000 test images is an estimate with sampling error. These
functions put error bars on it and test whether one model is really better
than another on the same images.

Each `correct` argument is a boolean array with one entry per test image
(True = the model got that image right).
"""
import math
from statistics import NormalDist
import numpy as np


def mean_std(values):
    """Mean and sample standard deviation (n - 1 denominator), e.g. over seeds."""
    values = np.asarray(values, float)
    return float(values.mean()), float(values.std(ddof=1))


def wald_ci(correct, level=0.95):
    """Normal-approximation confidence interval: p +/- z * sqrt(p (1 - p) / n)."""
    correct = np.asarray(correct, float)
    n, p = len(correct), correct.mean()
    z = NormalDist().inv_cdf(0.5 + level / 2)
    half = z * math.sqrt(p * (1 - p) / n)
    return float(p - half), float(p + half)


def bootstrap_ci(correct, n_boot, rng, level=0.95):
    """Bootstrap confidence interval: resample the images with replacement n_boot
    times and take quantiles of the resampled accuracies."""
    correct = np.asarray(correct, float)
    n = len(correct)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = correct[rng.integers(0, n, n)].mean()
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def paired_bootstrap(correct_a, correct_b, n_boot, rng, level=0.95):
    """Paired bootstrap test of model B against model A on the same images.

    Per image, d = b - a is +1 (B fixed it), -1 (B broke it) or 0. Resampling
    images, the same ones for both models, cancels the difficulty they share,
    which makes this far more sensitive than comparing two separate intervals.

    Returns {"diff" (B minus A), "ci_low", "ci_high", "p_value" (two-sided)}.
    """
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    d = b - a
    n = len(d)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = d[rng.integers(0, n, n)].mean()
    alpha = (1 - level) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p))}


def mcnemar_test(correct_a, correct_b):
    """McNemar's test for two classifiers on the same data.

    Only disagreements carry information: a_only = right for A and wrong for B,
    b_only = the reverse. If the models are equally good, each disagreement is a
    fair coin flip.

        statistic     = (|a_only - b_only| - 1)^2 / (a_only + b_only)   (chi-squared, 1 dof)
        p_value       = P(chi2_1 > statistic) = erfc(sqrt(statistic / 2))
        exact_p_value = 2 * P(Binomial(a_only + b_only, 1/2) <= min(a_only, b_only))
    """
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "statistic": 0.0, "p_value": 1.0, "exact_p_value": 1.0}
    stat = (abs(a_only - b_only) - 1) ** 2 / n
    p_chi2 = math.erfc(math.sqrt(stat / 2))
    k = min(a_only, b_only)
    exact = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": a_only, "b_only": b_only, "statistic": float(stat),
            "p_value": float(p_chi2), "exact_p_value": float(exact)}
