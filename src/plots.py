"""Figures for the README. Every function takes plain numbers/arrays and saves one PNG."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Palette: light surface, quiet ink, categorical slots in a fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "font.size": 10, "text.color": INK,
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ training
def learning_curves(history, path, title="Learning curves"):
    _style()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(epochs, history["train_loss"], color=SERIES[0], label="train loss")
    ax1.plot(epochs, history["val_loss"], color=SERIES[1], label="validation loss")
    ax1.set_xlabel("epoch"), ax1.set_ylabel("cross-entropy loss")
    ax1.set_title(title)
    ax1.legend()
    ax2.plot(epochs, 100 * np.array(history["val_acc"]), color=SERIES[0])
    ax2.set_xlabel("epoch"), ax2.set_ylabel("validation accuracy (%)")
    ax2.set_title("Validation accuracy")
    if history.get("best_epoch") is not None:
        b = history["best_epoch"]
        for ax, ys in ((ax1, history["val_loss"]), (ax2, 100 * np.array(history["val_acc"]))):
            ax.plot(b + 1, ys[b], "o", color=SERIES[1] if ax is ax1 else SERIES[0],
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax1.annotate("best (weights restored)", (b + 1, history["val_loss"][b]),
                     textcoords="offset points", xytext=(6, 10), color=INK_2, fontsize=9)
    _save(fig, path)


# ------------------------------------------------------------------ ablation
def ablation_waterfall(names, mean_err, std_err, path):
    """Test error (%) falling step by step. First and last bars are totals."""
    _style()
    mean_err, std_err = np.asarray(mean_err), np.asarray(std_err)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(7, 1.25 * (n + 1)), 4.2))
    labels = list(names) + ["Final"]
    x = np.arange(n + 1)
    w = 0.55
    ax.bar(0, mean_err[0], w, color=INK_2, zorder=3)
    ax.text(0, mean_err[0] + std_err[0] + 0.08, f"{mean_err[0]:.2f}%", ha="center", color=INK, fontsize=9)
    for i in range(1, n):
        prev, cur = mean_err[i - 1], mean_err[i]
        color = SERIES[0] if cur <= prev else SERIES[1]
        ax.bar(i, abs(cur - prev), w, bottom=min(prev, cur), color=color, zorder=3)
        ax.plot([i - 1 + w / 2, i - w / 2], [prev, prev], color=AXIS, linewidth=1, zorder=2)
        ax.text(i, max(prev, cur) + 0.08, f"{cur - prev:+.2f}", ha="center", color=INK_2, fontsize=9)
    ax.plot([n - 1 + w / 2, n - w / 2], [mean_err[-1], mean_err[-1]], color=AXIS, linewidth=1, zorder=2)
    ax.bar(n, mean_err[-1], w, color=INK_2, zorder=3)
    ax.text(n, mean_err[-1] + std_err[-1] + 0.08, f"{mean_err[-1]:.2f}%", ha="center", color=INK, fontsize=9)
    ax.errorbar([0, n], [mean_err[0], mean_err[-1]], yerr=[std_err[0], std_err[-1]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("test error (%)")
    ax.set_ylim(0, mean_err[0] * 1.2 + std_err[0])
    ax.grid(axis="x", visible=False)
    ax.set_title("Where the accuracy came from (mean of seeds; whiskers = 1 std)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK_2, label="total"), Patch(color=SERIES[0], label="error falls"),
                       Patch(color=SERIES[1], label="error rises")], loc="upper right")
    _save(fig, path)


def lr_sweep_plot(rows, path):
    _style()
    lrs = np.array([r["lr"] for r in rows])
    acc = np.array([100 * r["val_acc"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(lrs, acc, color=SERIES[0], marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    best = int(np.argmax(acc))
    ax.annotate(f"best {acc[best]:.1f}% at lr={lrs[best]:g}", (lrs[best], acc[best]),
                textcoords="offset points", xytext=(0, -16), ha="center", va="top",
                color=INK_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Learning-rate sweep (plain SGD)")
    _save(fig, path)


# ------------------------------------------------------------------ error analysis and calibration
def confusion_heatmap(cm, path):
    """Mistakes only: the diagonal is blanked so the errors are visible."""
    _style()
    errors = cm.astype(float).copy()
    np.fill_diagonal(errors, np.nan)
    errors[errors == 0] = np.nan            # no mistakes -> blank cell
    fig, ax = plt.subplots(figsize=(6, 5.2))
    cmap = SEQUENTIAL.copy()
    cmap.set_bad(SURFACE)
    im = ax.imshow(errors, cmap=cmap, vmin=0, vmax=np.nanmax(errors))
    ax.grid(False)
    k = cm.shape[0]
    ax.set_xticks(range(k)), ax.set_yticks(range(k))
    ax.set_xlabel("predicted digit"), ax.set_ylabel("true digit")
    vmax = np.nanmax(errors)
    for i in range(k):
        for j in range(k):
            if i == j:
                ax.text(j, i, f"{cm[i, i]}", ha="center", va="center", fontsize=7, color=MUTED)
            elif cm[i, j] > 0:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.55 * vmax else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="test images")
    ax.set_title("Confusion matrix: mistakes only")
    _save(fig, path)


def misclassified_gallery(images, y_true, y_pred, confidence, path, n=24):
    _style()
    n = min(n, len(images))
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.55))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].reshape(28, 28), cmap="gray_r")
            ax.set_title(f"{y_true[i]}→{y_pred[i]}  {100 * confidence[i]:.0f}%", fontsize=8,
                         color=INK_2, loc="center", fontweight="normal")
    fig.suptitle("Most confident mistakes (true→predicted, model confidence)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    _save(fig, path)


def reliability_diagram(before, after, ece_before, ece_after, T, path, min_count=20):
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, label="perfect calibration")
    for bins, color, label in ((before, SERIES[1], f"before (ECE {100 * ece_before:.2f}%)"),
                               (after, SERIES[0], f"after T={T:.2f} (ECE {100 * ece_after:.2f}%)")):
        keep = bins["count"] >= min_count
        ax.plot(bins["confidence"][keep], bins["accuracy"][keep], color=color, marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    ax.set_xlabel("confidence (max predicted probability)")
    ax.set_ylabel("accuracy")
    shown = [b[k][b["count"] >= min_count] for b in (before, after) for k in ("confidence", "accuracy")]
    lo = min(min(np.nanmin(v) for v in shown if len(v)), 0.5)
    ax.set_xlim(lo - 0.02, 1.01), ax.set_ylim(lo - 0.02, 1.01)
    ax.set_title("Reliability diagram (test set)")
    ax.legend(loc="upper left")
    _save(fig, path)


def first_layer_weights(W, path, n=32):
    """Each hidden unit's 784 incoming weights, drawn as a 28x28 picture."""
    _style()
    n = min(n, W.shape[1])
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            w = W[:, i].reshape(28, 28)
            m = np.abs(w).max()
            ax.imshow(w, cmap=DIVERGING, vmin=-m, vmax=m)
    fig.suptitle("First-layer weights (blue = negative, red = positive)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    _save(fig, path)


def pca_small_multiples(projected, labels, ratio, path):
    """One panel per digit: that digit in blue, every other digit in grey."""
    _style()
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6), sharex=True, sharey=True)
    for d, ax in enumerate(axes.ravel()):
        ax.scatter(projected[:, 0], projected[:, 1], s=3, color=GRID, linewidths=0)
        sel = labels == d
        ax.scatter(projected[sel, 0], projected[sel, 1], s=4, color=SERIES[0], linewidths=0)
        ax.set_title(f"digit {d}", fontsize=10)
        ax.grid(False)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"Hidden layer, projected onto its first 2 principal components "
                 f"({100 * ratio[0]:.0f}% + {100 * ratio[1]:.0f}% of variance)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    _save(fig, path)


# ------------------------------------------------------------------ robustness
def shift_curves(shifts, curves, path):
    """curves: dict name -> accuracies (one per shift)."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (name, accs) in enumerate(curves.items()):
        accs = 100 * np.asarray(accs)
        ax.plot(shifts, accs, color=SERIES[i], marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        ax.annotate(f"{accs[-1]:.0f}%", (shifts[-1], accs[-1]), textcoords="offset points",
                    xytext=(8, -3), color=INK_2, fontsize=9)
    ax.set_xticks(shifts)
    ax.set_xlabel("shift (pixels, averaged over up/down/left/right)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(shifts[0] - 0.2, shifts[-1] + 0.6)
    ax.set_title("How far can a digit move before the model fails?")
    ax.legend(loc="lower left")
    _save(fig, path)
