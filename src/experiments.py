"""Building models from configs and running experiments.

The ablation study starts from the original network and changes one thing at
a time (config.ABLATION_STEPS), training every step with several seeds, so
each change's effect can be separated from seed-to-seed noise.
"""
import copy
import json
import time
import numpy as np

from src.layers import Linear, ReLU, Sigmoid, Dropout, Flatten, Reshape
from src.conv import Conv2D, MaxPool2D
from src.network import Sequential
from src.losses import SoftmaxCrossEntropy
from src.optimizers import SGD, Adam
from src.train import train, evaluate
from src.robustness import random_shift

ACTIVATIONS = {"relu": ReLU, "sigmoid": Sigmoid}


def apply_steps(baseline, steps):
    """Turn the cumulative (name, changes) list into a list of (name, full config)."""
    configs, cfg = [], copy.deepcopy(baseline)
    for name, changes in steps:
        cfg = {**cfg, **copy.deepcopy(changes)}
        configs.append((name, cfg))
    return configs


def save_model(model, cfg, path):
    arrays = {key.replace(".", "__"): p for key, p, _ in model.parameters()}
    np.savez(path, __config__=np.array(json.dumps(cfg)), **arrays)


def load_model(path):
    data = np.load(path)
    cfg = json.loads(str(data["__config__"]))
    model = build_model(cfg, np.random.default_rng(0))
    model.load_state_dict({k.replace("__", "."): data[k] for k in data.files if k != "__config__"})
    return model, cfg


def build_mlp(cfg, rng):
    """784 -> [Linear -> activation -> (Dropout)] per hidden width -> Linear -> 10 logits."""
    layers, fan_in = [], 784
    for width in cfg["hidden"]:
        layers.append(Linear(fan_in, width, rng, init=cfg["init"]))
        layers.append(ACTIVATIONS[cfg["activation"]]())
        if cfg["dropout"] > 0:
            layers.append(Dropout(cfg["dropout"], rng))
        fan_in = width
    layers.append(Linear(fan_in, 10, rng, init=cfg["init"]))
    return Sequential(layers)


def build_cnn(cfg, rng):
    """Reshape to (1, 28, 28) -> [Conv -> activation -> 2x2 max-pool] per entry of
    cfg["conv_channels"] -> Flatten -> optional dense layers -> Linear -> 10 logits.
    Padding k // 2 keeps each conv's output the same size as its input."""
    layers = [Reshape((1, 28, 28))]
    channels, size, k = 1, 28, cfg["kernel_size"]
    for out_ch in cfg["conv_channels"]:
        layers += [Conv2D(channels, out_ch, k, k // 2, rng, init=cfg["init"]),
                   ACTIVATIONS[cfg["activation"]](), MaxPool2D(2)]
        channels, size = out_ch, size // 2
    layers.append(Flatten())
    fan_in = channels * size * size
    for width in cfg["hidden"]:
        layers += [Linear(fan_in, width, rng, init=cfg["init"]), ACTIVATIONS[cfg["activation"]]()]
        if cfg["dropout"] > 0:
            layers.append(Dropout(cfg["dropout"], rng))
        fan_in = width
    layers.append(Linear(fan_in, 10, rng, init=cfg["init"]))
    return Sequential(layers)


def build_model(cfg, rng):
    if cfg["arch"] == "mlp":
        return build_mlp(cfg, rng)
    if cfg["arch"] == "cnn":
        return build_cnn(cfg, rng)
    raise ValueError(f"unknown architecture {cfg['arch']!r}")


def make_optimizer(cfg):
    if cfg["optimizer"] == "sgd":
        return SGD(cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "adam":
        return Adam(cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError(f"unknown optimizer {cfg['optimizer']!r}")


def run_config(cfg, data, seed, verbose=False):
    """Train one model from one config and seed, then score it on the test set.

    A single generator built from the seed drives initialisation, dropout,
    batch order and augmentation, so (cfg, seed) fully determines the run.
    Early stopping uses the validation set; the test set is touched only at the
    end, to record which test images the model gets right.

    Returns {"model", "history", "val_acc", "test_acc", "test_correct", "seconds"}.
    """
    rng = np.random.default_rng(seed)
    model = build_model(cfg, rng)
    optimizer = make_optimizer(cfg)
    loss_fn = SoftmaxCrossEntropy()
    augment = None
    if cfg["augment_max_shift"]:
        max_shift = cfg["augment_max_shift"]
        augment = lambda xb, r: random_shift(xb, max_shift, r)
    t0 = time.time()
    history = train(model, loss_fn, optimizer, data["X_train"], data["y_train"],
                    data["X_val"], data["y_val"], cfg["epochs"], cfg["batch_size"], rng,
                    patience=cfg["early_stopping_patience"], augment=augment, verbose=verbose)
    seconds = time.time() - t0
    _, val_acc = evaluate(model, loss_fn, data["X_val"], data["y_val"])
    correct = model.predict(data["X_test"]) == data["y_test"]
    return {"model": model, "history": history, "val_acc": val_acc,
            "test_acc": float(correct.mean()), "test_correct": correct, "seconds": seconds}


def run_ablation(data, baseline, steps, seeds, verbose=True):
    """Run every cumulative ablation step once per seed.

    Returns one dict per step: {"name", "config", "runs": [run_config outputs]}.
    """
    results = []
    for name, cfg in apply_steps(baseline, steps):
        runs = []
        for seed in seeds:
            if verbose:
                print(f"[{name}] seed {seed}")
            runs.append(run_config(cfg, data, seed, verbose=False))
        accs = [r["test_acc"] for r in runs]
        if verbose and len(accs) > 1:
            print(f"  test acc {np.mean(accs):.4f} +/- {np.std(accs, ddof=1):.4f}")
        elif verbose:
            print(f"  test acc {accs[0]:.4f}")
        results.append({"name": name, "config": cfg, "runs": runs})
    return results


def lr_sweep(data, cfg, lrs, epochs, seed):
    """Short training runs over a range of learning rates, compared on validation
    accuracy. Very large rates diverge (inf/nan); the warnings are silenced and
    such runs simply score around 10%."""
    out = []
    for lr in lrs:
        run_cfg = {**cfg, "lr": lr, "epochs": epochs}
        with np.errstate(all="ignore"):
            r = run_config(run_cfg, data, seed)
        out.append({"lr": lr, "val_acc": r["val_acc"],
                    "final_train_loss": r["history"]["train_loss"][-1]})
    return out
