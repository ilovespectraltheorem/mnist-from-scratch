"""Every setting for the project lives here.

Experiments are changed by editing this file, never numbers inside src/.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"
DEMO_DIR = ROOT / "demo"

# ---------------------------------------------------------------- data
VAL_SIZE = 5000      # images held out from the 60,000 training images
SPLIT_SEED = 0       # fixed, so every experiment sees the same validation set
SEEDS = [0, 1, 2]    # every ablation step is trained once per seed

# ---------------------------------------------------------------- baseline
# The original network (legacy/) written as a config: 784-128-10, sigmoid,
# plain SGD, learning rate 0.1, batches of 100, ~17 passes over the data
# (10,000 steps x 100 images = 1,000,000 images = ~17 epochs of 60,000).
BASELINE = {
    "arch": "mlp",
    "hidden": [128],
    "activation": "sigmoid",
    "init": "lecun",
    "dropout": 0.0,
    "optimizer": "sgd",
    "lr": 0.1,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "batch_size": 100,
    "epochs": 17,
    "early_stopping_patience": None,
    "augment_max_shift": 0,
}

# ---------------------------------------------------------------- ablation study
# Each step lists only what changes relative to the PREVIOUS step, so the
# steps are cumulative.
ABLATION_STEPS = [
    ("Baseline", {}),
    ("ReLU", {"activation": "relu"}),
    ("He init", {"init": "he"}),
    # lr is cut 10x so that lr / (1 - momentum) stays 0.1: the same effective
    # step size as before, so this step measures momentum, not a bigger step.
    ("Momentum", {"momentum": 0.9, "lr": 0.01}),
    ("Adam", {"optimizer": "adam", "lr": 1e-3, "momentum": 0.0}),
    ("Wider (256)", {"hidden": [256]}),
    ("Dropout + early stop", {"dropout": 0.2, "epochs": 40,
                              "early_stopping_patience": 5}),
]

# Learning-rate sweep: plain SGD on the "He init" network, short runs.
LR_SWEEP_LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
LR_SWEEP_EPOCHS = 5

# ---------------------------------------------------------------- significance tests
N_BOOTSTRAP = 10_000
CONFIDENCE_LEVEL = 0.95

# ---------------------------------------------------------------- calibration
N_CALIBRATION_BINS = 15

# ---------------------------------------------------------------- robustness
SHIFTS = [0, 1, 2, 3, 4]           # pixels, for the accuracy-vs-shift curve
AUGMENT_MAX_SHIFT = 2              # random shift range used for augmentation
CNN = {
    "arch": "cnn",
    "conv_channels": [8, 16],      # conv -> ReLU -> 2x2 max-pool, per entry
    "kernel_size": 3,
    "hidden": [],                  # extra dense layers after the convs
    "activation": "relu",
    "init": "he",
    "dropout": 0.0,
    "optimizer": "adam",
    "lr": 1e-3,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "batch_size": 64,
    "epochs": 5,
    "early_stopping_patience": None,
    "augment_max_shift": 0,
}

# ---------------------------------------------------------------- web demo
DEMO_DECIMALS = 5                  # rounding for exported weights (file size)
