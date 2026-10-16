"""Run every experiment from the command line.

    python main.py gradcheck            check backprop against numerical gradients
    python main.py train                train the original network, plot learning curves
    python main.py train --config final train the last ablation step instead
    python main.py ablation             every ablation step x every seed
    python main.py lr-sweep             learning-rate sweep
    python main.py stats                confidence intervals + significance tests
    python main.py analyse              confusion matrix, mistakes, calibration, weights, PCA
    python main.py robustness           accuracy vs shift: MLP, MLP + augmentation, CNN
    python main.py export               write demo/model.js for the web demo
    python main.py draw                 draw digits in a window and see predictions
    python main.py all                  everything, in order

Results go to results/ (numbers as JSON/Markdown, figures in results/figures/).
"""
import argparse
import json
import time
import numpy as np

import config
from src.data import get_data
from src import plots

FIG = config.FIG_DIR
RES = config.RESULTS_DIR
MOD = config.MODEL_DIR


def _setup():
    for d in (RES, FIG, MOD, config.DEMO_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _data():
    t = time.time()
    data = get_data(config.DATA_DIR, config.VAL_SIZE, config.SPLIT_SEED)
    print(f"Loaded MNIST: {len(data['X_train'])} train / {len(data['X_val'])} val / "
          f"{len(data['X_test'])} test images ({time.time() - t:.1f}s)")
    return data


def _final_config():
    from src.experiments import apply_steps
    return apply_steps(config.BASELINE, config.ABLATION_STEPS)[-1][1]


def _load(name):
    from src.experiments import load_model
    path = MOD / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python main.py ablation` (or `train --config final`) first.")
    return load_model(path)


# ------------------------------------------------------------------ the library and single runs
def cmd_gradcheck(args):
    from src.gradcheck import gradient_check
    from src.layers import Linear, ReLU, Sigmoid
    from src.network import Sequential
    from src.losses import SoftmaxCrossEntropy
    rng = np.random.default_rng(0)
    nets = {
        "MLP, ReLU": lambda: Sequential([Linear(12, 8, rng), ReLU(), Linear(8, 6, rng), ReLU(), Linear(6, 4, rng)]),
        "MLP, sigmoid": lambda: Sequential([Linear(12, 8, rng), Sigmoid(), Linear(8, 4, rng)]),
    }
    try:
        from src.conv import Conv2D, MaxPool2D
        from src.layers import Flatten, Reshape
        nets["CNN"] = lambda: Sequential([Reshape((1, 6, 6)), Conv2D(1, 3, 3, 1, rng), ReLU(), MaxPool2D(2),
                                          Flatten(), Linear(27, 4, rng)])
    except ImportError:
        pass
    lines = ["| network | parameter | relative error |", "|---|---|---|"]
    for name, make in nets.items():
        X = rng.standard_normal((6, 36 if name == "CNN" else 12))
        y = rng.integers(0, 4, 6)
        try:
            errors = gradient_check(make(), SoftmaxCrossEntropy(), X, y)
        except NotImplementedError as e:
            print(f"{name}: skipped (not implemented yet: {e})")
            continue
        for key, err in errors.items():
            flag = "ok" if err < 1e-6 else "CHECK THIS"
            print(f"{name:14s} {key:8s} {err:.2e}  {flag}")
            lines.append(f"| {name} | {key} | {err:.1e} |")
    (RES / "gradcheck.md").write_text("\n".join(lines) + "\n")
    print(f"Table saved to {RES / 'gradcheck.md'}")


def cmd_train(args):
    from src.experiments import run_config, save_model
    configs = {"baseline": config.BASELINE, "final": _final_config(), "cnn": config.CNN}
    cfg = dict(configs[args.config])
    if args.epochs:
        cfg["epochs"] = args.epochs
    data = _data()
    print(f"Training '{args.config}' for up to {cfg['epochs']} epochs")
    out = run_config(cfg, data, seed=0, verbose=True)
    print(f"Validation accuracy {out['val_acc']:.4f}   Test accuracy {out['test_acc']:.4f}   "
          f"({out['seconds']:.0f}s, {out['model'].num_params():,} parameters)")
    save_model(out["model"], cfg, MOD / f"{args.config}.npz")
    plots.learning_curves(out["history"], FIG / f"learning_curves_{args.config}.png",
                          title=f"Learning curves: {args.config}")
    print(f"Saved model and {FIG / f'learning_curves_{args.config}.png'}")


# ------------------------------------------------------------------ experiments and analysis
def cmd_ablation(args):
    from src.experiments import run_ablation, save_model
    data = _data()
    results = run_ablation(data, config.BASELINE, config.ABLATION_STEPS, config.SEEDS, verbose=True)
    summary, correct = [], {}
    for i, step in enumerate(results):
        accs = [r["test_acc"] for r in step["runs"]]
        summary.append({"name": step["name"], "config": step["config"], "test_acc": accs,
                        "val_acc": [r["val_acc"] for r in step["runs"]],
                        "seconds": [r["seconds"] for r in step["runs"]],
                        "epochs_run": [len(r["history"]["val_loss"]) for r in step["runs"]]})
        for seed, r in zip(config.SEEDS, step["runs"]):
            correct[f"{i}_{seed}"] = r["test_correct"]
    (RES / "ablation.json").write_text(json.dumps({"seeds": config.SEEDS, "steps": summary}, indent=2))
    np.savez(RES / "ablation_correct.npz", **correct)
    save_model(results[0]["runs"][0]["model"], results[0]["config"], MOD / "baseline.npz")
    save_model(results[-1]["runs"][0]["model"], results[-1]["config"], MOD / "final.npz")
    plots.learning_curves(results[-1]["runs"][0]["history"], FIG / "learning_curves_final.png",
                          title=f"Learning curves: {results[-1]['name']}")
    names = [s["name"] for s in summary]
    err = [100 * (1 - np.mean(s["test_acc"])) for s in summary]
    std = [100 * np.std(s["test_acc"], ddof=1) if len(s["test_acc"]) > 1 else 0 for s in summary]
    plots.ablation_waterfall(names, err, std, FIG / "ablation_waterfall.png")
    print("\n| step | test accuracy (mean ± std over seeds) |\n|---|---|")
    for n_, s in zip(names, summary):
        print(f"| {n_} | {100 * np.mean(s['test_acc']):.2f}% ± {100 * np.std(s['test_acc'], ddof=1):.2f} |")
    print(f"\nSaved results/ablation.json and {FIG / 'ablation_waterfall.png'}")


def cmd_lr_sweep(args):
    from src.experiments import apply_steps, lr_sweep
    cfg = dict(apply_steps(config.BASELINE, config.ABLATION_STEPS)[2][1])   # the "He init" step
    data = _data()
    rows = lr_sweep(data, cfg, config.LR_SWEEP_LRS, config.LR_SWEEP_EPOCHS, seed=0)
    for r in rows:
        print(f"lr {r['lr']:<7g} val acc {r['val_acc']:.4f}  final train loss {r['final_train_loss']:.4f}")
    (RES / "lr_sweep.json").write_text(json.dumps(rows, indent=2))
    plots.lr_sweep_plot(rows, FIG / "lr_sweep.png")
    print(f"Saved {FIG / 'lr_sweep.png'}")


def cmd_stats(args):
    from src.stats import mean_std, wald_ci, bootstrap_ci, paired_bootstrap, mcnemar_test
    path = RES / "ablation_correct.npz"
    if not path.exists():
        raise SystemExit("Run `python main.py ablation` first.")
    summary = json.loads((RES / "ablation.json").read_text())
    correct = np.load(path)
    seeds = summary["seeds"]
    steps = summary["steps"]
    rng = np.random.default_rng(0)
    s0 = seeds[0]
    lines = ["## Accuracy with uncertainty", "",
             "| step | mean ± std over seeds | seed-0 accuracy | 95% CI (normal) | 95% CI (bootstrap) |",
             "|---|---|---|---|---|"]
    for i, step in enumerate(steps):
        m, s = mean_std(step["test_acc"])
        c = correct[f"{i}_{s0}"]
        lo, hi = wald_ci(c, config.CONFIDENCE_LEVEL)
        blo, bhi = bootstrap_ci(c, config.N_BOOTSTRAP, rng, config.CONFIDENCE_LEVEL)
        lines.append(f"| {step['name']} | {100 * m:.2f}% ± {100 * s:.2f} | {100 * c.mean():.2f}% | "
                     f"[{100 * lo:.2f}, {100 * hi:.2f}] | [{100 * blo:.2f}, {100 * bhi:.2f}] |")
    lines += ["", "## Is each step a real improvement? (paired tests on the same 10,000 test images, seed "
              f"{s0})", "", "| comparison | change | 95% CI of change | paired bootstrap p | McNemar p (exact) | "
              "fixed / broken |", "|---|---|---|---|---|---|"]
    pairs = [(i - 1, i) for i in range(1, len(steps))] + [(0, len(steps) - 1)]
    for a, b in pairs:
        ca, cb = correct[f"{a}_{s0}"], correct[f"{b}_{s0}"]
        pb = paired_bootstrap(ca, cb, config.N_BOOTSTRAP, rng, config.CONFIDENCE_LEVEL)
        mc = mcnemar_test(ca, cb)
        lines.append(f"| {steps[a]['name']} → {steps[b]['name']} | {100 * pb['diff']:+.2f} pts | "
                     f"[{100 * pb['ci_low']:+.2f}, {100 * pb['ci_high']:+.2f}] | {pb['p_value']:.4f} | "
                     f"{mc['exact_p_value']:.2g} | {mc['b_only']} / {mc['a_only']} |")
    if len(seeds) > 1:
        last = len(steps) - 1
        ca, cb = correct[f"{last}_{seeds[0]}"], correct[f"{last}_{seeds[1]}"]
        pb = paired_bootstrap(ca, cb, config.N_BOOTSTRAP, rng, config.CONFIDENCE_LEVEL)
        mc = mcnemar_test(ca, cb)
        lines.append(f"| *Same config, seed {seeds[0]} → seed {seeds[1]} (noise floor)* | "
                     f"{100 * pb['diff']:+.2f} pts | [{100 * pb['ci_low']:+.2f}, {100 * pb['ci_high']:+.2f}] | "
                     f"{pb['p_value']:.4f} | {mc['exact_p_value']:.2g} | {mc['b_only']} / {mc['a_only']} |")
    text = "\n".join(lines) + "\n"
    (RES / "significance.md").write_text(text)
    print(text)
    print(f"Saved {RES / 'significance.md'}")


def cmd_analyse(args):
    from src.analysis import confusion_matrix, precision_recall, top_confusions, pca, hidden_activations
    from src.calibration import reliability_bins, expected_calibration_error, fit_temperature
    from src.losses import softmax
    from src.layers import Linear
    model, cfg = _load("final")
    data = _data()
    X, y = data["X_test"], data["y_test"]
    logits = model.logits(X)
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y, pred, 10)
    precision, recall = precision_recall(cm)
    print("Most common mistakes (true -> predicted: count):")
    for t, p, c in top_confusions(cm, 5):
        print(f"  {t} -> {p}: {c}")
    print("digit  precision  recall")
    for d in range(10):
        print(f"  {d}    {precision[d]:.4f}    {recall[d]:.4f}")
    plots.confusion_heatmap(cm, FIG / "confusion_matrix.png")
    wrong = np.where(pred != y)[0]
    order = wrong[np.argsort(-probs[wrong].max(axis=1))]
    plots.misclassified_gallery(X[order], y[order], pred[order], probs[order].max(axis=1),
                                FIG / "misclassified.png")
    # calibration: fit T on validation, judge on test
    T = fit_temperature(model.logits(data["X_val"]), data["y_val"])
    probs_T = softmax(logits / T)
    nb = config.N_CALIBRATION_BINS
    ece0, ece1 = expected_calibration_error(probs, y, nb), expected_calibration_error(probs_T, y, nb)
    print(f"Mean confidence {probs.max(axis=1).mean():.4f} vs accuracy {(pred == y).mean():.4f}")
    print(f"ECE before {100 * ece0:.2f}%  ->  after temperature T={T:.3f}: {100 * ece1:.2f}%")
    plots.reliability_diagram(reliability_bins(probs, y, nb), reliability_bins(probs_T, y, nb),
                              ece0, ece1, T, FIG / "reliability.png")
    first = next(layer for layer in model.layers if isinstance(layer, Linear))
    plots.first_layer_weights(first.params["W"], FIG / "first_layer_weights.png")
    n_hidden = 2 if cfg["arch"] == "mlp" else len(model.layers) - 1
    H = hidden_activations(model, X[:3000], n_hidden)
    proj, _, ratio = pca(H, 2)
    plots.pca_small_multiples(proj, y[:3000], ratio, FIG / "pca_hidden.png")
    out = {"top_confusions": [list(map(int, t)) for t in top_confusions(cm, 5)],
           "precision": precision.tolist(), "recall": recall.tolist(),
           "temperature": T, "ece_before": ece0, "ece_after": ece1,
           "mean_confidence": float(probs.max(axis=1).mean()), "accuracy": float((pred == y).mean())}
    (RES / "analysis.json").write_text(json.dumps(out, indent=2))
    print(f"Saved results/analysis.json and 5 figures in {FIG}")


# ------------------------------------------------------------------ robustness and demo
def cmd_robustness(args):
    from src.experiments import run_config, save_model
    from src.robustness import accuracy_vs_shift
    data = _data()
    final = _final_config()
    models = {"MLP": final, "MLP + shift augmentation": {**final, "augment_max_shift": config.AUGMENT_MAX_SHIFT}}
    if not args.skip_cnn:
        models["CNN (NumPy, from scratch)"] = config.CNN
    if args.cnn_aug:
        models["CNN + shift augmentation"] = {**config.CNN, "augment_max_shift": config.AUGMENT_MAX_SHIFT}
    files = {"MLP": "final_rob", "MLP + shift augmentation": "mlp_aug", "CNN (NumPy, from scratch)": "cnn",
             "CNN + shift augmentation": "cnn_aug"}
    curves, rows = {}, []
    for name, cfg in models.items():
        if name == "MLP" and (MOD / "final.npz").exists():
            print("MLP: reusing results/models/final.npz from the ablation")
            model, seconds = _load("final")[0], None
        else:
            print(f"Training {name} ...")
            out = run_config(cfg, data, seed=0, verbose=True)
            model, seconds = out["model"], out["seconds"]
            save_model(model, cfg, MOD / f"{files[name]}.npz")
        accs = accuracy_vs_shift(model, data["X_test"], data["y_test"], config.SHIFTS)
        curves[name] = accs
        rows.append({"model": name, "accuracy_by_shift": accs.tolist(), "seconds": seconds})
        print(f"  accuracy at shifts {config.SHIFTS}: {np.round(100 * accs, 1)}")
    (RES / "robustness.json").write_text(json.dumps({"shifts": config.SHIFTS, "rows": rows}, indent=2))
    plots.shift_curves(config.SHIFTS, curves, FIG / "shift_robustness.png")
    print(f"Saved {FIG / 'shift_robustness.png'}")


def cmd_export(args):
    from src.experiments import load_model
    from src.export import export_model_js
    path = MOD / "mlp_aug.npz"
    if not path.exists():
        path = MOD / "final.npz"
    if not path.exists():
        raise SystemExit("No trained MLP found. Run `python main.py robustness` or `ablation` first.")
    model, cfg = load_model(path)
    data = _data()
    out = config.DEMO_DIR / "model.js"
    export_model_js(model, out, data["X_test"][0], config.DEMO_DECIMALS)
    print(f"Exported {path.name} to {out} ({out.stat().st_size / 1e6:.1f} MB). "
          "Open demo/index.html in a browser.")


def cmd_draw(args):
    import draw_app
    path = MOD / "mlp_aug.npz"
    draw_app.main(path if path.exists() else MOD / "final.npz")


def cmd_all(args):
    for f in (cmd_gradcheck, cmd_ablation, cmd_lr_sweep, cmd_stats, cmd_analyse, cmd_robustness, cmd_export):
        print(f"\n===== {f.__name__[4:]} =====")
        f(args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["gradcheck", "train", "ablation", "lr-sweep", "stats", "analyse",
                                       "robustness", "export", "draw", "all"])
    p.add_argument("--config", default="baseline", choices=["baseline", "final", "cnn"])
    p.add_argument("--epochs", type=int, default=None, help="override the number of epochs (train only)")
    p.add_argument("--skip-cnn", action="store_true", help="robustness without the CNN")
    p.add_argument("--cnn-aug", action="store_true", help="robustness: also train a CNN with augmentation")
    args = p.parse_args()
    _setup()
    cmds = {"gradcheck": cmd_gradcheck, "train": cmd_train, "ablation": cmd_ablation,
            "lr-sweep": cmd_lr_sweep, "stats": cmd_stats, "analyse": cmd_analyse,
            "robustness": cmd_robustness, "export": cmd_export, "draw": cmd_draw, "all": cmd_all}
    cmds[args.command](args)


if __name__ == "__main__":
    main()
