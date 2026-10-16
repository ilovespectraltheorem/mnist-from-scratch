"""Exporting the trained network for the web demo.

The weights are written to demo/model.js, and demo/index.html runs the same
forward pass in JavaScript.
"""
import json
import numpy as np
from src.layers import Linear, ReLU, Sigmoid, Dropout


def model_to_dict(model, sample_x, decimals=5):
    """Describe an MLP as plain lists and dicts:

        {"layers": [{"type": "linear", "W": [[...]], "b": [...]}, {"type": "relu"}, ...],
         "check":  {"x": [784 numbers], "probs": [10 numbers]}}

    W is stored (in, out) like numpy and rounded to `decimals` places to keep
    the file small. Dropout is skipped (it does nothing at inference) and any
    other layer type raises ValueError. "check" lets the page verify that its
    JavaScript reproduces Python's probabilities.
    """
    layers = []
    for layer in model.layers:
        if isinstance(layer, Linear):
            layers.append({"type": "linear",
                           "W": np.round(layer.params["W"], decimals).tolist(),
                           "b": np.round(layer.params["b"], decimals).tolist()})
        elif isinstance(layer, ReLU):
            layers.append({"type": "relu"})
        elif isinstance(layer, Sigmoid):
            layers.append({"type": "sigmoid"})
        elif isinstance(layer, Dropout):
            continue
        else:
            raise ValueError(f"cannot export layer {type(layer).__name__}")
    probs = model.predict_proba(np.asarray(sample_x)[None])[0]
    return {"layers": layers,
            "check": {"x": np.asarray(sample_x, float).tolist(), "probs": probs.tolist()}}


def export_model_js(model, path, sample_x, decimals=5):
    """Write `window.MODEL = {...};`, which a page can load with a plain <script>
    tag, so the demo also works when opened straight from disk."""
    d = model_to_dict(model, sample_x, decimals)
    with open(path, "w") as f:
        f.write("window.MODEL = ")
        json.dump(d, f, separators=(",", ":"))
        f.write(";\n")
