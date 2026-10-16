## Accuracy with uncertainty

| step | mean ± std over seeds | seed-0 accuracy | 95% CI (normal) | 95% CI (bootstrap) |
|---|---|---|---|---|
| Baseline | 94.51% ± 0.15 | 94.49% | [94.04, 94.94] | [94.04, 94.93] |
| ReLU | 97.56% ± 0.10 | 97.66% | [97.36, 97.96] | [97.35, 97.94] |
| He init | 97.55% ± 0.10 | 97.66% | [97.36, 97.96] | [97.35, 97.95] |
| Momentum | 97.52% ± 0.02 | 97.49% | [97.18, 97.80] | [97.18, 97.79] |
| Adam | 97.74% ± 0.04 | 97.79% | [97.50, 98.08] | [97.49, 98.07] |
| Wider (256) | 97.91% ± 0.11 | 98.03% | [97.76, 98.30] | [97.75, 98.29] |
| Dropout + early stop | 98.06% ± 0.08 | 98.08% | [97.81, 98.35] | [97.81, 98.34] |

## Is each step a real improvement? (paired tests on the same 10,000 test images, seed 0)

| comparison | change | 95% CI of change | paired bootstrap p | McNemar p (exact) | fixed / broken |
|---|---|---|---|---|---|
| Baseline → ReLU | +3.17 pts | [+2.80, +3.55] | 0.0000 | 5.5e-68 | 350 / 33 |
| ReLU → He init | +0.00 pts | [-0.15, +0.15] | 1.0000 | 1 | 30 / 30 |
| He init → Momentum | -0.17 pts | [-0.30, -0.04] | 0.0148 | 0.019 | 15 / 32 |
| Momentum → Adam | +0.30 pts | [+0.06, +0.55] | 0.0184 | 0.02 | 93 / 63 |
| Adam → Wider (256) | +0.24 pts | [+0.00, +0.48] | 0.0540 | 0.058 | 86 / 62 |
| Wider (256) → Dropout + early stop | +0.05 pts | [-0.16, +0.26] | 0.6814 | 0.71 | 61 / 56 |
| Baseline → Dropout + early stop | +3.59 pts | [+3.20, +3.99] | 0.0000 | 1.5e-78 | 393 / 34 |
| *Same config, seed 0 → seed 1 (noise floor)* | -0.11 pts | [-0.34, +0.12] | 0.3558 | 0.39 | 61 / 72 |
