"""Class-conditional (Mondrian) conformal prediction — the FLARE UQ layer.

Distribution-free, finite-sample coverage guarantee per class. Given
a calibration set (val predictions), the per-class LAC quantile q_c is the
(1-alpha) empirical quantile of the non-conformity score s = 1 - p_trueclass.
At test time, class c is in the prediction set iff p_c >= 1 - q_c.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from .taxonomy import BROAD_CLASSES, NUM_CLASSES


def calibrate(cal_probs: np.ndarray, cal_y: np.ndarray,
              alpha: float = 0.10) -> np.ndarray:
    """Per-class LAC quantiles from a calibration set."""
    q = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        s = 1.0 - cal_probs[cal_y == c, c]
        n = len(s)
        if n == 0:
            q[c] = 1.0
            continue
        k = int(np.ceil((n + 1) * (1 - alpha)))
        q[c] = 1.0 if k > n else np.sort(s)[k - 1]
    return q


def prediction_sets(probs: np.ndarray, qhat: np.ndarray) -> np.ndarray:
    """Boolean (N, C) membership matrix."""
    return probs >= (1.0 - qhat)[None, :]


def coverage_report(sets: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
    out = {}
    for c in range(NUM_CLASSES):
        m = y == c
        out[BROAD_CLASSES[c]] = {
            "n": int(m.sum()),
            "coverage": float(sets[m, c].mean()) if m.sum() else float("nan"),
            "avg_set_size": float(sets[m].sum(1).mean()) if m.sum() else float("nan"),
        }
    out["marginal"] = {
        "n": int(len(y)),
        "coverage": float(sets[np.arange(len(y)), y].mean()),
        "avg_set_size": float(sets.sum(1).mean()),
    }
    return out


def save(path: Path, qhat: np.ndarray, alpha: float, n_cal: int) -> None:
    Path(path).write_text(json.dumps({
        "alpha": alpha, "n_cal": int(n_cal),
        "classes": BROAD_CLASSES,
        "qhat": {BROAD_CLASSES[c]: float(qhat[c]) for c in range(NUM_CLASSES)},
        "method": "mondrian_lac",
    }, indent=2))


def load(path: Path) -> "tuple[np.ndarray, float]":
    d = json.loads(Path(path).read_text())
    q = np.array([d["qhat"][c] for c in BROAD_CLASSES])
    return q, d["alpha"]
