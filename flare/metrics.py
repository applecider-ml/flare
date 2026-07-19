"""Evaluation metrics with bootstrap confidence intervals.

Accuracy, balanced accuracy, macro F1, macro AUPRC (one-vs-rest average
precision, macro-averaged), ECE (15 equal-width confidence bins), and
per-class recall, with stratified-bootstrap 95% confidence intervals.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, recall_score)

from .taxonomy import BROAD_CLASSES, NUM_CLASSES


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        out += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def macro_auprc(y_true: np.ndarray, probs: np.ndarray) -> float:
    aps = []
    for c in range(NUM_CLASSES):
        yc = (y_true == c).astype(int)
        if yc.sum() == 0:
            continue
        aps.append(average_precision_score(yc, probs[:, c]))
    return float(np.mean(aps))


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "macro_auprc": macro_auprc(y_true, probs),
        "ece": ece(y_true, probs),
    }
    rec = recall_score(y_true, pred, average=None,
                       labels=list(range(NUM_CLASSES)), zero_division=0)
    for i, c in enumerate(BROAD_CLASSES):
        out[f"recall_{c}"] = float(rec[i])
    return out


def bootstrap_metrics(y_true: np.ndarray, probs: np.ndarray,
                      n_boot: int = 2000, seed: int = 0,
                      stratified: bool = True) -> Dict[str, Dict[str, float]]:
    """Point estimate + percentile 95% CI for every metric.

    Stratified resampling (within-class) keeps every class present in each
    replicate — without it the 9-object TDE class vanishes from ~37% of
    replicates and recall_TDE/balanced_accuracy CIs are biased.
    """
    rng = np.random.default_rng(seed)
    point = compute_metrics(y_true, probs)
    idx_by_class = [np.where(y_true == c)[0] for c in range(NUM_CLASSES)]
    samples: Dict[str, list] = {k: [] for k in point}
    n = len(y_true)
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                                  for ix in idx_by_class if len(ix) > 0])
        else:
            idx = rng.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], probs[idx])
        for k, v in m.items():
            samples[k].append(v)
    out = {}
    for k, v in samples.items():
        arr = np.asarray(v)
        out[k] = {
            "point": point[k],
            "lo": float(np.percentile(arr, 2.5)),
            "hi": float(np.percentile(arr, 97.5)),
            "std": float(arr.std()),
        }
    return out


def format_metrics_table(boot: Dict[str, Dict[str, float]]) -> str:
    """Markdown table of point estimates with bootstrap 95% CIs."""
    lines = ["| Metric | Value [95% CI] |", "|---|---|"]
    for k, v in boot.items():
        lines.append(f"| {k} | {v['point']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] |")
    return "\n".join(lines)

