"""Venn-Abers probability intervals for the FLARE classes.

For each class c (one against the rest), an inductive Venn-Abers predictor fits isotonic
regression to the calibration scores p_c twice, once with the test object appended with label 0
and once with label 1, and returns the interval [p0, p1] for P(class = c) (Vovk & Petej 2014).
The isotonic fit depends only on the rank of the test score among the calibration scores, so
p0 and p1 are step functions of the score; `build_table` evaluates them once per gap between
consecutive calibration scores and stores only the points where they change, and `intervals`
reads them back with a binary search.

The calibration scores are the out-of-fold probabilities of the five-fold cross-fit that also
calibrates the conformal sets (scripts/export_venn_abers.py). The test objects are scored by the
model trained on the whole pool, so, as for the conformal sets, the validity of the intervals is
checked empirically out of fold and not claimed as a finite-sample guarantee.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_table(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Step functions p0(s), p1(s) for one class: `edges` are calibration scores, and segment j
    (scores between edges[j-1] and edges[j]) has interval [p0[j], p1[j]]."""
    order = np.argsort(scores, kind="mergesort")
    s, y = np.asarray(scores, float)[order], np.asarray(labels, float)[order]
    from sklearn.isotonic import IsotonicRegression
    n = len(s)
    p0, p1 = np.empty(n + 1), np.empty(n + 1)
    x = np.arange(n + 1, dtype=float)            # isotonic fit depends only on ranks
    ir = IsotonicRegression(y_min=0.0, y_max=1.0)
    for i in range(n + 1):                       # test score inserted at rank i
        p0[i] = ir.fit(x, np.r_[y[:i], 0.0, y[i:]]).predict([float(i)])[0]
        p1[i] = ir.fit(x, np.r_[y[:i], 1.0, y[i:]]).predict([float(i)])[0]
    keep = np.r_[True, (np.diff(p0) != 0) | (np.diff(p1) != 0)]   # segments where the interval changes
    idx = np.flatnonzero(keep)
    edges = s[idx[1:] - 1]                       # score at which segment idx[k] begins
    return dict(edges=np.round(edges, 7).tolist(), p0=np.round(p0[idx], 6).tolist(),
                p1=np.round(p1[idx], 6).tolist(), n_cal=int(n), n_pos=int(y.sum()))


def intervals(probs: np.ndarray, table: dict, classes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower, upper and merged probability for every class; probs is (n, n_classes)."""
    probs = np.atleast_2d(np.asarray(probs, float))
    L, U = np.empty_like(probs), np.empty_like(probs)
    for j, c in enumerate(classes):
        t = table[c]
        k = np.searchsorted(np.asarray(t["edges"]), probs[:, j], side="right")
        L[:, j], U[:, j] = np.asarray(t["p0"])[k], np.asarray(t["p1"])[k]
    M = U / (1.0 - L + U)
    return L, U, M / M.sum(axis=1, keepdims=True)


def load_table(path: str | Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text())["classes"] if path.exists() else None
