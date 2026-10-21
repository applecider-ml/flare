"""Review priority from human confirmations: the supervised stage of the anomaly layer.

The blind consensus (flare.consensus) and its false-alarm rate per unit time (flare.lifetime) need no
labels. Once operators confirm or reject candidates on Fritz, a class-balanced logistic combiner over the
fourteen LUNA scores of the full-curve consensus is refitted on those confirmations
(scripts/fit_review_combiner.py) and ranks the candidates for review, as the review loop of the FLARE paper
does. The combiner orders the review queue; the alarm and its false-alarm rate stay with the blind
consensus, so a combiner trained on a few families cannot hide an object of a new family from the alarm.

The combiner is a small JSON file: the scorer names in order, the standardisation, the coefficients, the
sorted combiner values of held-out known-class objects (for a conformal review p-value), and the counts of
confirmations per family. It is read from FLARE_REVIEW_COMBINER when set (so a nightly refit needs no
new image), else from the model directory (``review_combiner.json``), and is used only when it carries
confirmations from at least MIN_FAMILIES families, since a combiner fitted on one family learns that family
instead of unfamiliarity (FLARE paper, Sec. 5.2).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FILENAME = "review_combiner.json"
MIN_FAMILIES = 3


class ReviewCombiner:
    def __init__(self, table: dict):
        self.names = list(table["scorers"])
        self.mean = np.asarray(table["mean"], dtype=float)
        self.scale = np.asarray(table["scale"], dtype=float)
        self.coef = np.asarray(table["coef"], dtype=float)
        self.intercept = float(table["intercept"])
        self.ref = np.sort(np.asarray(table["reference_known"], dtype=float))
        self.families = dict(table.get("families", {}))
        self.meta = {k: v for k, v in table.items() if k not in ("reference_known",)}

    @property
    def usable(self) -> bool:
        return sum(1 for f, n in self.families.items() if n > 0) >= MIN_FAMILIES

    def score(self, scores: dict) -> dict | None:
        """Review score and its conformal p-value against held-out known objects (small = review first)."""
        if not all(n in scores for n in self.names):
            return None
        x = np.array([scores[n] for n in self.names], dtype=float)
        x = np.nan_to_num((x - self.mean) / self.scale)
        s = float(x @ self.coef + self.intercept)
        p = (1.0 + len(self.ref) - np.searchsorted(self.ref, s, side="left")) / (len(self.ref) + 1.0)
        return {"review_score": round(s, 4), "review_p": round(float(p), 5),
                "n_confirmations": int(sum(self.families.values())), "fitted": self.meta.get("fitted")}


def load(model_dir: str | Path) -> ReviewCombiner | None:
    path = os.environ.get("FLARE_REVIEW_COMBINER") or str(Path(model_dir) / FILENAME)
    if not Path(path).exists():
        return None
    rc = ReviewCombiner(json.loads(Path(path).read_text()))
    return rc if rc.usable else None
