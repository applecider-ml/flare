"""Blind-consensus anomaly score for the shipped models.

The consensus is the mean percentile rank of an object across the LUNA scorers (energy,
Mahalanobis, kNN, density and hybrid scores) in the anomaly space of the model that classifies
it. The percentile ranks are frozen on known-class objects and the false-alarm thresholds are
set on separate known-class objects (scripts/export_consensus.py), so the score of an alert does
not depend on the other alerts scored with it. On the BTS benchmark the consensus detects more
of the out-of-taxonomy objects than the energy alone at every horizon and at 1%, 5% and 10%
false-alarm rates.

Each model directory ships `consensus.pkl.gz`, whose sha256 is checked against the directory's
manifest before it is unpickled. LUNA (luna-ood >= 0.3) must be installed to read it; without it,
or without the file, `load` returns None and FLARE reports the energy alarm alone.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

FILENAME = "consensus.pkl.gz"


class Consensus:
    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.columns = list(bundle["columns"])
        self.median = np.asarray(bundle["median"], dtype=float)
        self.iqr = np.asarray(bundle["iqr"], dtype=float)
        self.fars = tuple(float(a) for a in bundle["fars"])
        self.note = bundle.get("note", "")

    def embed(self, X) -> np.ndarray:
        """The training-time embedding: median-imputed, IQR-scaled, clipped AD-space features."""
        v = X.reindex(columns=self.columns).to_numpy(dtype=float)
        v = np.where(np.isnan(v), self.median, v)
        v = np.clip((v - self.median) / self.iqr, -10, 10)
        return np.nan_to_num(v, nan=0.0).astype(np.float32)

    def score(self, X, booster) -> list[dict]:
        """Consensus, its conformal p-value and the alarm at every rate, per row of X.

        `booster` is the AD-space LightGBM booster the model uses (its logits feed the
        uncertainty scorers), the same one that gives the energy.
        """
        logits = booster.predict(X.reindex(columns=booster.feature_name()), raw_score=True)
        out = self.pipeline.predict(dict(embeddings=self.embed(X), logits=logits))
        return [dict(consensus=float(out["ensemble_score"][i]), p_value=float(out["p_value"][i]),
                     alarms={a: bool(out["flags"][a][i]) for a in self.fars})
                for i in range(len(out["ensemble_score"]))]


def load(directory: str | Path) -> Consensus | None:
    """The consensus shipped in a model directory, or None when absent or LUNA is missing."""
    directory = Path(directory)
    path = directory / FILENAME
    if not path.exists():
        return None
    try:
        import luna  # the pickle holds a luna.LUNAPipeline
    except ImportError:
        return None
    if tuple(int(x) for x in luna.__version__.split(".")[:2]) < (0, 3):
        return None     # p-values and multi-rate flags arrived in luna-ood 0.3
    raw = path.read_bytes()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get(FILENAME) != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"{path}: sha256 does not match {directory / 'manifest.json'}")
    return Consensus(pickle.loads(gzip.decompress(raw)))
