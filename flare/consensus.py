"""Blind-consensus anomaly score for the shipped models.

The consensus is the mean percentile rank of an object across the LUNA scorers (energy,
Mahalanobis, kNN, density and hybrid scores) in the anomaly space of the model that classifies
it. The percentile ranks are frozen on known-class objects and the false-alarm thresholds are
set on separate known-class objects (scripts/export_consensus.py), so the score of an alert does
not depend on the other alerts scored with it. The p-values are Mondrian in the number of
detections: an alert is compared only with calibration objects that have a similar number of
detections, because with a single reference the alarm fires far more often on young sources with
many detections than on those with few. On the BTS benchmark the consensus detects more of the
out-of-taxonomy objects than the energy alone at every horizon and at 1%, 5% and 10% false-alarm
rates.

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
        m = bundle.get("mondrian")
        self.edges = None if m is None else np.asarray(m["edges"], dtype=float)
        self.ndet_feature = None if m is None else m["feature"]
        self.cal_scores = None if m is None else [np.sort(np.asarray(s, dtype=float)) for s in m["cal_scores"]]

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
        s, p = out["ensemble_score"], out["p_value"].astype(float)
        group = [None] * len(s)
        if self.edges is not None and self.ndet_feature in X:
            n = X[self.ndet_feature].to_numpy(dtype=float)
            for i in np.flatnonzero(np.isfinite(n)):
                k = int(np.searchsorted(self.edges, n[i], side="left"))
                cal = self.cal_scores[k]
                p[i] = (1.0 + len(cal) - np.searchsorted(cal, s[i], side="left")) / (len(cal) + 1.0)
                group[i] = k
        return [dict(consensus=float(s[i]), p_value=float(p[i]),
                     alarms={a: bool(p[i] <= a) for a in self.fars}, ndet_group=group[i])
                for i in range(len(s))]


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
    # The pickles were written with scikit-learn 1.8; later versions warn on every estimator when
    # unpickling, although the consensus and its p-values are identical (checked under 1.8 and 1.9).
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        return Consensus(pickle.loads(gzip.decompress(raw)))
