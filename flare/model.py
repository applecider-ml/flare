"""FLARE — Feature-based Light-curve Aggregated Ranking Ensemble.

Physics light-curve features → gradient-boosted trees, with class-conditional
conformal uncertainty. Pure Python, no deep-learning dependencies.

Usage
-----
    from flare import FlareClassifier
    clf = FlareClassifier.from_pretrained()          # bundled weights
    proba = clf.predict_proba_from_files(["obj.npz"])  # (N, 5)
    label = clf.predict_from_files(["obj.npz"])        # class names
    sets  = clf.prediction_sets_from_files(["obj.npz"], alpha=0.10)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import conformal
from .features import extract_object_features
from .taxonomy import BROAD_CLASSES, NUM_CLASSES

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PKG_ROOT / "models/flare_lgbm.txt"
DEFAULT_CONFORMAL = PKG_ROOT / "models/conformal.json"


class FlareClassifier:
    """LightGBM over physics features + Mondrian conformal calibrator."""

    def __init__(self, booster, qhat: Optional[np.ndarray] = None,
                 alpha: float = 0.10, feature_names: Optional[List[str]] = None):
        self.booster = booster
        self.qhat = qhat
        self.alpha = alpha
        self.feature_names = feature_names or list(booster.feature_name())

    # ---------- construction ----------
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path] = DEFAULT_MODEL,
                        conformal_path: Union[str, Path] = DEFAULT_CONFORMAL):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(model_path))
        qhat, alpha = (None, 0.10)
        if Path(conformal_path).exists():
            qhat, alpha = conformal.load(conformal_path)
        return cls(booster, qhat, alpha)

    # ---------- feature extraction ----------
    def features_from_files(self, filepaths: Sequence[Union[str, Path]],
                            horizon_days: float = 100.0) -> pd.DataFrame:
        rows = [extract_object_features(str(fp), horizon_days=horizon_days)
                for fp in filepaths]
        X = pd.DataFrame(rows)
        # align to the training feature order; missing cols -> NaN (LightGBM ok)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.feature_names]

    # ---------- prediction ----------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X
        return self.booster.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([BROAD_CLASSES[i] for i in self.predict_proba(X).argmax(1)])

    def prediction_sets(self, X: pd.DataFrame,
                        alpha: Optional[float] = None) -> List[List[str]]:
        if self.qhat is None:
            raise RuntimeError("No conformal calibration loaded; "
                               "run scripts/train.py to produce models/conformal.json")
        q = self.qhat
        if alpha is not None and abs(alpha - self.alpha) > 1e-9:
            raise ValueError(f"model calibrated at alpha={self.alpha}; "
                             f"recalibrate for alpha={alpha}")
        proba = self.predict_proba(X)
        member = conformal.prediction_sets(proba, q)
        return [[BROAD_CLASSES[c] for c in np.where(row)[0]] for row in member]

    # ---------- file convenience wrappers ----------
    def predict_proba_from_files(self, filepaths, horizon_days=100.0):
        return self.predict_proba(self.features_from_files(filepaths, horizon_days))

    def predict_from_files(self, filepaths, horizon_days=100.0):
        return self.predict(self.features_from_files(filepaths, horizon_days))

    def prediction_sets_from_files(self, filepaths, alpha=None, horizon_days=100.0):
        return self.prediction_sets(
            self.features_from_files(filepaths, horizon_days), alpha)


__all__ = ["FlareClassifier", "BROAD_CLASSES", "NUM_CLASSES"]
