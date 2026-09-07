"""The bts6 classifier: five-way top level with an SLSN head inside SN_CC+.

Prediction factorises as
    P(SLSN)  = P(SN_CC+) * P(SLSN | SN_CC+)
    P(SN_CC) = P(SN_CC+) * (1 - P(SLSN | SN_CC+))
with every other class passed through from the top level. The branch decision
uses a threshold tuned on held-out data at training time (stored in the model
card) rather than 0.5, since a 6% positive rate makes the argmax operating
point needlessly conservative; the paper's cost-decoding study showed threshold
tuning and cost-sensitive decoding reach the same operating point.

Inputs are the 160 light-curve features plus, when available, the host block
and the photo-z pseudo-absolute magnitude (see flare.context). All external
columns are optional: the boosters were trained with each block independently
blanked on 30% of rows, so NaN degrades gracefully. Never impute them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import BTS6, SchemeConfig
from .features import extract_object_features

SIX = BTS6.classes
FIVE = BTS6.top_classes
I_CCP = FIVE.index("SN_CC+")
I_SLSN = SIX.index("SLSN")
I_SNCC = SIX.index("SN_CC")


class HierarchicalFlare:
    """Top-level booster + SLSN branch + per-class conformal thresholds."""

    def __init__(self, top, branch, branch_threshold: float = 0.5,
                 qhat: Optional[np.ndarray] = None, alpha: float = 0.10,
                 cal_scores: Optional[dict] = None,
                 coverage: Optional[dict] = None):
        self.top = top
        self.branch = branch
        self.branch_threshold = float(branch_threshold)
        self.qhat = qhat
        self.alpha = alpha
        # sorted calibration nonconformity scores per class; without them the
        # layer can only threshold, not report a p-value
        self.cal_scores = ({k: np.asarray(v, dtype=float)
                            for k, v in cal_scores.items()}
                           if cal_scores else None)
        # out-of-sample empirical coverage per class (five-fold, from the paper)
        self.coverage = coverage
        self.feature_names = list(top.feature_name())

    # ---------- construction ----------
    @classmethod
    def from_pretrained(cls, cfg: SchemeConfig = BTS6) -> "HierarchicalFlare":
        import lightgbm as lgb
        top = lgb.Booster(model_file=str(cfg.model_paths["top"]))
        branch = lgb.Booster(model_file=str(cfg.model_paths["slsn_branch"]))
        thr, qhat, alpha = 0.5, None, cfg.alpha
        cal_scores = coverage = None
        card_path = cfg.model_paths.get("card")
        if card_path and Path(card_path).exists():
            card = json.loads(Path(card_path).read_text())
            thr = card.get("branch_threshold", 0.5)
        conf_path = cfg.model_paths.get("conformal")
        if conf_path and Path(conf_path).exists():
            d = json.loads(Path(conf_path).read_text())
            qhat = np.array([d["qhat"][c] for c in SIX])
            alpha = d["alpha"]
            cal_scores = d.get("cal_scores")
            coverage = d.get("empirical_coverage")
        inst = cls(top, branch, thr, qhat, alpha, cal_scores, coverage)
        ad_path = cfg.model_paths.get("ad_space")
        if ad_path and Path(ad_path).exists():
            inst.ad_space = lgb.Booster(model_file=str(ad_path))
        cal_path = cfg.model_paths.get("anomaly_calibration")
        if cal_path and Path(cal_path).exists():
            inst.anomaly_cal = json.loads(Path(cal_path).read_text())
        return inst

    # ---------- features ----------
    def features_from_files(self, filepaths: Sequence[Union[str, Path]],
                            context: Optional[pd.DataFrame] = None,
                            horizon_days: float = 100.0) -> pd.DataFrame:
        rows = [extract_object_features(str(fp), horizon_days=horizon_days)
                for fp in filepaths]
        X = pd.DataFrame(rows)
        if context is not None:
            X = pd.concat([X.reset_index(drop=True),
                           context.reset_index(drop=True)], axis=1)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.feature_names]

    # ---------- prediction ----------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """(N, 6) factorised probabilities over the reported classes."""
        X = X[self.feature_names]
        P5 = self.top.predict(X)
        pb = self.branch.predict(X)
        P6 = np.zeros((len(P5), 6))
        for j, c in enumerate(SIX):
            if c == "SN_CC":
                P6[:, j] = P5[:, I_CCP] * (1 - pb)
            elif c == "SLSN":
                P6[:, j] = P5[:, I_CCP] * pb
            else:
                P6[:, j] = P5[:, FIVE.index(c)]
        return P6

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Class names, using the tuned branch threshold inside SN_CC+."""
        X = X[self.feature_names]
        P5 = self.top.predict(X)
        pb = self.branch.predict(X)
        top = P5.argmax(1)
        pred = np.empty(len(top), int)
        for j, c in enumerate(FIVE):
            if c != "SN_CC+":
                pred[top == j] = SIX.index(c)
        in_cc = top == I_CCP
        pred[in_cc & (pb > self.branch_threshold)] = I_SLSN
        pred[in_cc & (pb <= self.branch_threshold)] = I_SNCC
        return np.array([SIX[i] for i in pred])

    def p_values(self, X: pd.DataFrame) -> np.ndarray:
        """(N, 6) class-conditional conformal p-values.

        For class c the nonconformity is s = 1 - P(c) and

            p_c(x) = (1 + #{s_i in cal_c : s_i >= s(x)}) / (n_c + 1),

        the standard Mondrian construction. p_c is the probability, under
        exchangeability with that class's calibration sample, of seeing an
        object of class c at least as nonconforming as this one. The floor
        1/(n_c + 1) is set by the calibration size: with fifteen SLSNe no
        p-value below 0.06 is expressible, which is why the sets, not the
        p-values, carry the guarantee for the rare classes.
        """
        if self.cal_scores is None:
            raise RuntimeError("no calibration scores bundled; retrain with "
                               "scripts/train_bts6.py to export them")
        S = 1.0 - self.predict_proba(X)
        P = np.empty_like(S)
        for j, c in enumerate(SIX):
            cal = self.cal_scores[c]
            n = len(cal)
            ge = n - np.searchsorted(cal, S[:, j], side="left")
            P[:, j] = (1.0 + ge) / (n + 1.0)
        return P

    def prediction_sets(self, X: pd.DataFrame,
                        alpha: Optional[float] = None) -> List[List[str]]:
        """Mondrian conformal sets over the six reported classes.

        At the bundled alpha this thresholds the stored per-class quantiles.
        Passing another alpha re-derives the sets from the p-values, which
        needs the calibration scores to have been exported.
        """
        if alpha is None or (self.cal_scores is None):
            if self.qhat is None:
                raise RuntimeError("no conformal calibration bundled; "
                                   "run scripts/train_bts6.py")
            if alpha is not None:
                raise RuntimeError("changing alpha needs the calibration "
                                   "scores; retrain to export them")
            P6 = self.predict_proba(X)
            member = (1.0 - P6) <= self.qhat[None, :]
        else:
            member = self.p_values(X) > float(alpha)
        return [[SIX[c] for c in np.where(row)[0]] for row in member]

    def anomaly_probability(self, X: pd.DataFrame,
                            base_rate: Optional[float] = None) -> List[dict]:
        """Turn the anomaly energy into a probability, and say what it assumes.

        A score is not a probability until it is told how common the thing it
        looks for actually is. The shipped calibration is a class-balanced
        logistic fit of the energy on the benchmark's own two populations
        (1575 known-class test objects, 52 out-of-taxonomy rarities), so

            likelihood_ratio = exp(a * E + b)

        carries no prior at all, and

            P(anomaly) = LR * pi / (1 - pi + LR * pi)

        for whatever base rate pi the operator's stream has. The default is
        the benchmark's own 3.2%, which is far richer in rarities than a raw
        alert stream: quote the likelihood ratio when the prior is unknown.
        Also returned is a distribution-free novelty p-value, the fraction of
        known-class objects at least this extreme, which needs no prior.
        """
        if getattr(self, "anomaly_cal", None) is None:
            raise RuntimeError("no anomaly calibration bundled; retrain with "
                               "scripts/train_bts6.py")
        cal = self.anomaly_cal
        pi = float(cal["base_rate"] if base_rate is None else base_rate)
        E = self.anomaly_energy(X)
        lr = np.exp(cal["coef"] * E + cal["intercept"])
        odds = lr * pi / (1.0 - pi)
        p = odds / (1.0 + odds)
        known = np.asarray(cal.get("known_energies", []), dtype=float)
        out = []
        for i in range(len(E)):
            rec = dict(energy=float(E[i]), p_anomaly=float(p[i]),
                       likelihood_ratio=float(lr[i]), base_rate=pi)
            if known.size:
                ge = int((known >= E[i]).sum())
                rec["novelty_p"] = float((1 + ge) / (known.size + 1))
                rec["energy_percentile"] = float(100 * (1 - ge / known.size))
            out.append(rec)
        return out

    def conformal_report(self, X: pd.DataFrame,
                         alpha: Optional[float] = None) -> List[dict]:
        """Per-object conformal output *with its error statement*.

        A prediction set means nothing without the error it admits, so every
        entry carries:

            alpha                the miscoverage the set is calibrated for
            guaranteed_error     the error the construction admits (= alpha)
            empirical_coverage   measured out of fold for the predicted class
                                 (five-fold; None if not bundled)
            set / set_size       the set itself
            p_values             per-class conformal p-values
            credibility          max_c p_c -- how well the object matches any
                                 class at all; low means it conforms to
                                 nothing, the conformal echo of the anomaly score
            confidence           1 - second-largest p_c -- how firmly the
                                 runner-up is excluded
            p_value_floor        1/(n_c + 1) for the winning class
        """
        a = float(self.alpha if alpha is None else alpha)
        pred = self.predict(X)
        sets = self.prediction_sets(X, None if alpha is None else a)
        out = []
        have_p = self.cal_scores is not None
        P = self.p_values(X) if have_p else None
        for i, (p_, s_) in enumerate(zip(pred, sets)):
            rec = dict(predicted=str(p_), set=list(s_), set_size=len(s_),
                       alpha=a, guaranteed_error=a,
                       empirical_coverage=(self.coverage or {}).get(str(p_)))
            if have_p:
                row = P[i]
                order = np.argsort(row)[::-1]
                rec.update(
                    p_values={c: float(round(v, 4)) for c, v in zip(SIX, row)},
                    credibility=float(row[order[0]]),
                    confidence=float(1.0 - row[order[1]]),
                    p_value_floor=float(
                        1.0 / (len(self.cal_scores[SIX[order[0]]]) + 1)))
            out.append(rec)
        return out

    def anomaly_energy(self, X: pd.DataFrame) -> np.ndarray:
        """The classifier-native anomaly score: -logsumexp of raw top scores.

        The paper's scorer study found this the single usable blind OOD score
        (uncertainty family; distance scorers require a geometry a CE-trained
        space does not keep). Higher = more anomalous.

        Scored in the anomaly layer's own space (features + host + Gaia/WISE;
        the M_pseudo and counterpart columns help classification but harm
        blind detection) when the ad_space booster is shipped; falls back to
        the classifier's logits otherwise.
        """
        from scipy.special import logsumexp
        m = getattr(self, "ad_space", None) or self.top
        raw = m.predict(X.reindex(columns=m.feature_name()), raw_score=True)
        return -logsumexp(raw, axis=1)

    # ---------- file convenience ----------
    def predict_from_files(self, filepaths, context=None, horizon_days=100.0):
        return self.predict(self.features_from_files(filepaths, context,
                                                     horizon_days))

    def predict_proba_from_files(self, filepaths, context=None,
                                 horizon_days=100.0):
        return self.predict_proba(self.features_from_files(filepaths, context,
                                                           horizon_days))

    def prediction_sets_from_files(self, filepaths, context=None,
                                   horizon_days=100.0):
        return self.prediction_sets(self.features_from_files(filepaths, context,
                                                             horizon_days))


__all__ = ["HierarchicalFlare", "SIX", "FIVE"]
