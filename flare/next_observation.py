"""Calibrated next-observation set-closure probabilities and latent-class mixtures.

Provides two production enhancements to ``FlareAnalysis.next_observation_value``
at horizons 5 and 10 days:
  1. Inductive Venn-Abers recalibration (``flare.venn_abers``) of the shipped
     Gaussian-draw singleton score (``recipe='grid'``), exposing calibrated
     ``p_singleton_calibrated`` and ``p_singleton_interval`` without altering
     existing ``p_singleton`` or ``delta_S`` values.
  2. Latent-class next-point mixture (``recipe='gamut'``): simulates next-night
     magnitudes from a hypothesis-conditional quantile forecaster paired with
     per-class out-of-fold empirical standardized residual quantiles, weighted
     by temperature-scaled phase posteriors and calibrated via Venn-Abers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .config import BTS6

CLASSES = tuple(BTS6.classes)
SUPPORTED_HORIZONS = (5, 10)
SCALE_FLOOR = 0.05
NEXT_OBS_DT = 1.0
NEXT_OBS_ERR = 0.08


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    logits = np.log(np.maximum(p, 1e-8)) / float(temperature)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum(axis=-1, keepdims=True)


def venn_abers_binary(score: float, table: dict) -> tuple[float, float, float]:
    """Evaluate a binary Venn-Abers step table at ``score`` -> (merged_p, p0, p1)."""
    edges = np.asarray(table["edges"], dtype=float)
    k = int(np.searchsorted(edges, float(score), side="right"))
    lo = float(table["p0"][k])
    hi = float(table["p1"][k])
    denom = 1.0 - lo + hi
    merged = hi / denom if denom > 0.0 else 0.0
    return float(merged), lo, hi


class NextObservationPredictor:
    """Manifest-verified runtime evaluator for calibrated next-observation value."""

    def __init__(self, directory: str | Path, num_threads: int = 2):
        self.directory = Path(directory)
        self.num_threads = num_threads
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest.get("algorithm") != "sha256":
            raise ValueError("unsupported next-observation manifest hash")
        self.manifest = manifest["files"]
        self.models = {}
        self.meta_cache = {}

    def checked_path(self, name: str) -> Path:
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("model path escapes next-observation directory")
        if name not in self.manifest or hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest[name]:
            raise ValueError(f"next-observation model manifest mismatch: {name}")
        return path

    def metadata(self, horizon: int) -> dict:
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"next-observation horizon must be one of {SUPPORTED_HORIZONS}")
        h = int(horizon)
        if h not in self.meta_cache:
            self.meta_cache[h] = json.loads(self.checked_path(f"h{h}.json").read_text())
        return self.meta_cache[h]

    def _booster(self, cfg: dict):
        import lightgbm as lgb

        filename = cfg["file"]
        if filename not in self.models:
            booster = lgb.Booster(model_file=str(self.checked_path(filename)))
            if booster.feature_name() != cfg["features"]:
                raise ValueError(f"next-observation feature order mismatch: {filename}")
            self.models[filename] = booster
        return self.models[filename]

    def calibrate_grid_score(self, horizon: int, raw_score: float) -> tuple[float, list[float]]:
        meta = self.metadata(horizon)
        prob, lo, hi = venn_abers_binary(raw_score, meta["grid_venn_abers"])
        return prob, [lo, hi]

    def predict(
        self,
        events: np.ndarray,
        classifier,
        context: dict | None,
        horizon: int,
    ) -> dict:
        from .analysis import append_detection, feature_frame

        meta = self.metadata(horizon)
        arr_clean = D.recompute_colors(events)
        x_pre, peak = feature_frame(arr_clean, context)
        cx = x_pre.reindex(columns=classifier.feature_names).astype(float)
        before = list(classifier.conformal_report(cx)[0]["set"])
        raw_probs = classifier.predict_proba(cx)[0]

        temperature = float(meta["temperature"])
        phat = apply_temperature(raw_probs, temperature)
        p_min = float(meta.get("p_min", 0.01))
        active_mask = phat >= p_min
        active_mask[int(phat.argmax())] = True
        keep = np.flatnonzero(active_mask)
        weights = phat[keep] / phat[keep].sum()

        t_next = float(horizon) + NEXT_OBS_DT
        x_base = x_pre.reindex(columns=meta["feature_names"]).astype(float).reset_index(drop=True)
        boosters = {name: self._booster(cfg) for name, cfg in meta["models"].items()}
        k_cls = len(CLASSES)

        # Query C1 next-point forecasts across both bands and active classes
        band_rows = pd.concat(
            [
                x_base.assign(fc_day=t_next, fc_band=float(b), fc_lead=NEXT_OBS_DT)
                for b in (0, 1)
            ],
            ignore_index=True,
        )
        hyp_frames = []
        for c in keep:
            df_c = band_rows.copy()
            for j in range(k_cls):
                df_c[f"hyp_{j}"] = 1.0 if j == int(c) else 0.0
            hyp_frames.append(df_c)
        stacked = pd.concat(hyp_frames, ignore_index=True)

        preds = {
            name: booster.predict(stacked, num_threads=self.num_threads).reshape(len(keep), 2).T
            for name, booster in boosters.items()
        }
        hmu = preds["mu"]  # (2, len(keep))
        hsig = np.maximum(0.5 * np.abs(preds["q84"] - preds["q16"]), SCALE_FLOOR)
        quantiles_map = {
            int(k): np.asarray(v, dtype=float) for k, v in meta["residual_quantiles"].items()
        }
        n_q = len(next(iter(quantiles_map.values())))

        # Build simulated feature rows for both bands in one batch
        sim_frames = []
        for b_id in (0, 1):
            for idx_c, c in enumerate(keep):
                draws = peak + hmu[b_id, idx_c] + hsig[b_id, idx_c] * quantiles_map[int(c)]
                for m_val in draws:
                    aug = D.recompute_colors(append_detection(arr_clean, t_next, b_id, float(m_val), NEXT_OBS_ERR))
                    fx, _ = feature_frame(aug, context)
                    sim_frames.append(fx)
        sim_df = pd.concat(sim_frames, ignore_index=True).reindex(columns=classifier.feature_names).astype(float)
        sim_reports = classifier.conformal_report(sim_df)
        sim_sizes = np.array([len(r["set"]) for r in sim_reports], dtype=float).reshape(2, len(keep), n_q)

        bands = {}
        for b_id, name in enumerate(("g", "r")):
            mean_sizes_c = sim_sizes[b_id].mean(axis=1)
            pclose_c = (sim_sizes[b_id] == 1.0).mean(axis=1)
            exp_size = float(np.sum(weights * mean_sizes_c))
            delta_s = float(len(before) - exp_size)
            p_raw = float(np.sum(weights * pclose_c))
            p_cal, p_lo, p_hi = venn_abers_binary(p_raw, meta["gamut_venn_abers"])

            means_c = peak + hmu[b_id]
            mag_mean = float(np.sum(weights * means_c))
            mag_var = float(np.sum(weights * (hsig[b_id] ** 2 + (means_c - mag_mean) ** 2)))
            bands[name] = dict(
                status="ok",
                next_day=t_next,
                magnitude_mean=mag_mean,
                magnitude_scatter=float(np.sqrt(max(mag_var, 0.0))),
                expected_set_size=exp_size,
                delta_S=delta_s,
                p_singleton=p_cal,
                p_singleton_raw=p_raw,
                p_singleton_interval=[p_lo, p_hi],
                per_class_p_singleton={
                    CLASSES[int(c)]: round(float(pclose_c[idx_c]), 4) for idx_c, c in enumerate(keep)
                },
            )

        best = max(bands, key=lambda k: bands[k]["delta_S"])
        best_closure = max(bands, key=lambda k: bands[k]["p_singleton"])
        return dict(
            status="ok",
            recipe="gamut",
            observation_cutoff_days=int(horizon),
            set_before=before,
            set_size_before=len(before),
            temperature=round(temperature, 6),
            membership_probabilities={c: round(float(p), 6) for c, p in zip(CLASSES, phat, strict=False)},
            bands=bands,
            best_band=best if bands[best]["delta_S"] > 0 else None,
            best_closure_band=best_closure,
            n_quantiles_per_class=n_q,
            confidence_level=0.90,
            calibration=meta["calibration_note"],
        )
