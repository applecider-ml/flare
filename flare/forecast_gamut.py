"""GAMUT class-conditional conformal forecast bands under latent type uncertainty.

Constructs prediction regions as superlevel sets of a dual-weighted posterior
predictive density mixture over all six BTS transient classes:
  * C1 (hypothesis-conditional forecaster): shared LightGBM median and 16%/84%
    quantile boosters conditioned on one-hot class indicators, paired with
    smoothed standardized-residual densities per (class, lead-time bin);
  * C2 (calibrated membership): temperature-scaled probabilities from FLARE's
    phase-matched early classifier;
  * C3 (dual-weighted level-set calibration): multiplicative fixed-point dual
    weights equalising per-class object-level conformal risk-control thresholds
    within merged lead-time cells.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

CLASSES = ("SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV")
FLARE_GRID = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0])
LEAD_EDGES = 0.5 * (FLARE_GRID[1:] + FLARE_GRID[:-1])
EGRID = np.linspace(-60.0, 60.0, 4801)
ESTEP = float(EGRID[1] - EGRID[0])
LOCAL = np.linspace(-3.0, 3.0, 49)
SCALE_FLOOR = 0.05
SUPPORTED_HORIZONS = (5, 10, 15, 20, 30)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled probability vector over the six BTS classes."""
    p = np.asarray(probs, dtype=float)
    logits = np.log(np.maximum(p, 1e-8)) / float(temperature)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum(axis=-1, keepdims=True)


def adaptive_grid(lo_g: np.ndarray, hi_g: np.ndarray, centers: np.ndarray, scales: np.ndarray, n_grid: int = 400):
    """Dual-resolution evaluation grid: uniform global points plus local knots around each class hypothesis."""
    u = lo_g[:, None] + (hi_g - lo_g)[:, None] * np.linspace(0.0, 1.0, n_grid)[None, :]
    loc = (centers[:, :, None] + scales[:, :, None] * LOCAL[None, None, :]).reshape(len(lo_g), -1)
    y = np.sort(np.concatenate([u, loc], axis=1), axis=1)
    gaps = np.diff(y, axis=1)
    w = np.zeros_like(y)
    w[:, 1:] += 0.5 * gaps
    w[:, :-1] += 0.5 * gaps
    return y, w


def extract_components(y_grid: np.ndarray, inside: np.ndarray, offset: float = 0.0):
    """Connected interval components, outer hull [lo, hi], and Lebesgue width per row."""
    n_rows, n_pts = y_grid.shape
    lo = np.empty(n_rows, dtype=float)
    hi = np.empty(n_rows, dtype=float)
    width = np.zeros(n_rows, dtype=float)
    intervals = []
    for i in range(n_rows):
        row_in = inside[i]
        yv = y_grid[i]
        padded = np.r_[False, row_in, False]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2] - 1
        comps = []
        total_w = 0.0
        for s_idx, e_idx in zip(starts, ends, strict=False):
            left = yv[s_idx] - 0.5 * (yv[s_idx] - yv[max(s_idx - 1, 0)])
            right = yv[e_idx] + 0.5 * (yv[min(e_idx + 1, n_pts - 1)] - yv[e_idx])
            total_w += float(right - left)
            comps.append([float(left + offset), float(right + offset)])
        intervals.append(comps)
        lo[i] = comps[0][0]
        hi[i] = comps[-1][1]
        width[i] = total_w
    return lo, hi, width, intervals


def unpack_densities(packed: list, n_classes: int, n_bins: int) -> np.ndarray:
    """Reconstruct (K, n_bins, len(EGRID)) standardized residual densities from sparse JSON segments."""
    dens = np.full((n_classes, n_bins, len(EGRID)), 1e-9, dtype=float)
    for entry in packed:
        c = int(entry["class_index"])
        b = int(entry["lead_bin"])
        floor = float(entry.get("floor", 1e-9))
        dens[c, b, :] = floor
        for i0, vals in entry.get("segments", ()):
            arr_v = np.asarray(vals, dtype=float)
            dens[c, b, int(i0) : int(i0) + len(arr_v)] = arr_v
    return dens


class GamutForecaster:
    """Manifest-verified runtime evaluator for GAMUT class-conditional forecast bands."""

    def __init__(self, directory: str | Path, num_threads: int = 2):
        self.directory = Path(directory)
        self.num_threads = num_threads
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest.get("algorithm") != "sha256":
            raise ValueError("unsupported GAMUT manifest hash")
        self.manifest = manifest["files"]
        self.models = {}
        self.meta_cache = {}

    def checked_path(self, name: str) -> Path:
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("model path escapes GAMUT directory")
        if name not in self.manifest or hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest[name]:
            raise ValueError(f"GAMUT model manifest mismatch: {name}")
        return path

    def metadata(self, horizon: int) -> dict:
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"GAMUT forecast horizon must be one of {SUPPORTED_HORIZONS}; no automatic rounding")
        h = int(horizon)
        if h not in self.meta_cache:
            raw = json.loads(self.checked_path(f"h{h}.json").read_text())
            n_bins = int(raw["n_lead_bins"])
            for mode_cfg in raw["calibration_modes"].values():
                mode_cfg["_dens"] = unpack_densities(mode_cfg["residual_densities"], len(CLASSES), n_bins)
            self.meta_cache[h] = raw
        return self.meta_cache[h]

    def classifier_horizon(self, horizon: int) -> int | None:
        ch = self.metadata(horizon).get("classifier_horizon_days")
        return None if ch is None else int(ch)

    def _booster(self, cfg: dict):
        import lightgbm as lgb

        filename = cfg["file"]
        if filename not in self.models:
            booster = lgb.Booster(model_file=str(self.checked_path(filename)))
            if booster.feature_name() != cfg["features"]:
                raise ValueError(f"GAMUT feature order mismatch: {filename}")
            self.models[filename] = booster
        return self.models[filename]

    def _level_scores(
        self,
        y_grid: np.ndarray,
        hmu: np.ndarray,
        hsig: np.ndarray,
        lead_bins: np.ndarray,
        phat: np.ndarray,
        dens: np.ndarray,
        level_cfg: dict,
    ):
        k_cls = len(CLASSES)
        cell_map = np.asarray(level_cfg["cell_map"], dtype=int)
        weights_v = np.asarray(level_cfg["dual_weights"], dtype=float)
        thresholds = np.asarray(level_cfg["thresholds"], dtype=float)
        cells = cell_map[lead_bins]
        sbar = (hsig * phat[None, :]).sum(axis=1)
        scores = np.zeros_like(y_grid)
        for c in range(k_cls):
            e = (y_grid - hmu[:, c][:, None]) / hsig[:, c][:, None]
            idx = np.clip(np.round((e - EGRID[0]) / ESTEP).astype(int), 0, len(EGRID) - 1)
            fc = dens[c][lead_bins[:, None], idx] / hsig[:, c][:, None]
            fc[(e < EGRID[0]) | (e > EGRID[-1])] = 0.0
            scores += weights_v[cells, c][:, None] * phat[c] * fc
        scores *= sbar[:, None]
        inside = scores >= thresholds[cells][:, None]
        empty = ~inside.any(axis=1)
        if empty.any():
            best_idx = scores[empty].argmax(axis=1)
            inside[np.flatnonzero(empty), best_idx] = True
        return scores, inside

    def predict(
        self,
        events: np.ndarray,
        frame: pd.DataFrame,
        probs: np.ndarray,
        peak: float,
        horizon: int,
        days=None,
        mode: str = "crossfit",
    ) -> dict:
        if mode not in ("crossfit", "split"):
            raise ValueError("gamut_mode must be 'crossfit' or 'split'")
        meta = self.metadata(horizon)
        grid = np.asarray(meta["grid_days"], dtype=float)
        requested = grid if days is None else np.asarray(days, dtype=float)
        if (
            requested.ndim != 1
            or not len(requested)
            or not np.isfinite(requested).all()
            or np.any(requested <= horizon)
            or np.any(requested > 90.0)
            or np.any(np.diff(requested) <= 0)
        ):
            raise ValueError("forecast days must be a nonempty, increasing sequence in (horizon, 90]")

        cal_block = meta["calibration_modes"][mode]
        temperature = float(cal_block["temperature"])
        phat = apply_temperature(np.asarray(probs, dtype=float), temperature)
        prune = float(meta.get("prune_threshold", 1e-3))
        active = phat >= prune
        active[int(phat.argmax())] = True

        x_base = frame.reindex(columns=meta["feature_names"]).astype(float).reset_index(drop=True)
        boosters = {name: self._booster(cfg) for name, cfg in meta["models"].items()}
        dens = cal_block["_dens"]
        k_cls = len(CLASSES)
        lead_bins_req = np.searchsorted(LEAD_EDGES, requested, side="right")

        bands = {}
        for band_id, label in enumerate(("g", "r")):
            in_band = events[events[:, D.COL["band_id"]] == band_id]
            if not len(in_band):
                bands[label] = dict(status="insufficient_data", reason="no detection to anchor this band")
                continue
            last = float(23.9 - 2.5 * in_band[-1, D.COL["logflux"]])
            m_pts = len(requested)
            base_rows = pd.concat(
                [
                    x_base.assign(fc_day=float(d), fc_band=float(band_id), fc_lead=float(d - horizon))
                    for d in requested
                ],
                ignore_index=True,
            )
            hyp_frames = []
            for c in range(k_cls):
                df_c = base_rows.copy()
                for j in range(k_cls):
                    df_c[f"hyp_{j}"] = 1.0 if j == c else 0.0
                hyp_frames.append(df_c)
            stacked = pd.concat(hyp_frames, ignore_index=True)

            preds = {
                name: booster.predict(stacked, num_threads=self.num_threads).reshape(k_cls, m_pts).T
                for name, booster in boosters.items()
            }
            hmu = preds["mu"]
            hsig = np.maximum(0.5 * np.abs(preds["q84"] - preds["q16"]), SCALE_FLOOR)

            lo_g = np.where(active[None, :], hmu - 8.0 * hsig, np.inf).min(axis=1)
            hi_g = np.where(active[None, :], hmu + 8.0 * hsig, -np.inf).max(axis=1)
            y_grid, _ = adaptive_grid(lo_g, hi_g, hmu, hsig)

            scores68, inside68 = self._level_scores(
                y_grid, hmu, hsig, lead_bins_req, phat, dens, cal_block["levels"]["0.68"]
            )
            _, inside95 = self._level_scores(
                y_grid, hmu, hsig, lead_bins_req, phat, dens, cal_block["levels"]["0.95"]
            )
            inside95 = inside95 | inside68

            lo68, hi68, w68, comp68 = extract_components(y_grid, inside68, offset=peak)
            lo95, hi95, w95, comp95 = extract_components(y_grid, inside95, offset=peak)

            post_mean = (hmu * phat[None, :]).sum(axis=1) + peak
            mode_mag = y_grid[np.arange(m_pts), scores68.argmax(axis=1)] + peak
            outside_hull = (post_mean < lo68) | (post_mean > hi68)
            fut_mag = np.where(outside_hull, mode_mag, post_mean)

            bands[label] = dict(
                status="ok",
                days=np.r_[float(horizon), requested],
                magnitude=np.r_[last, fut_mag],
                lower68=np.r_[last, lo68],
                upper68=np.r_[last, hi68],
                lower95=np.r_[last, lo95],
                upper95=np.r_[last, hi95],
                width68=np.r_[0.0, w68],
                width95=np.r_[0.0, w95],
                intervals68=[[[last, last]], *comp68],
                intervals95=[[[last, last]], *comp95],
                multimodal68=[False, *[len(c) > 1 for c in comp68]],
                multimodal95=[False, *[len(c) > 1 for c in comp95]],
            )

        top_cls = CLASSES[int(phat.argmax())]
        return dict(
            status="ok" if any(v["status"] == "ok" for v in bands.values()) else "insufficient_data",
            model_horizon_days=int(horizon),
            observation_cutoff_days=int(horizon),
            recipe="gamut",
            gamut_mode=mode,
            model_role="hypothesis_conditional_gamut",
            classifier_horizon_days=meta.get("classifier_horizon_days"),
            routing_class=top_cls,
            temperature=round(temperature, 6),
            membership_probabilities={c: round(float(p), 6) for c, p in zip(CLASSES, phat, strict=False)},
            band_calibration=f"GAMUT dual-weighted posterior-mixture level sets ({mode})",
            reference_magnitude=float(peak),
            bands=bands,
            units="AB mag",
            time_origin="first detection",
            calibration=cal_block["note"],
            anchor_note="Horizon entry is the last observed band magnitude, shown without a forecast interval.",
            extrapolation="none beyond day 90",
        )
