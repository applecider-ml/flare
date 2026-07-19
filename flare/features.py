"""Physics-informed feature extraction for one object (Stream A).

Per band (g, r, i):
  - statistical / variability features from the `light-curve` package,
    computed on logflux (mag-like space) with logflux_err as sigma
  - Bazin transient fit in linear flux space (rise/fall timescales,
    amplitude, baseline, reduced chi2)
  - cadence features (n_obs, span, gaps)
Cross-band:
  - g−r and r−i color statistics from the precomputed per-event colors
  - color evolution slope (TDE ~ constant blue color; SNe redden)
  - peak-time difference and peak-flux ratio between g and r

Missing bands / failed fits → NaN (LightGBM handles natively).
"""
from __future__ import annotations

from typing import Dict

import light_curve as licu
import numpy as np

from . import data as D

# --- light-curve extractor (shared, stateless → safe across processes) ---
_STAT_FEATURES = [
    licu.Amplitude(),
    licu.AndersonDarlingNormal(),
    licu.BeyondNStd(1.0),
    licu.BeyondNStd(2.0),
    licu.Cusum(),
    licu.EtaE(),
    licu.ExcessVariance(),
    licu.InterPercentileRange(0.10),
    licu.InterPercentileRange(0.25),
    licu.Kurtosis(),
    licu.LinearFit(),
    licu.LinearTrend(),
    licu.MagnitudePercentageRatio(0.4, 0.05),
    licu.MagnitudePercentageRatio(0.2, 0.10),
    licu.MaximumSlope(),
    licu.Mean(),
    licu.MeanVariance(),
    licu.MedianAbsoluteDeviation(),
    licu.MedianBufferRangePercentage(0.10),
    licu.PercentAmplitude(),
    licu.PercentDifferenceMagnitudePercentile(0.05),
    licu.ReducedChi2(),
    licu.Skew(),
    licu.StandardDeviation(),
    licu.StetsonK(),
    licu.WeightedMean(),
]
_STAT_EXTRACTOR = licu.Extractor(*_STAT_FEATURES)
_STAT_NAMES = list(_STAT_EXTRACTOR.names)

_BAZIN = licu.BazinFit(algorithm="ceres")
# raw names carry a 'bazin_fit_' prefix — strip it, we add our own
_BAZIN_NAMES = [n.replace("bazin_fit_", "") for n in _BAZIN.names]
# amplitude, baseline, reference_time, rise_time, fall_time, reduced_chi2

_MIN_STAT_POINTS = 4
_MIN_BAZIN_POINTS = 6


def _band_features(lc: D.BandLC, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(lc.t)
    # cadence
    out[f"{prefix}_n_obs"] = float(n)
    out[f"{prefix}_span"] = float(lc.t[-1] - lc.t[0]) if n > 1 else 0.0
    if n > 1:
        gaps = np.diff(lc.t)
        out[f"{prefix}_gap_median"] = float(np.median(gaps))
        out[f"{prefix}_gap_max"] = float(gaps.max())
    else:
        out[f"{prefix}_gap_median"] = np.nan
        out[f"{prefix}_gap_max"] = np.nan
    # peak info in flux space
    ipk = int(np.argmax(lc.flux))
    out[f"{prefix}_t_peak"] = float(lc.t[ipk])
    out[f"{prefix}_logflux_peak"] = float(lc.logflux[ipk])
    out[f"{prefix}_logflux_last_minus_peak"] = float(lc.logflux[-1] - lc.logflux[ipk])
    # rise/fade fractions relative to peak
    out[f"{prefix}_frac_obs_before_peak"] = float((lc.t < lc.t[ipk]).mean())
    # mean SNR
    out[f"{prefix}_snr_mean"] = float(np.mean(D.LOG_CONST / np.clip(lc.logflux_err, 1e-6, None)))

    # statistical features on logflux (mag-like)
    if n >= _MIN_STAT_POINTS:
        vals = _STAT_EXTRACTOR(lc.t, lc.logflux, lc.logflux_err,
                               fill_value=np.nan, check=False)
        for name, v in zip(_STAT_NAMES, vals):
            out[f"{prefix}_{name}"] = float(v)
    else:
        for name in _STAT_NAMES:
            out[f"{prefix}_{name}"] = np.nan

    # Bazin fit in flux space
    if n >= _MIN_BAZIN_POINTS:
        try:
            bv = _BAZIN(lc.t, lc.flux, lc.flux_err, fill_value=np.nan, check=False)
            for name, v in zip(_BAZIN_NAMES, bv):
                out[f"{prefix}_bazin_{name}"] = float(v)
            # scale-free derived quantities
            rise, fall = out[f"{prefix}_bazin_rise_time"], out[f"{prefix}_bazin_fall_time"]
            out[f"{prefix}_bazin_fall_over_rise"] = (
                fall / rise if np.isfinite(rise) and np.isfinite(fall) and rise > 0 else np.nan)
        except Exception:
            for name in _BAZIN_NAMES:
                out[f"{prefix}_bazin_{name}"] = np.nan
            out[f"{prefix}_bazin_fall_over_rise"] = np.nan
    else:
        for name in _BAZIN_NAMES:
            out[f"{prefix}_bazin_{name}"] = np.nan
        out[f"{prefix}_bazin_fall_over_rise"] = np.nan
    return out


def _color_features(colors: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in ["g_r", "r_i"]:
        t, v, e = colors[c + "_t"], colors[c], colors[c + "_err"]
        n = len(v)
        out[f"{c}_n"] = float(n)
        if n == 0:
            for suf in ["mean", "wmean", "std", "slope", "early", "late", "late_minus_early"]:
                out[f"{c}_{suf}"] = np.nan
            continue
        w = 1.0 / np.clip(e, 1e-6, None) ** 2
        out[f"{c}_mean"] = float(v.mean())
        out[f"{c}_wmean"] = float(np.average(v, weights=w))
        out[f"{c}_std"] = float(v.std()) if n > 1 else np.nan
        if n > 2 and np.ptp(t) > 0:
            out[f"{c}_slope"] = float(np.polyfit(t, v, 1)[0])
        else:
            out[f"{c}_slope"] = np.nan
        half = t.min() + np.ptp(t) / 2 if n > 1 else t[0]
        early, late = v[t <= half], v[t > half]
        out[f"{c}_early"] = float(early.mean()) if len(early) else np.nan
        out[f"{c}_late"] = float(late.mean()) if len(late) else np.nan
        out[f"{c}_late_minus_early"] = (
            out[f"{c}_late"] - out[f"{c}_early"]
            if np.isfinite(out[f"{c}_late"]) and np.isfinite(out[f"{c}_early"]) else np.nan)
    return out


def extract_object_features(filepath: str,
                            horizon_days: float = D.HORIZON_DAYS) -> Dict[str, float]:
    arr = D.load_events(filepath, horizon_days)
    bands = D.reconstruct_bands(arr)
    out: Dict[str, float] = {}

    # global cadence
    out["n_obs_total"] = float(len(arr))
    out["n_bands"] = float(len(bands))
    out["span_total"] = float(arr[:, D.COL["dt"]].max()) if len(arr) else np.nan

    for name in D.BAND_NAMES:
        if name in bands:
            out.update(_band_features(bands[name], name))
        else:
            # emit full NaN block so every row has identical columns
            dummy_keys = _band_features(
                D.BandLC(t=np.array([0.0, 1.0, 2.0, 3.0]),
                         flux=np.ones(4), flux_err=np.ones(4) * 0.1,
                         logflux=np.zeros(4), logflux_err=np.ones(4) * 0.04),
                name).keys()
            out.update({k: np.nan for k in dummy_keys})

    out.update(_color_features(D.color_series(arr)))

    # cross-band peak relations
    if "g" in bands and "r" in bands:
        tg, tr = out["g_t_peak"], out["r_t_peak"]
        out["gr_peak_dt"] = tg - tr
        out["gr_peak_logflux_ratio"] = out["g_logflux_peak"] - out["r_logflux_peak"]
        out["gr_n_ratio"] = out["g_n_obs"] / max(out["r_n_obs"], 1.0)
    else:
        out["gr_peak_dt"] = np.nan
        out["gr_peak_logflux_ratio"] = np.nan
        out["gr_n_ratio"] = np.nan
    return out
