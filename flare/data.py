"""Data loading, quality filtering, and light-curve reconstruction.

The quality filter keeps objects with enough well-sampled photometry inside the
observation horizon (defaults below); light curves are reconstructed per band in
linear flux from the stored log-flux events.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import os

DATA_DIR = Path(os.environ.get(
    "FLARE_DATA",
    "/work/hdd/bcrv/ffontinelenunes/data/AppleCider/photo_events"))
HORIZON_DAYS = 100.0
# Pairing window for per-event g−r / r−i colours. The BTS event files were built
# with 1.5 d; FLARE_COLOR_WINDOW overrides it for sensitivity studies.
COLOR_WINDOW_DAYS = float(os.environ.get("FLARE_COLOR_WINDOW", "1.5"))

# Quality-filter defaults
MIN_OBS_TOTAL = 8
MIN_OBS_G = 2
MIN_OBS_R = 2
MIN_OBS_I = 0
MIN_BANDS_OBSERVED = 2

LOG_CONST = 1.0 / np.log(10)  # 0.43429; logflux_err = flux_err/flux * LOG_CONST

# npz `data` column indices (see `columns` array in any object file)
COL = {name: i for i, name in enumerate([
    "dt", "dt_prev", "band_id", "logflux", "logflux_err",
    "band_ztfg", "band_ztfr", "band_ztfi",
    "g_r", "g_r_err", "r_i", "r_i_err", "has_g_r", "has_r_i", "label",
])}

BAND_NAMES = ["g", "r", "i"]


def load_manifest(split: str, data_dir: Path = None) -> pd.DataFrame:
    # read the module global at call time so scripts can set flare.data.DATA_DIR
    dd = Path(data_dir) if data_dir is not None else DATA_DIR
    df = pd.read_csv(dd / f"manifest_{split}.csv")
    df["filepath"] = df["filepath"].map(lambda p: str(dd / p))
    return df


def load_events(filepath: str, horizon_days: float = HORIZON_DAYS) -> np.ndarray:
    """Return the (n_events, 15) array truncated to the horizon window."""
    raw = np.load(filepath, allow_pickle=False)
    arr = raw["data"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
    return arr[arr[:, 0] <= horizon_days]


def passes_quality(arr: np.ndarray) -> bool:
    """Quality cut on a horizon-truncated event array."""
    if len(arr) == 0:
        return False
    band = arr[:, COL["band_id"]].astype(np.int64)
    counts = np.array([(band == b).sum() for b in [0, 1, 2]], dtype=int)
    if arr.shape[0] < MIN_OBS_TOTAL:
        return False
    if counts[0] < MIN_OBS_G or counts[1] < MIN_OBS_R or counts[2] < MIN_OBS_I:
        return False
    if int((counts > 0).sum()) < MIN_BANDS_OBSERVED:
        return False
    return True


def filter_manifest_quality(df: pd.DataFrame,
                            horizon_days: float = HORIZON_DAYS) -> Tuple[pd.DataFrame, dict]:
    keep = np.zeros(len(df), dtype=bool)
    for i, row in df.iterrows():
        try:
            keep[i] = passes_quality(load_events(row.filepath, horizon_days))
        except Exception:
            continue
    out = df.loc[keep].reset_index(drop=True)
    return out, {
        "rows_before": int(len(df)),
        "rows_after": int(len(out)),
        "rows_dropped": int(len(df) - len(out)),
    }


@dataclass
class BandLC:
    """Single-band light curve in linear flux space (μJy)."""
    t: np.ndarray       # days since first event of the object
    flux: np.ndarray
    flux_err: np.ndarray
    logflux: np.ndarray
    logflux_err: np.ndarray


def reconstruct_bands(arr: np.ndarray) -> Dict[str, BandLC]:
    """Split the event array into per-band light curves.

    logflux = log10(flux_μJy)  →  flux = 10**logflux
    logflux_err = flux_err/flux * LOG_CONST  →  flux_err = logflux_err*flux/LOG_CONST
    Times are deduplicated (light-curve extractors need strictly increasing t).
    """
    out: Dict[str, BandLC] = {}
    band = arr[:, COL["band_id"]].astype(np.int64)
    for b, name in enumerate(BAND_NAMES):
        m = band == b
        if m.sum() == 0:
            continue
        t = arr[m, COL["dt"]].astype(np.float64)
        lf = arr[m, COL["logflux"]].astype(np.float64)
        lfe = arr[m, COL["logflux_err"]].astype(np.float64)
        order = np.argsort(t, kind="stable")
        t, lf, lfe = t[order], lf[order], lfe[order]
        # strictly increasing time (merge exact duplicates, keep first)
        uniq = np.concatenate([[True], np.diff(t) > 0])
        t, lf, lfe = t[uniq], lf[uniq], lfe[uniq]
        flux = 10.0 ** lf
        flux_err = lfe * flux / LOG_CONST
        out[name] = BandLC(t=t, flux=flux, flux_err=flux_err,
                           logflux=lf, logflux_err=lfe)
    return out


def recompute_colors(events):
    """Rebuild color channels from this observed prefix, never cached full curves.

    A cached color attached to an early detection can have used a counterpart
    after the cutoff. Clearing and recomputing these six columns is essential
    even after the event rows themselves have been truncated.
    """
    a = np.asarray(events, dtype=float).copy()
    for name, first, second in [('g_r', 0, 1), ('r_i', 1, 2)]:
        a[:, [COL[name], COL[name + '_err'], COL['has_' + name]]] = 0.
        for i in range(len(a)):
            band = a[i, COL['band_id']]
            if band not in (first, second):
                continue
            other = np.flatnonzero(a[:, COL['band_id']] == (second if band == first else first))
            if not len(other):
                continue
            j = other[np.argmin(np.abs(a[other, 0] - a[i, 0]))]
            if abs(a[j, 0] - a[i, 0]) > COLOR_WINDOW_DAYS:
                continue
            lf_first = a[i if band == first else j, COL['logflux']]
            lf_second = a[j if band == first else i, COL['logflux']]
            a[i, COL[name]] = -2.5 * (lf_first - lf_second)
            a[i, COL[name + '_err']] = 2.5 * np.hypot(a[i, COL['logflux_err']], a[j, COL['logflux_err']])
            a[i, COL['has_' + name]] = 1.
    return a


def color_series(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """g−r / r−i series rebuilt from available rows, ignoring cached colors."""
    arr = recompute_colors(arr)
    out = {}
    for c, flag in [("g_r", "has_g_r"), ("r_i", "has_r_i")]:
        m = arr[:, COL[flag]] > 0.5
        out[c + "_t"] = arr[m, COL["dt"]].astype(np.float64)
        out[c] = arr[m, COL[c]].astype(np.float64)
        out[c + "_err"] = arr[m, COL[c + "_err"]].astype(np.float64)
    return out


def broad_labels(df: pd.DataFrame) -> np.ndarray:
    from .taxonomy import ID2BROAD_ID
    return df["label"].map(lambda x: ID2BROAD_ID[int(x)]).values
