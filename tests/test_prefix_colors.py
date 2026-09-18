"""Colours must be rebuilt from the observed prefix, never read from a cache that
saw the future. Synthetic light curve, no data files needed.

The failure this guards against: a g-band epoch at day 9.5 whose cached g-r used
the r-band epoch at day 10.5. Truncating the rows at day 10 keeps the epoch and
its colour; the colour then carries information from after the cutoff.
"""
import numpy as np
from flare import data as D

COL = D.COL


def _events(rows):
    """rows: (dt, band_id, mag). Cached colour columns filled from the FULL curve, 1.5 d pairing."""
    a = np.zeros((len(rows), len(COL)), dtype=np.float32)
    for i, (dt, b, mag) in enumerate(rows):
        a[i, 0] = dt; a[i, COL["band_id"]] = b; a[i, COL["logflux"]] = (23.9 - mag) / 2.5; a[i, COL["logflux_err"]] = 0.02
    for i in range(len(a)):                                    # the leaky cache: pair within the whole array
        for name, b1, b2 in [("g_r", 0, 1), ("r_i", 1, 2)]:
            b = a[i, COL["band_id"]]
            if b not in (b1, b2):
                continue
            other = np.flatnonzero(a[:, COL["band_id"]] == (b2 if b == b1 else b1))
            if len(other):
                j = other[np.argmin(np.abs(a[other, 0] - a[i, 0]))]
                if abs(a[j, 0] - a[i, 0]) <= 1.5:
                    lf1 = a[i if b == b1 else j, COL["logflux"]]; lf2 = a[j if b == b1 else i, COL["logflux"]]
                    a[i, COL[name]] = -2.5 * (lf1 - lf2); a[i, COL["has_" + name]] = 1.0
    return a


def test_colour_from_after_the_cutoff_is_discarded():
    full = _events([(0.0, 0, 18.0), (0.5, 1, 18.2), (9.5, 0, 17.6), (10.5, 1, 17.9), (12.0, 0, 17.7)])
    assert full[2, COL["has_g_r"]] == 1.0                     # the cache paired day 9.5 (g) with day 10.5 (r)
    prefix = full[full[:, 0] <= 10.0]
    fixed = D.recompute_colors(prefix)
    assert fixed[2, COL["has_g_r"]] == 0.0 and fixed[2, COL["g_r"]] == 0.0   # no counterpart inside the prefix
    assert fixed[0, COL["has_g_r"]] == 1.0 and abs(fixed[0, COL["g_r"]] - (18.0 - 18.2)) < 1e-4  # a pair inside the prefix survives
    series = D.color_series(prefix)
    assert len(series["g_r"]) == 2 and np.all(series["g_r_t"] <= 10.0)   # both rows of the surviving pair, nothing after day 10


def test_full_curve_unchanged_by_recompute():
    full = _events([(0.0, 0, 18.0), (0.5, 1, 18.2), (9.5, 0, 17.6), (10.5, 1, 17.9), (12.0, 0, 17.7)])
    fixed = D.recompute_colors(full)
    for c in ("g_r", "has_g_r", "r_i", "has_r_i"):
        assert np.allclose(fixed[:, COL[c]], full[:, COL[c]], atol=1e-4)


def test_window_constant_is_the_dataset_default():
    assert D.COLOR_WINDOW_DAYS == 1.5
