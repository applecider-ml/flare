"""flare.context offline path: pre-staged cross-matches -> features, no network.

Pins that context_features(provided=...) computes the point-source blocks from the
caller's nearest-source rows (BOOM-style native columns), carries the pre-computed
aggregated blocks through, and never falls back to the flag value or a query."""
import math

from flare import context as C


def test_provided_blocks_compute_without_network():
    prov = {
        "wise": {"w1mpro": 15.0, "w2mpro": 14.6, "w3mpro": 12.0, "sep": 1.2},
        "gaia": {"parallax": 0.1, "parallax_error": 0.5, "pmra": 1.0, "pmdec": 1.0,
                 "pmra_error": 0.5, "pmdec_error": 0.5, "phot_g_mean_mag": 20.3,
                 "ruwe": 1.1, "sep": 0.3},
        "pos": {"gMeanPSFMag": 21.0, "rMeanPSFMag": 20.5, "iMeanPSFMag": 20.2,
                "iMeanKronMag": 20.1, "nDetections": 12, "sep": 0.8},
        "host": {"near_sep": 18.8, "n_ext_30": 0.0},
        "photoz": {"z_phot": 0.14, "z_phot_std": 0.03},
    }
    out = C.context_features(150.0, 2.5, peak_mag=18.0, provided=prov)
    assert out["wise_sep"] == 1.2 and out["w1"] == 15.0
    assert round(out["w1w2"], 3) == 0.4 and round(out["w2w3"], 3) == 2.6
    assert out["gaia_g"] == 20.3 and out["ruwe"] == 1.1 and round(out["pm_over_error"], 3) == 2.0
    assert round(out["pos_gr"], 3) == 0.5 and out["pos_star"] == 0.0  # PSF-Kron 0.1 > 0.05 -> extended
    assert out["near_sep"] == 18.8 and out["z_phot"] == 0.14
    assert math.isfinite(out["M_pseudo"])


def test_absent_blocks_are_nan_not_fetched():
    # Empty provided dict -> every block NaN, still no network call.
    out = C.context_features(150.0, 2.5, provided={})
    for k in ("w1", "w1w2", "w2w3", "gaia_g", "ruwe", "pos_g", "near_sep", "z_phot"):
        assert math.isnan(out[k])


def test_wise_colours_match_live_compute():
    # The same row through the row= path and a NaN W3 -> colour drops to NaN.
    assert math.isnan(C.wise_features(0, 0, row={"w1mpro": 15.0, "w2mpro": 14.0})["w2w3"])
    w = C.wise_features(0, 0, row={"w1mpro": 15.0, "w2mpro": 14.0, "w3mpro": 11.0, "sep": 2.0})
    assert w["w1w2"] == 1.0 and w["w2w3"] == 3.0 and w["wise_sep"] == 2.0
