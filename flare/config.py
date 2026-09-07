"""Scheme selection and model paths — the single switch between taxonomies.

FLARE ships two classification schemes:

    broad5   the original AppleCiDEr taxonomy: SNI | SNII | CV | AGN | TDE,
             one flat LightGBM. Kept for backwards compatibility; the tutorial
             notebooks and the no-TDE teaching model use it.

    bts6     the BTS taxonomy of the paper: a five-way top level
             (SN_Ia | SN_CC+ | AGN | TDE | CV) with a dedicated SLSN head
             inside SN_CC+, reported as six classes. Consumes host-galaxy
             photometry, the photo-z pseudo-absolute magnitude, and Gaia/WISE
             point-source context when available (all optional at inference;
             the models are trained with each external block independently
             blanked, so absence is safe).

The active scheme is decided, in order of precedence:
    1. an explicit `scheme=` argument to `flare.load_classifier(...)`
    2. the FLARE_SCHEME environment variable
    3. the default below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEME = "bts6"

HOST_COLS = ["host_sep", "host_r", "host_i", "host_gr", "host_ri", "host_iz",
             "host_ext", "n_ext_30", "near_sep", "near_ext"]
PZ_COLS = ["M_pseudo"]
GW_COLS = ["gaia_sep", "parallax", "parallax_over_error", "pm", "pm_over_error",
           "gaia_g", "ruwe", "wise_sep", "w1", "w1w2", "w2w3"]
POS_COLS = ["pos_sep", "pos_g", "pos_r", "pos_i", "pos_gr", "pos_ri",
            "pos_star", "pos_ndet"]
# the anomaly layer deliberately consumes a narrower context than the
# classifier: PS1-position and M_pseudo help classification but harm blind
# detection (measured in the paper), so the AD space excludes them
AD_SPACE_COLS = HOST_COLS + GW_COLS


@dataclass(frozen=True)
class SchemeConfig:
    name: str
    classes: List[str]                       # reported classes
    top_classes: List[str]                   # top-level softmax classes
    model_paths: dict                        # role -> Path
    external_cols: List[str] = field(default_factory=list)
    alpha: float = 0.10
    horizon_days: float = 100.0


BROAD5 = SchemeConfig(
    name="broad5",
    classes=["SNI", "SNII", "CV", "AGN", "TDE"],
    top_classes=["SNI", "SNII", "CV", "AGN", "TDE"],
    model_paths={
        "top": PKG_ROOT / "models/flare_lgbm.txt",
        "conformal": PKG_ROOT / "models/conformal.json",
    },
)

BTS6 = SchemeConfig(
    name="bts6",
    classes=["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"],
    top_classes=["SN_Ia", "SN_CC+", "AGN", "TDE", "CV"],
    model_paths={
        "top": PKG_ROOT / "models/bts6/top.txt",
        "slsn_branch": PKG_ROOT / "models/bts6/slsn_branch.txt",
        "conformal": PKG_ROOT / "models/bts6/conformal.json",
        "ad_space": PKG_ROOT / "models/bts6/ad_space.txt",
        "anomaly_calibration": PKG_ROOT / "models/bts6/anomaly_calibration.json",
        "card": PKG_ROOT / "models/bts6/model_card.json",
    },
    external_cols=HOST_COLS + PZ_COLS + GW_COLS + POS_COLS,
)

SCHEMES = {"broad5": BROAD5, "bts6": BTS6}


def resolve_scheme(scheme: Optional[str] = None) -> SchemeConfig:
    name = scheme or os.environ.get("FLARE_SCHEME", DEFAULT_SCHEME)
    if name not in SCHEMES:
        raise ValueError(f"unknown scheme '{name}'; choose from {sorted(SCHEMES)}")
    return SCHEMES[name]
