"""The BTS benchmark shipped in `benchmark/`, ready for `flare.data`.

`prepare()` unpacks the light curves (`benchmark/light_curves.tar.gz` -> `benchmark/events/`)
and writes `manifest_{train,val,test}.csv` in the format of `flare.data.load_manifest`
(`filepath`, integer `label` of the legacy five-class taxonomy). Objects outside that
taxonomy (hydrogen-poor superluminous supernovae) are left out of these manifests; the six-class
labels of the paper stay in `benchmark/splits/*.csv` (column `cls`).
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import pandas as pd

from .taxonomy import SUBCLASS_ID2NAME

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark"
# one representative subclass per broad class, so that taxonomy.ID2BROAD_ID maps it back
_BROAD_TO_SUBCLASS = {"SNI": "SN Ia", "SNII": "SN II", "CV": "Cataclysmic", "AGN": "AGN",
                      "TDE": "Tidal Disruption Event"}


def prepare(root: str | Path | None = None) -> Path:
    """Unpack the light curves and write the five-class manifests; returns the data directory."""
    root = Path(root) if root is not None else BENCHMARK
    if not (root / "events").exists():
        with tarfile.open(root / "light_curves.tar.gz") as tar:
            if sys.version_info >= (3, 12):
                tar.extractall(root, filter="data")
            else:
                tar.extractall(root)
    for split in ("train", "val", "test"):
        out = root / f"manifest_{split}.csv"
        if out.exists():
            continue
        df = pd.read_csv(root / "splits" / f"{split}.csv")
        df = df[df.broad.isin(_BROAD_TO_SUBCLASS)].copy()
        df["label"] = df.broad.map(lambda b: SUBCLASS_ID2NAME.index(_BROAD_TO_SUBCLASS[b]))
        df[["obj_id", "filepath", "label", "cls"]].to_csv(out, index=False)
    return root


SIX = ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"]


def load_split(split: str, root: str | Path | None = None):
    """Features of one split of the benchmark, ready for `flare.load_classifier()`.

    Returns (X, y, table): X has the 160 light-curve features plus the context columns
    (Pan-STARRS host, M_pseudo from the Legacy Surveys photo-z, Gaia/WISE, Pan-STARRS
    counterpart) in the order of the classifier; y holds the six-class labels as indices of
    `SIX`; table is the split file. Pass `X.reindex(columns=clf.feature_names)` to the classifier. The shipped classifier is trained on train+val, so only the
    test split gives an honest evaluation of it.
    """
    import numpy as np
    from .config import BTS6
    from .context import lumdist_mpc

    root = Path(root) if root is not None else BENCHMARK
    if split == "anomaly":
        table = pd.read_csv(root / "splits" / "anomaly.csv")
    else:
        table = pd.read_csv(root / "splits" / f"{split}.csv")
    ids = table.obj_id
    feats = pd.read_parquet(root / "features_all.parquet").set_index("obj_id").reindex(ids)
    X = feats.reset_index(drop=True)
    for name in ("hosts.csv", "context_gaia_wise.csv", "ps1_position.csv"):
        t = pd.read_csv(root / name).drop_duplicates("obj_id").set_index("obj_id").reindex(ids).reset_index(drop=True)
        for c in t.columns:
            if c in BTS6.external_cols and c not in X:
                X[c] = t[c].to_numpy()
    pz = pd.read_csv(root / "hosts_photoz.csv").drop_duplicates("obj_id").set_index("obj_id").reindex(ids)
    peak = 23.9 - 2.5 * feats[["g_logflux_peak", "r_logflux_peak", "i_logflux_peak"]].max(axis=1).to_numpy()
    z = pz["z_phot"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        X["M_pseudo"] = peak - 5 * np.log10(lumdist_mpc(z) * 1e5)
    y = np.array([SIX.index(c) if c in SIX else -1 for c in table.get("cls", pd.Series([None] * len(table)))])
    return X, y, table
