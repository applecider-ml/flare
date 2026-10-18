"""Rebuild results/reconstruction/cache/X_pool_h{h}.parquet with prefix colours.

Pool order = train, val, test splits (the order forecast_v2_h{h}.npz uses). Each row is
flare.analysis.feature_frame(observed_events(load_events(path), h), context), i.e. the same
recipe evaluate_unified.py applies to the novel objects; colours come from flare.data.load_events,
which recomputes them on the observed prefix (the 2026-09-17 colour-leak fix).
Run: python scripts/build_horizon_cache.py [--threads 14] [--horizons 5 10 20]
"""
import argparse
from multiprocessing import Pool
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from flare.analysis import feature_frame, observed_events  # noqa: E402
from flare.data import load_events  # noqa: E402
from evaluate_unified import context_table  # noqa: E402

ROOT = Path(os.environ.get("FLARE_PAPER_DATA", PKG / "paper_data"))  # python scripts/setup_data.py
_CTX = None


def _row(args):
    path, oid, h = args
    arr = load_events(str(ROOT / "data" / path))
    f, _peak = feature_frame(observed_events(arr, h), _CTX.loc[oid].to_dict())
    return f


def main():
    global _CTX
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    a = p.parse_args()
    df = pd.concat([pd.read_csv(ROOT / f"data/splits/{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    _CTX = context_table(ROOT, df.obj_id.values)
    out = ROOT / "results/reconstruction/cache"
    for h in a.horizons:
        z = np.load(ROOT / f"results/reconstruction/forecast_v2_h{h}.npz", allow_pickle=True)
        assert np.array_equal(z["obj_id"], df.obj_id), "pool order differs from forecast_v2"
        with Pool(a.threads) as pool:
            rows = pool.map(_row, [(r.filepath, r.obj_id, float(h)) for r in df.itertuples()], chunksize=64)
        X = pd.concat(rows, ignore_index=True)
        X.to_parquet(out / f"X_pool_h{h}.parquet", index=False)
        print(f"h={h}: {X.shape}, M_pseudo finite {np.isfinite(X.M_pseudo).mean():.2f}, "
              f"n_obs_total median {X.n_obs_total.median():.0f}", flush=True)


if __name__ == "__main__":
    main()
