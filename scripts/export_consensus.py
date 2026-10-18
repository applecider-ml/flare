"""Blind-consensus anomaly layer for the shipped models (full curve and day 5/10/20).

For every model directory, the LUNA scorers (luna-ood >= 0.3) are fitted on the training split
in that model's anomaly space: the embedding is the AD-space feature vector, median-imputed and
scaled by the training interquartile range, and the logits are the raw scores of the shipped
`ad_space.txt` booster that the model uses (its own at day 5, the full-curve one at day 10 and
20, on light curves truncated at that horizon). The consensus is the mean percentile rank across the scorers, with the
ranks frozen on one half of the known-class objects that calibrate the shipped energy alarm and
the thresholds set on the other half (test split for the full-curve model, validation split for
the day-5/10/20 models; neither was used to train the corresponding booster). Before anything is
written, the energies recomputed here must match the bundled `known_energies`.

Writes `consensus.pkl.gz` into each model directory and adds its sha256 to the manifest.
Run: FLARE_PAPER_DATA=<dir> python scripts/export_consensus.py --models models/bts6
"""
import argparse, gzip, hashlib, json, os, pickle, sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logsumexp

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
from evaluate_unified import context_table, SIX  # noqa: E402
from luna import LUNAPipeline, __version__ as luna_version  # noqa: E402
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", PKG / "paper_data"))  # python paper/setup_data.py
FARS = (0.01, 0.05, 0.10)


def design(ids, h, pool_ids):
    """AD-space inputs for `ids` at horizon h (None = full curve)."""
    if h is None:
        lc = pd.read_parquet(ROOT / "data/features_all.parquet").set_index("obj_id").reindex(ids)
        return pd.concat([lc, context_table(ROOT, ids).reindex(ids)], axis=1)
    X = pd.read_parquet(ROOT / f"results/reconstruction/cache/X_pool_h{h}.parquet").iloc[:len(pool_ids)]
    X.index = pool_ids
    return X.reindex(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--horizons", nargs="+", default=["full", "5", "10", "20"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    split = {s: pd.read_csv(ROOT / f"data/splits/{s}.csv") for s in ("train", "val", "test")}
    pool_ids = pd.concat([split["train"], split["val"]]).obj_id.values
    for hs in a.horizons:
        h = None if hs == "full" else int(hs)
        out = a.models if h is None else a.models / "early" / f"h{h}"
        # the phase's own AD booster where one ships (day 5), else the full-curve one, as in
        # flare.skyportal._classifier
        own = out / "ad_space.txt"
        booster = lgb.Booster(model_file=str(own if own.exists() else a.models / "ad_space.txt"))
        cols = booster.feature_name()
        known = np.sort(json.loads((out / "anomaly_calibration.json").read_text())["known_energies"])
        cal_split = "test" if h is None else "val"
        Xtr = design(split["train"].obj_id.values, h, pool_ids)[cols]
        Xkn = design(split[cal_split].obj_id.values, h, pool_ids)[cols]
        logits = lambda X: booster.predict(X, raw_score=True)
        energy = np.sort(-logsumexp(logits(Xkn), axis=1))
        gap = float(np.max(np.abs(energy - known))) if len(energy) == len(known) else np.inf
        if gap > 1e-3:
            raise SystemExit(f"{hs}: energies do not reproduce the bundled calibration (max gap {gap})")
        med = Xtr.median().fillna(0)
        iqr = (Xtr.quantile(.75) - Xtr.quantile(.25)).replace(0, 1).fillna(1)
        # float32 halves the kNN and LOF banks; flare.consensus casts the same way at run time
        emb = lambda X: np.clip((X.fillna(med) - med) / iqr, -10, 10).fillna(0).to_numpy(np.float32)
        ytr = np.array([SIX.index(c) for c in split["train"].cls])
        pipe = LUNAPipeline(target_far=FARS).fit(dict(embeddings=emb(Xtr), logits=logits(Xtr), labels=ytr))
        pipe.fit_blind(dict(embeddings=emb(Xkn), logits=logits(Xkn)), seed=a.seed)
        bundle = dict(
            pipeline=pipe, columns=list(cols), median=med.to_numpy(), iqr=iqr.to_numpy(),
            fars=list(FARS), luna_version=luna_version, horizon_days=h,
            note=(f"blind consensus (mean percentile rank over {len(pipe.active_methods)} LUNA scorers); "
                  f"scorers fitted on the {len(Xtr)} training objects, ranks frozen on half of the "
                  f"{len(Xkn)} {cal_split} objects and thresholds set on the other half"))
        path = out / "consensus.pkl.gz"
        path.write_bytes(gzip.compress(pickle.dumps(bundle), 6, mtime=0))
        man_path = out / "manifest.json"; man = json.loads(man_path.read_text())
        man["consensus.pkl.gz"] = hashlib.sha256(path.read_bytes()).hexdigest()
        man_path.write_text(json.dumps(dict(sorted(man.items())), indent=2))
        print(f"{hs}: energy parity {gap:.1e}; {len(pipe.active_methods)} scorers; thresholds "
              f"{ {k: round(v, 4) for k, v in pipe.thresholds.items()} }; "
              f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
