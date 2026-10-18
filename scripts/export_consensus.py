"""Blind-consensus anomaly layer for the shipped models (full curve and day 5/10/20).

For every model directory, the LUNA scorers (luna-ood >= 0.3) are fitted on the training split
in that model's anomaly space: the embedding is the AD-space feature vector, median-imputed and
scaled by the training interquartile range, and the logits are the raw scores of the shipped
`ad_space.txt` booster that the model uses (its own at day 5, the full-curve one at day 10 and
20, on light curves truncated at that horizon). The consensus is the mean percentile rank across
the scorers.

Full-curve model: the ranks are frozen on one half of the test split (the known-class objects that
calibrate the shipped energy alarm, untouched by any fit) and the thresholds are set on the other
half, with one reference for every alert.

Day-5/10/20 models: the p-values come from the whole validation split, which no fit touched, and
the ranks are frozen on the test split. The p-values are Mondrian in the number of detections: the
calibration objects are grouped into contiguous bins of `n_obs_total` with at least MIN_BIN objects
each, and every bin has its own reference, because a single threshold flags young objects with
many detections far more often than those with few (at day 5, 0% of the test normals with <= 3
detections against 10% with >= 9 at a nominal 1%; with the bins, 0% and 0%). The full-curve model
shows no such dependence (0.3-2.2% across detection counts) and keeps the single reference.
Before anything is written, the energies recomputed here must match the bundled `known_energies`.

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
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", PKG / "paper_data"))  # python scripts/setup_data.py
FARS = (0.01, 0.05, 0.10)
MIN_BIN = 100       # calibration objects per n_det bin, so that a 1% alarm stays attainable


def design(ids, h, all_ids):
    """AD-space inputs for `ids` at horizon h (None = full curve); the horizon caches hold the
    train, validation and test rows in that order (`all_ids`)."""
    if h is None:
        lc = pd.read_parquet(ROOT / "data/features_all.parquet").set_index("obj_id").reindex(ids)
        return pd.concat([lc, context_table(ROOT, ids).reindex(ids)], axis=1)
    X = pd.read_parquet(ROOT / f"results/reconstruction/cache/X_pool_h{h}.parquet")
    assert len(X) == len(all_ids), "horizon cache does not match train + val + test"
    X.index = all_ids
    return X.reindex(ids)


def ndet_bins(n, min_size=MIN_BIN):
    """Upper edges of contiguous bins of the detection count, each holding at least `min_size`
    calibration objects; the last bin is open-ended (edge = inf)."""
    vals, counts = np.unique(n, return_counts=True)
    edges, acc = [], 0
    for v, c in zip(vals, counts):
        acc += c
        if acc >= min_size:
            edges.append(float(v)); acc = 0
    # the last full bin absorbs any smaller remainder and is left open-ended
    return edges[:-1] + [float("inf")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--horizons", nargs="+", default=["full", "5", "10", "20"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    split = {s: pd.read_csv(ROOT / f"data/splits/{s}.csv") for s in ("train", "val", "test")}
    all_ids = pd.concat([split["train"], split["val"], split["test"]]).obj_id.values
    for hs in a.horizons:
        h = None if hs == "full" else int(hs)
        out = a.models if h is None else a.models / "early" / f"h{h}"
        # the phase's own AD booster where one ships (day 5), else the full-curve one, as in
        # flare.skyportal._classifier
        own = out / "ad_space.txt"
        booster = lgb.Booster(model_file=str(own if own.exists() else a.models / "ad_space.txt"))
        cols = booster.feature_name()
        known = np.sort(json.loads((out / "anomaly_calibration.json").read_text())["known_energies"])
        cal_split, ref_split = ("test", "test") if h is None else ("val", "test")
        Xtr = design(split["train"].obj_id.values, h, all_ids)[cols]
        Xkn = design(split[cal_split].obj_id.values, h, all_ids)[cols]
        Xref = design(split[ref_split].obj_id.values, h, all_ids)[cols]
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
        if h is None:       # halves of the test split, one reference
            pipe.fit_blind(dict(embeddings=emb(Xkn), logits=logits(Xkn)), seed=a.seed)
            mondrian, edges = None, None
        else:               # ranks on the test split, Mondrian p-values on the validation split
            cal = dict(embeddings=emb(Xkn), logits=logits(Xkn))
            pipe.fit_blind(dict(embeddings=emb(Xref), logits=logits(Xref)), cal)
            cal_score = pipe._combiner.predict(pipe._scores_to_matrix(pipe.score(cal)))
            n_obs = Xkn["n_obs_total"].to_numpy(float)
            edges = ndet_bins(n_obs)
            b = np.searchsorted(edges, n_obs, side="left")
            mondrian = dict(feature="n_obs_total", edges=edges,
                            cal_scores=[np.sort(cal_score[b == k]).tolist() for k in range(len(edges))])
        bundle = dict(
            pipeline=pipe, columns=list(cols), median=med.to_numpy(), iqr=iqr.to_numpy(),
            fars=list(FARS), luna_version=luna_version, horizon_days=h, mondrian=mondrian,
            note=(f"blind consensus (mean percentile rank over {len(pipe.active_methods)} LUNA scorers); "
                  f"scorers fitted on the {len(Xtr)} training objects; "
                  + (f"ranks frozen on half of the {len(Xkn)} {cal_split} objects and thresholds set on "
                     f"the other half" if h is None else
                     f"ranks frozen on the {len(Xref)} {ref_split} objects, p-values from the {len(Xkn)} "
                     f"{cal_split} objects separately for each bin of the number of detections")))
        path = out / "consensus.pkl.gz"
        path.write_bytes(gzip.compress(pickle.dumps(bundle), 6, mtime=0))
        man_path = out / "manifest.json"; man = json.loads(man_path.read_text())
        man["consensus.pkl.gz"] = hashlib.sha256(path.read_bytes()).hexdigest()
        man_path.write_text(json.dumps(dict(sorted(man.items())), indent=2))
        bins = "one reference" if mondrian is None else \
            f"n_det bins {edges} with {[len(s) for s in mondrian['cal_scores']]} calibration objects"
        print(f"{hs}: energy parity {gap:.1e}; {len(pipe.active_methods)} scorers; {bins}; "
              f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
