"""Train the 4-class "no-TDE" FLARE model for the ZTF summer-school tutorial.

Purpose: a classifier that has NEVER seen a TDE. When real TDEs are pushed
through it they must land in one of the four known classes (usually AGN) —
often with atypical probability patterns. This is the setup for the anomaly-
detection part of the tutorial (LUNA): finding the objects the classifier
cannot know about.

Uses the published FLARE hyperparameters (models/model_card.json) — no tuning,
trains in ~a minute on a laptop. Saves:
    models/flare_lgbm_no_tde.txt
    models/model_card_no_tde.json

Run:  python train_no_tde.py [--data-dir /path/to/photo_events]
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flare import data as D                                    # noqa: E402
from flare.features import extract_object_features             # noqa: E402
from flare.taxonomy import BROAD_CLASSES, ID2BROAD_ID          # noqa: E402

MODELS = ROOT / "models"
KNOWN = ["SNI", "SNII", "CV", "AGN"]          # class order preserved, TDE removed
TDE_IDX = BROAD_CLASSES.index("TDE")


def _work(a):
    oid, fp, lab, horizon = a
    try:
        f = extract_object_features(str(fp), horizon_days=horizon)
    except Exception:
        f = {}
    f["obj_id"], f["label"] = oid, lab
    return f


def load_features(split, horizon=100.0, workers=8):
    """Features for the quality-filtered split (cached in flare/artifacts)."""
    cache = ROOT / "artifacts" / f"features_{split}_h{int(horizon)}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        from multiprocessing import Pool
        mf = D.load_manifest(split)
        kept, _ = D.filter_manifest_quality(mf, horizon_days=horizon)
        jobs = [(r.obj_id, r.filepath, r.label, horizon) for r in kept.itertuples()]
        with Pool(workers) as p:
            rows = p.map(_work, jobs, chunksize=64)
        df = pd.DataFrame(rows)
        fc = sorted(c for c in df.columns if c not in ("obj_id", "label"))
        df = df[["obj_id", "label"] + fc]
        cache.parent.mkdir(exist_ok=True)
        df.to_parquet(cache, index=False)
    y = df["label"].map(lambda x: ID2BROAD_ID[int(x)]).values
    X = df.drop(columns=["obj_id", "label"])
    return df["obj_id"].values, X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.data_dir:
        D.DATA_DIR = Path(args.data_dir)

    card = json.load(open(MODELS / "model_card.json"))
    params = {**card["params"], "objective": "multiclass",
              "num_class": len(KNOWN), "metric": "multi_logloss",
              "verbosity": -1, "bagging_freq": 1, "max_depth": -1,
              "num_threads": 0, "seed": args.seed}
    beta = card["beta_class_weight"]

    _, Xtr, ytr = load_features("train")
    _, Xva, yva = load_features("val")

    # remove every TDE from training AND validation — the model must never see one
    mtr, mva = ytr != TDE_IDX, yva != TDE_IDX
    Xtr, ytr = Xtr[mtr], ytr[mtr]
    Xva, yva = Xva[mva], yva[mva]
    print(f"train {Xtr.shape} (TDEs removed: {(~mtr).sum()}), "
          f"val {Xva.shape} (removed: {(~mva).sum()})")

    counts = np.bincount(ytr, minlength=len(KNOWN)).astype(float)
    w = ((len(ytr) / (len(KNOWN) * counts)) ** beta)[ytr]
    dtr = lgb.Dataset(Xtr, label=ytr, weight=w)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])

    pva = model.predict(Xva, num_iteration=model.best_iteration)
    acc = (pva.argmax(1) == yva).mean()
    print(f"4-class val accuracy (no TDEs anywhere): {acc:.4f}")

    model.save_model(str(MODELS / "flare_lgbm_no_tde.txt"))
    (MODELS / "model_card_no_tde.json").write_text(json.dumps({
        "model": "FLARE no-TDE (4-class, summer-school teaching model)",
        "classes": KNOWN, "excluded_class": "TDE",
        "purpose": "anomaly-detection demo: TDEs are unknown to this model",
        "params": card["params"], "beta_class_weight": beta,
        "trees": model.num_trees(), "val_accuracy_4class": round(float(acc), 4),
    }, indent=2))
    print(f"saved -> {MODELS}/flare_lgbm_no_tde.txt ({model.num_trees()} trees)")


if __name__ == "__main__":
    main()
