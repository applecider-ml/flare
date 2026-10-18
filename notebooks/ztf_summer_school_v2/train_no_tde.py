"""Train the five-class "no-TDE" FLARE model for the ZTF summer-school tutorial.

We remove every tidal disruption event (TDE) from the six-class bts6 taxonomy and train a
LightGBM classifier on the remaining five classes (SN_Ia, SN_CC, SLSN, AGN, CV). The model has
never seen a TDE, so when real TDEs pass through it they must land in one of the five known
classes, often with unusual probability patterns. This is the starting point of the anomaly
detection part of the tutorial (LUNA), where we search for the objects the classifier cannot
know about.

Data: the BTS benchmark shipped in `benchmark/` (`flare.benchmark.load_split`). We train on the
train split and stop early on the validation split, both with their TDEs removed; the test
split is never read here. The input space is the one FLARE's own anomaly layer reads: the 160
light-curve features plus the Pan-STARRS host and Gaia/WISE blocks (`flare.config.AD_SPACE_COLS`).
Hyperparameters follow the top-level bts6 model (`scripts/train_bts6.py`); there is no tuning,
and training takes well under a minute on a laptop.

Run:  python train_no_tde.py [--out flare_no_tde.txt]
The notebook calls `fit_no_tde()` directly and keeps the model in memory.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flare import benchmark as B                    # noqa: E402
from flare.config import AD_SPACE_COLS               # noqa: E402

HERE = Path(__file__).resolve().parent
KNOWN = [c for c in B.SIX if c != "TDE"]            # SN_Ia, SN_CC, SLSN, AGN, CV
TDE = B.SIX.index("TDE")
PARAMS = dict(objective="multiclass", num_class=len(KNOWN), metric="multi_logloss",
              verbosity=-1, bagging_freq=1, max_depth=-1, num_threads=4, seed=42,
              deterministic=True, force_row_wise=True,
              learning_rate=0.06, num_leaves=40, min_child_samples=20,
              feature_fraction=0.55, bagging_fraction=0.85, lambda_l2=1.0)
BETA = 0.8          # class weight (N / (K n_c))**beta, as for the bts6 models


def feature_columns(X):
    """Light-curve features + host + Gaia/WISE: the anomaly-layer space of FLARE."""
    lc = list(X.columns[:160])          # load_split puts the 160 light-curve features first
    return lc + [c for c in AD_SPACE_COLS if c in X.columns]


def known_only(split):
    """Features and five-class labels of a split, with every TDE removed."""
    X, y, table = B.load_split(split)
    keep = y != TDE
    cols = feature_columns(X)
    y5 = np.array([KNOWN.index(B.SIX[i]) for i in y[keep]])
    return X.loc[keep, cols].reset_index(drop=True), y5, int((~keep).sum())


def fit_no_tde(verbose=True):
    """Train on train (no TDEs), early stop on val (no TDEs). Returns (booster, info)."""
    Xtr, ytr, ntr = known_only("train")
    Xva, yva, nva = known_only("val")
    counts = np.bincount(ytr, minlength=len(KNOWN)).astype(float)
    w = ((len(ytr) / (len(KNOWN) * counts)) ** BETA)[ytr]
    dtr = lgb.Dataset(Xtr, label=ytr, weight=w)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(120, verbose=False)])
    pva = model.predict(Xva, num_iteration=model.best_iteration)
    acc = float((pva.argmax(1) == yva).mean())
    info = {"classes": KNOWN, "excluded_class": "TDE",
            "train_objects": int(len(ytr)), "train_tde_removed": ntr,
            "val_objects": int(len(yva)), "val_tde_removed": nva,
            "features": int(Xtr.shape[1]), "best_iteration": int(model.best_iteration),
            "val_accuracy_5class": round(acc, 4)}
    if verbose:
        print(f"train {Xtr.shape} (TDEs removed: {ntr}), val {Xva.shape} (TDEs removed: {nva})")
        print(f"best iteration {model.best_iteration}; five-class val accuracy "
              f"(no TDEs anywhere): {acc:.4f}")
    return model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "flare_no_tde.txt"),
                    help="where to write the LightGBM model (a JSON card is written next to it)")
    args = ap.parse_args()
    model, info = fit_no_tde()
    out = Path(args.out)
    model.save_model(str(out), num_iteration=model.best_iteration)
    out.with_suffix(".json").write_text(json.dumps(
        {"model": "FLARE no-TDE (five-class, summer-school teaching model)",
         "purpose": "anomaly-detection demo: TDEs are unknown to this model",
         "params": PARAMS, "beta_class_weight": BETA, **info}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
