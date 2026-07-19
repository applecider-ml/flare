"""Train FLARE from raw data — reproduces models/flare_lgbm.txt + conformal.json.

Steps:
  1. quality-filter each split
  2. extract 160 physics features per object (light-curve pkg + Bazin fits)
  3. Optuna-tune LightGBM (hyperparams + class-weight beta) on val
  4. refit, save booster + Mondrian conformal calibrated on val

Data dir must contain manifest_{train,val,test}.csv, per-object .npz, and
feature_stats_day100.npz. Set FLARE_DATA or pass --data-dir.

Run:  python scripts/train.py --data-dir /path/to/photo_events --trials 100
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flare import conformal                                    # noqa: E402
from flare import data as D                                    # noqa: E402
from flare.features import extract_object_features             # noqa: E402
from flare.metrics import compute_metrics                      # noqa: E402
from flare.taxonomy import ID2BROAD_ID, NUM_CLASSES            # noqa: E402

MODELS = ROOT / "models"
CACHE = ROOT / "artifacts"
CACHE.mkdir(exist_ok=True)


def _feat_worker(args):
    obj_id, fp, label, horizon = args
    try:
        f = extract_object_features(str(fp), horizon_days=horizon)
    except Exception:
        f = {}
    f["obj_id"] = obj_id
    f["label"] = label
    return f


def extract_split(split, horizon, workers):
    cache = CACHE / f"features_{split}_h{int(horizon)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = D.load_manifest(split)
    kept, _ = D.filter_manifest_quality(df, horizon_days=horizon)
    jobs = [(r.obj_id, r.filepath, r.label, horizon) for r in kept.itertuples()]
    t0 = time.time()
    with Pool(workers) as pool:
        rows = pool.map(_feat_worker, jobs, chunksize=64)
    out = pd.DataFrame(rows)
    feat_cols = sorted(c for c in out.columns if c not in ("obj_id", "label"))
    out = out[["obj_id", "label"] + feat_cols]
    out.to_parquet(cache, index=False)
    print(f"  {split}: {len(out)} objs, {len(feat_cols)} feats, "
          f"{time.time()-t0:.0f}s", flush=True)
    return out


def xy(df):
    y = df["label"].map(lambda x: ID2BROAD_ID[int(x)]).values
    X = df.drop(columns=["obj_id", "label"])
    return X, y


def cw(y, beta):
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    return ((len(y) / (NUM_CLASSES * counts)) ** beta)[y]


def fit(params, beta, Xtr, ytr, Xva, yva, seed):
    dtr = lgb.Dataset(Xtr, label=ytr, weight=cw(ytr, beta))
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    m = lgb.train({**params, "seed": seed}, dtr, num_boost_round=2000,
                  valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.predict(Xva, num_iteration=m.best_iteration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()
    if args.data_dir:
        D.DATA_DIR = Path(args.data_dir)
    tag = "" if args.horizon == 100.0 else f"_day{int(args.horizon)}"

    tr = extract_split("train", args.horizon, args.workers)
    va = extract_split("val", args.horizon, args.workers)
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    print(f"train {Xtr.shape}, val {Xva.shape}", flush=True)

    base = {"objective": "multiclass", "num_class": NUM_CLASSES,
            "metric": "multi_logloss", "verbosity": -1, "bagging_freq": 1,
            "max_depth": -1, "num_threads": 0}

    def objective(t):
        p = {**base,
             "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
             "num_leaves": t.suggest_int("num_leaves", 16, 256, log=True),
             "min_child_samples": t.suggest_int("min_child_samples", 5, 100, log=True),
             "feature_fraction": t.suggest_float("feature_fraction", 0.4, 1.0),
             "bagging_fraction": t.suggest_float("bagging_fraction", 0.5, 1.0),
             "lambda_l1": t.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
             "lambda_l2": t.suggest_float("lambda_l2", 1e-8, 10.0, log=True)}
        beta = t.suggest_float("beta", 0.0, 1.0)
        _, pva = fit(p, beta, Xtr, ytr, Xva, yva, args.seed)
        m = compute_metrics(yva, pva)
        return 0.5 * m["macro_auprc"] + 0.5 * m["macro_f1"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    best = study.best_trial
    beta = best.params.pop("beta")
    params = {**base, **best.params}
    print(f"best val score {best.value:.4f}, beta={beta:.3f}", flush=True)

    model, pva = fit(params, beta, Xtr, ytr, Xva, yva, args.seed)
    model.save_model(str(MODELS / f"flare_lgbm{tag}.txt"))

    q = conformal.calibrate(pva, yva, alpha=args.alpha)
    conformal.save(MODELS / f"conformal{tag}.json", q, args.alpha, len(yva))
    (MODELS / f"model_card{tag}.json").write_text(json.dumps({
        "horizon_days": args.horizon, "features": Xtr.shape[1],
        "beta": beta, "params": best.params,
        "val_score": best.value, "trials": args.trials,
    }, indent=2))
    print(f"saved -> {MODELS}/flare_lgbm{tag}.txt, conformal{tag}.json", flush=True)


if __name__ == "__main__":
    main()
