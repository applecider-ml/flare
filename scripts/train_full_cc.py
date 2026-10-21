"""Full-curve BTS6 hierarchy with cross-conformal calibration (2026-10-06).

Same recipe as scripts/train_early_cc.py at the 100-day horizon: five stratified folds on
train+val (8,922 objects; the test split is not read), scores 1 - P(true class) pooled out
of fold, leave-one-fold-out coverage reported, final weights fitted on all of train+val.
The anomaly-space booster and its calibration are untouched (they do not depend on the
classifier weights). Run: python scripts/train_full_cc.py --out models/bts6_cc --threads 16
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
from evaluate_unified import context_table, SIX, quantile  # noqa: E402
from train_early_cc import hierarchy_nocal, sets_from  # noqa: E402
from flare.context import lumdist_mpc  # noqa: E402
ROOT = PKG.parent / "bts_dataset"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=PKG / "models/bts6_cc")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--rounds", type=int, default=2000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(ROOT / "data/splits/train.csv"); val = pd.read_csv(ROOT / "data/splits/val.csv")
    pool = pd.concat([train, val], ignore_index=True); n = len(pool)
    y = np.array([SIX.index(c) for c in pool.cls])
    # full-curve features: the 160 light-curve columns from features_all, the context columns from the
    # day-5 cache (they do not depend on the horizon), M_pseudo from the full-curve peak (as evaluate_unified)
    full = pd.read_parquet(ROOT / "data/features_all.parquet").set_index("obj_id").reindex(pool.obj_id)
    X = pd.read_parquet(ROOT / "results/reconstruction/cache/X_pool_h5.parquet").iloc[:n].copy()
    for col in full.columns:
        X[col] = full[col].to_numpy()
    ctx = context_table(ROOT, pool.obj_id.values)
    peak = 23.9 - 2.5 * full[["g_logflux_peak", "r_logflux_peak", "i_logflux_peak"]].max(axis=1).to_numpy()
    X["M_pseudo"] = peak - 5 * np.log10(lumdist_mpc(ctx.loc[pool.obj_id].z_phot.to_numpy()) * 1e5)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    P = np.full((n, 6), np.nan); fold_id = np.full(n, -1)
    for k, (tr0, te) in enumerate(folds):
        tr, es = train_test_split(tr0, test_size=.15, stratify=y[tr0], random_state=k)
        m = hierarchy_nocal(X, y, tr, es, a)
        P[te] = m.predict_proba(X.iloc[te]); fold_id[te] = k
        print(f"fold {k}: acc {np.mean(P[te].argmax(1) == y[te]):.3f}", flush=True)
    pred = P.argmax(1)
    scores = {c: sorted((1 - P[y == i, i]).tolist()) for i, c in enumerate(SIX)}
    cov = np.zeros(n, bool); size = np.zeros(n, int)
    for k in range(5):
        cal = {c: [1 - P[j, i] for j in np.where((fold_id != k) & (y == i))[0]] for i, c in enumerate(SIX)}
        mem = sets_from(P[fold_id == k], cal)
        cov[fold_id == k] = mem[np.arange(mem.shape[0]), y[fold_id == k]]; size[fold_id == k] = mem.sum(1)
    per_class = {c: dict(n=int((y == i).sum()), coverage=round(float(cov[y == i].mean()), 3),
                         f1=round(float(f1_score(y == i, pred == i)), 3), singleton=round(float((size[y == i] == 1).mean()), 3))
                 for i, c in enumerate(SIX)}
    rep = dict(n_pool=n, oof_accuracy=round(float((pred == y).mean()), 4), oof_macro_f1=round(float(f1_score(y, pred, average="macro")), 4),
               lofo_marginal_coverage=round(float(cov.mean()), 4), singleton_rate=round(float((size == 1).mean()), 4),
               singleton_correct=round(float(((size == 1) & (pred == y)).sum() / max((size == 1).sum(), 1)), 4),
               empty_rate=round(float((size == 0).mean()), 4), mean_set_size=round(float(size.mean()), 3), per_class=per_class,
               p_value_floor={c: round(1 / (len(v) + 1), 4) for c, v in scores.items()})
    print(json.dumps(rep, indent=1), flush=True)
    tr, es = train_test_split(np.arange(n), test_size=.15, stratify=y, random_state=42)
    final = hierarchy_nocal(X, y, tr, es, a)
    final.top.save_model(str(a.out / "top.txt")); final.branch.save_model(str(a.out / "slsn_branch.txt"))
    lofo = {c: v["coverage"] for c, v in per_class.items()}
    (a.out / "conformal.json").write_text(json.dumps(dict(
        alpha=.1, method="cross-conformal", qhat={c: float(quantile(scores[c], .9)) for c in SIX}, cal_scores=scores, n_cal=n,
        n_cal_per_class={c: len(v) for c, v in scores.items()}, p_value_floor=rep["p_value_floor"],
        empirical_coverage=lofo, lofo_coverage=lofo, marginal_coverage=rep["lofo_marginal_coverage"],
        empirical_coverage_note="leave-one-fold-out coverage at alpha 0.10 over train+val (each fold's sets calibrated on the other "
                                "four folds' out-of-fold scores); the paper's five-fold numbers over the full sample are in the model card",
        note="scores 1-P(true class) are out-of-fold over five stratified folds of train+val (8,922 objects); the exported weights are "
             "the same recipe fitted on all of train+val (85/15 fit / early stop)"), indent=2, allow_nan=False))
    old = json.loads((PKG / "models/bts6/model_card.json").read_text())
    card = {**old, "branch_threshold": final.branch_threshold, "feature_names": list(X.columns),
            "training": {**old.get("training", {}), "objects": n, "refit": "2026-10-06: fitted on train+val with cross-conformal calibration "
                         "(scripts/train_full_cc.py); the 2026-09 weights (train only, split calibration on val) are kept as models/bts6_valcal_20260915"},
            "oof_development_numbers": rep}
    (a.out / "model_card.json").write_text(json.dumps(card, indent=2))
    (a.out / "manifest.json").write_text(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.out.iterdir())}, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
