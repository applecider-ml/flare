"""Phase classifiers with cross-conformal calibration and phase-calibrated anomaly energies.

For each horizon h in (5, 10, 20), on the train+val pool (8,922 objects; the test split is
not read):
  * five stratified folds; each fold model is the paper's hierarchy fitted on the other four
    (85/15 fit / early stop); its held-out probabilities are out-of-fold (OOF);
  * conformal calibration scores s = 1 - P(true class) pooled over the five folds
    (cross-conformal), so the rare classes get 60 to 160 calibration objects instead of
    11 to 23 and the p-value floors drop below alpha;
  * honest numbers: OOF accuracy / macro F1, and leave-one-fold-out coverage and singleton
    rate of the alpha = 0.1 sets (calibrated on the other folds' OOF scores);
  * the exported weights are the same recipe fitted on all of train+val;
  * anomaly: energies of the shipped full-curve AD booster on the validation split at
    horizon h (known_energies) and on the 52 out-of-taxonomy rarities at horizon h; a
    class-balanced logistic fit gives the likelihood ratio, as in the full-curve calibration.
Run: python scripts/train_early_cc.py --out models/bts6/early_cc --threads 16
"""
import argparse, hashlib, json, sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
from evaluate_unified import fit, context_table, SIX, Y5, TOP, BR, quantile  # noqa: E402
from flare.analysis import feature_frame, observed_events  # noqa: E402
from flare.config import HOST_COLS, GW_COLS, POS_COLS  # noqa: E402
from flare.data import load_events  # noqa: E402
from flare.hierarchical import HierarchicalFlare  # noqa: E402

ROOT = PKG.parent / "bts_dataset"


def hierarchy_nocal(X, y, tr, es, args):
    xt = X.iloc[tr].copy()
    rng = np.random.default_rng(42)
    for block in (HOST_COLS, GW_COLS, POS_COLS, ["M_pseudo"]):
        xt.loc[rng.random(len(xt)) < .3, block] = np.nan
    y5 = Y5[y]
    counts = np.bincount(y5[tr], minlength=5)
    weights = (len(tr) / (5 * counts)) ** .8
    top = fit(xt, y5[tr], X.iloc[es], y5[es], TOP, args, weights[y5[tr]])
    ct, ce = np.isin(y[tr], [1, 2]), np.isin(y[es], [1, 2])
    yt, ye = (y[tr][ct] == 2).astype(int), (y[es][ce] == 2).astype(int)
    wb = np.where(yt == 1, (len(yt) / max(yt.sum(), 1)) ** .5, 1.)
    branch = fit(xt.loc[ct], yt, X.iloc[es].loc[ce], ye, BR, args, wb)
    pb = branch.predict(X.iloc[es].loc[ce], num_threads=args.threads)
    thr = max(np.linspace(.05, .9, 35), key=lambda t: f1_score(ye, pb > t, zero_division=0))
    return HierarchicalFlare(top, branch, thr)


def sets_from(P, cal, alpha=.1):
    q = np.array([quantile(cal[c], 1 - alpha) for c in SIX])
    return (1 - P) <= q[None, :]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=PKG / "models/bts6/early_cc")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(ROOT / "data/splits/train.csv"); val = pd.read_csv(ROOT / "data/splits/val.csv")
    pool = pd.concat([train, val], ignore_index=True)
    n = len(pool); y = np.array([SIX.index(c) for c in pool.cls])
    val_rows = np.arange(len(train), n)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    novel = pd.read_csv(ROOT / "data/splits/anomaly.csv")
    ctx = context_table(ROOT, novel.obj_id.values)
    ad = lgb.Booster(model_file=str(PKG / "models/bts6/ad_space.txt"))
    energy = lambda X: -logsumexp(ad.predict(X.reindex(columns=ad.feature_name()), raw_score=True), axis=1)
    report = {}
    for h in a.horizons:
        z = np.load(ROOT / f"results/reconstruction/forecast_v2_h{h}.npz", allow_pickle=True)
        assert np.array_equal(z["obj_id"][:n], pool.obj_id)
        X = pd.read_parquet(ROOT / f"results/reconstruction/cache/X_pool_h{h}.parquet").iloc[:n].copy()
        P = np.full((n, 6), np.nan); fold_id = np.full(n, -1)
        for k, (tr0, te) in enumerate(folds):
            tr, es = train_test_split(tr0, test_size=.15, stratify=y[tr0], random_state=k)
            m = hierarchy_nocal(X, y, tr, es, a)
            P[te] = m.predict_proba(X.iloc[te]); fold_id[te] = k
            print(f"h={h} fold {k}: acc {np.mean(P[te].argmax(1) == y[te]):.3f}", flush=True)
        pred = P.argmax(1)
        scores = {c: sorted((1 - P[y == i, i]).tolist()) for i, c in enumerate(SIX)}
        # leave-one-fold-out coverage of the alpha=0.1 sets
        cov = np.zeros(n, bool); size = np.zeros(n, int)
        for k in range(5):
            cal = {c: [1 - P[j, i] for j in np.where((fold_id != k) & (y == i))[0]] for i, c in enumerate(SIX)}
            mem = sets_from(P[fold_id == k], cal)
            cov[fold_id == k] = mem[np.arange(mem.shape[0]), y[fold_id == k]]; size[fold_id == k] = mem.sum(1)
        per_class = {c: dict(n=int((y == i).sum()), coverage=round(float(cov[y == i].mean()), 3),
                             recall=round(float((pred[y == i] == i).mean()), 3),
                             singleton=round(float((size[y == i] == 1).mean()), 3)) for i, c in enumerate(SIX)}
        single_ok = float(((size == 1) & (pred == y)).sum() / max((size == 1).sum(), 1))
        rep = dict(horizon_days=h, n_pool=n, oof_accuracy=round(float((pred == y).mean()), 4),
                   oof_macro_f1=round(float(f1_score(y, pred, average="macro")), 4),
                   lofo_marginal_coverage=round(float(cov.mean()), 4), singleton_rate=round(float((size == 1).mean()), 4),
                   singleton_correct=round(single_ok, 4), empty_rate=round(float((size == 0).mean()), 4),
                   mean_set_size=round(float(size.mean()), 3), per_class=per_class,
                   p_value_floor={c: round(1 / (len(v) + 1), 4) for c, v in scores.items()})
        print(json.dumps(rep, indent=1), flush=True)
        # final weights on all of train+val
        tr, es = train_test_split(np.arange(n), test_size=.15, stratify=y, random_state=42)
        final = hierarchy_nocal(X, y, tr, es, a)
        out = a.out / f"h{h}"; out.mkdir()
        final.top.save_model(str(out / "top.txt")); final.branch.save_model(str(out / "slsn_branch.txt"))
        qhat = {c: float(quantile(scores[c], .9)) for c in SIX}
        (out / "conformal.json").write_text(json.dumps(dict(
            alpha=.1, method="cross-conformal", qhat=qhat, cal_scores=scores, n_cal=n,
            n_cal_per_class={c: len(v) for c, v in scores.items()},
            p_value_floor=rep["p_value_floor"],
            lofo_coverage={c: v["coverage"] for c, v in per_class.items()}, lofo_marginal_coverage=rep["lofo_marginal_coverage"],
            note="scores 1-P(true class) are out-of-fold over five stratified folds of train+val (8,922 objects); "
                 "the exported weights are the same recipe fitted on all of train+val; coverage above is "
                 "leave-one-fold-out at alpha 0.1"), indent=2, allow_nan=False))
        # anomaly calibration at this horizon
        known = energy(X.iloc[val_rows])
        rows = []
        for r in novel.itertuples():
            arr = load_events(str(ROOT / "data" / r.filepath))
            try:
                f, _ = feature_frame(observed_events(arr, h), ctx.loc[r.obj_id].to_dict()); rows.append(f)
            except ValueError:
                pass
        rare = energy(pd.concat(rows, ignore_index=True))
        E = np.r_[known, rare]; lab = np.r_[np.zeros(len(known)), np.ones(len(rare))]
        lr = LogisticRegression(class_weight="balanced", C=1e6, max_iter=1000).fit(E[:, None], lab)
        thr99 = float(np.percentile(known, 99))
        anom = dict(method="platt_on_energy_balanced", horizon_days=h, coef=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]),
                    base_rate=round(len(rare) / (len(rare) + len(known)), 6), n_known=int(len(known)), n_anomalous=int(len(rare)),
                    auroc=round(float(roc_auc_score(lab, E)), 3), tpr_at_1pct_far=round(float((rare > thr99).mean()), 3),
                    threshold_1pct_far=round(thr99, 4), known_energies=[round(float(e), 5) for e in known],
                    note=f"energies of the full-curve AD booster on the validation split (1575 known-class objects) and on the "
                         f"out-of-taxonomy rarities, both truncated at {h} days; class-balanced logistic fit so exp(a*E+b) is the "
                         f"likelihood ratio; novelty p-values and the 1% FAR threshold are relative to this known population")
        (out / "anomaly_calibration.json").write_text(json.dumps(anom, indent=1))
        rep["anomaly"] = {k: anom[k] for k in ("auroc", "tpr_at_1pct_far", "n_anomalous")}
        print("anomaly", rep["anomaly"], flush=True)
        card = dict(horizon_days=h, branch_threshold=final.branch_threshold, feature_names=list(X.columns),
                    training="2026-10-05: fitted on train+val (85/15 fit/early stop); conformal and anomaly calibration are "
                             "cross-conformal / validation-split based, see conformal.json and anomaly_calibration.json; "
                             "horizon caches rebuilt with prefix colours (scripts/build_horizon_cache.py); test labels not read",
                    oof_development_numbers=rep, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (out / "model_card.json").write_text(json.dumps(card, indent=2))
        manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir())}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        report[h] = rep
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
