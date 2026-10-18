"""Anomaly-space booster per horizon: the energy scorer (-logsumexp of raw multiclass scores)
fitted on light curves truncated at h days, in the AD space (light-curve features + host +
Gaia/WISE; no M_pseudo, no counterpart block). Trained on the train split, calibrated on the
validation split and the 52 out-of-taxonomy rarities at the same horizon. Kept only where it
beats the full-curve booster's AUROC at that horizon (the comparison is printed)."""
import argparse, hashlib, json, sys
import os
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
from evaluate_unified import context_table, SIX, TOP  # noqa: E402
from flare.analysis import feature_frame, observed_events  # noqa: E402
from flare.config import AD_SPACE_COLS  # noqa: E402
from flare.data import load_events  # noqa: E402
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", PKG / "paper_data"))  # python paper/setup_data.py


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=PKG / "models/bts6/early_cc")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    train = pd.read_csv(ROOT / "data/splits/train.csv"); val = pd.read_csv(ROOT / "data/splits/val.csv")
    pool = pd.concat([train, val], ignore_index=True); n = len(pool)
    y = np.array([SIX.index(c) for c in pool.cls]); tr_all = np.arange(len(train)); va = np.arange(len(train), n)
    novel = pd.read_csv(ROOT / "data/splits/anomaly.csv"); ctx = context_table(ROOT, novel.obj_id.values)
    full = lgb.Booster(model_file=str(PKG / "models/bts6/ad_space.txt"))
    lc_cols = [c for c in full.feature_name()]            # the shipped AD space: 160 lc + host + gaia/wise
    for h in (5, 10, 20):
        X = pd.read_parquet(ROOT / f"results/reconstruction/cache/X_pool_h{h}.parquet").iloc[:n]
        Xa = X.reindex(columns=lc_cols)
        tr, es = train_test_split(tr_all, test_size=.15, stratify=y[tr_all], random_state=42)
        w = (len(tr) / (6 * np.bincount(y[tr], minlength=6))) ** .8
        d = lgb.Dataset(Xa.iloc[tr], label=y[tr], weight=w[y[tr]]); e = lgb.Dataset(Xa.iloc[es], label=y[es], reference=d)
        m = lgb.train(dict(TOP, num_class=6, verbosity=-1, seed=42, num_threads=a.threads, bagging_freq=1), d, 2000,
                      valid_sets=[e], callbacks=[lgb.early_stopping(80, verbose=False)])
        rows = []
        for r in novel.itertuples():
            try:
                f, _ = feature_frame(observed_events(load_events(str(ROOT / "data" / r.filepath)), h), ctx.loc[r.obj_id].to_dict()); rows.append(f)
            except ValueError:
                pass
        R = pd.concat(rows, ignore_index=True).reindex(columns=lc_cols)
        res = {}
        for tag, b in (("full", full), ("h", m)):
            E = lambda Z: -logsumexp(b.predict(Z, raw_score=True), axis=1)
            known, rare = E(Xa.iloc[va]), E(R)
            lab = np.r_[np.zeros(len(known)), np.ones(len(rare))]; Ev = np.r_[known, rare]
            thr = float(np.percentile(known, 99))
            res[tag] = dict(auroc=float(roc_auc_score(lab, Ev)), tpr1=float((rare > thr).mean()), known=known, rare=rare, thr=thr)
        print(f"h={h}: full booster AUROC {res['full']['auroc']:.3f} TPR@1% {res['full']['tpr1']:.3f} | "
              f"phase booster AUROC {res['h']['auroc']:.3f} TPR@1% {res['h']['tpr1']:.3f} (rounds {m.best_iteration})", flush=True)
        if res["h"]["auroc"] <= res["full"]["auroc"]:
            print(f"  keeping the full-curve booster at h={h}", flush=True); continue
        out = a.out / f"h{h}"
        m.save_model(str(out / "ad_space.txt"))
        known, rare = res["h"]["known"], res["h"]["rare"]
        lr = LogisticRegression(class_weight="balanced", C=1e6, max_iter=1000).fit(np.r_[known, rare][:, None], np.r_[np.zeros(len(known)), np.ones(len(rare))])
        (out / "anomaly_calibration.json").write_text(json.dumps(dict(
            method="platt_on_energy_balanced", horizon_days=h, scorer="ad_space.txt (phase booster)",
            coef=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]), base_rate=round(len(rare) / (len(rare) + len(known)), 6),
            n_known=int(len(known)), n_anomalous=int(len(rare)), auroc=round(res["h"]["auroc"], 3), tpr_at_1pct_far=round(res["h"]["tpr1"], 3),
            auroc_full_curve_booster=round(res["full"]["auroc"], 3), threshold_1pct_far=round(res["h"]["thr"], 4),
            known_energies=[round(float(v), 5) for v in known],
            note=f"energy of a 6-class booster trained in the AD space on light curves truncated at {h} days (train split); "
                 f"known population = validation split at {h} days, rarities = the 52 out-of-taxonomy objects at {h} days; "
                 "class-balanced logistic fit so exp(a*E+b) is the likelihood ratio"), indent=1))
        card = json.loads((out / "model_card.json").read_text())
        card["anomaly_space"] = dict(model="ad_space.txt", trained_at_horizon=h, auroc_rare52=round(res["h"]["auroc"], 3),
                                     auroc_full_curve_booster=round(res["full"]["auroc"], 3))
        card["oof_development_numbers"]["anomaly"] = dict(auroc=round(res["h"]["auroc"], 3), tpr_at_1pct_far=round(res["h"]["tpr1"], 3), n_anomalous=int(len(rare)), scorer="phase booster")
        (out / "model_card.json").write_text(json.dumps(card, indent=2))
        (out / "manifest.json").write_text(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.name != "manifest.json"}, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
