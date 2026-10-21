"""Full-curve anomaly-space booster and its energy calibration (models/bts6/ad_space.txt,
anomaly_calibration.json).

The booster is the six-class LightGBM of train_early_ad.py on the full curves: AD space = 160
light-curve features + host + Gaia/WISE (no M_pseudo, no counterpart block), trained on the train
split with an internal 15% early-stopping subset, class weights (n / 6 n_c)^0.8. The energy
calibration (legacy diagnostic next to the consensus) uses the validation split as the known
population and the benchmark's own out-of-taxonomy objects (source "bts"); out-of-taxonomy objects
from other sources are never read.
Run: FLARE_PAPER_DATA=<data root> python scripts/train_ad_space.py [--threads 16]
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
from evaluate_unified import context_table, SIX, TOP  # noqa: E402
from flare.config import AD_SPACE_COLS  # noqa: E402
from flare.analysis import feature_frame  # noqa: E402
from flare.data import load_events  # noqa: E402
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", PKG / "paper_data"))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--threads", type=int, default=16); a = p.parse_args()
    out = PKG / "models/bts6"
    F = pd.read_parquet(ROOT / "data/features_all.parquet").set_index("obj_id")
    def design(ids):
        return pd.concat([F.reindex(ids), context_table(ROOT, ids).reindex(ids)], axis=1)
    train = pd.read_csv(ROOT / "data/splits/train.csv"); val = pd.read_csv(ROOT / "data/splits/val.csv")
    cols = list(F.columns) + AD_SPACE_COLS
    Xtr, Xva = design(train.obj_id.values)[cols], design(val.obj_id.values)[cols]
    y = np.array([SIX.index(c) for c in train.cls])
    tr, es = train_test_split(np.arange(len(train)), test_size=.15, stratify=y, random_state=42)
    w = (len(tr) / (6 * np.bincount(y[tr], minlength=6))) ** .8
    d = lgb.Dataset(Xtr.iloc[tr], label=y[tr], weight=w[y[tr]]); e = lgb.Dataset(Xtr.iloc[es], label=y[es], reference=d)
    m = lgb.train(dict(TOP, num_class=6, verbosity=-1, seed=42, num_threads=a.threads, bagging_freq=1), d, 2000,
                  valid_sets=[e], callbacks=[lgb.early_stopping(80, verbose=False)])
    m.save_model(str(out / "ad_space.txt"))
    E = lambda Z: -logsumexp(m.predict(Z, raw_score=True), axis=1)
    novel = pd.read_csv(ROOT / "data/splits/anomaly.csv")
    novel = novel[novel.get("source", pd.Series("bts", index=novel.index)).fillna("bts") == "bts"]
    ctx = context_table(ROOT, novel.obj_id.values)
    R = pd.concat([feature_frame(load_events(str(ROOT / "data" / r.filepath)), ctx.loc[r.obj_id].to_dict())[0]
                   for r in novel.itertuples()], ignore_index=True)
    known, rare = E(Xva), E(R.reindex(columns=cols))
    lr = LogisticRegression(class_weight="balanced", C=1e6, max_iter=1000).fit(
        np.r_[known, rare][:, None], np.r_[np.zeros(len(known)), np.ones(len(rare))])
    (out / "anomaly_calibration.json").write_text(json.dumps(dict(
        method="platt_on_energy_balanced", coef=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]),
        base_rate=round(len(rare) / (len(rare) + len(known)), 6), n_known=int(len(known)), n_anomalous=int(len(rare)),
        known_energies=[round(float(v), 5) for v in known],
        note=f"energies of ad_space.txt on the validation split ({len(known)} known-class objects) and on the "
             f"{len(rare)} out-of-taxonomy objects of the benchmark; class-balanced logistic fit so exp(a*E+b) is the likelihood ratio"), indent=1))
    man_path = out / "manifest.json"; man = json.loads(man_path.read_text())
    for f in ("ad_space.txt", "anomaly_calibration.json"):
        man[f] = hashlib.sha256((out / f).read_bytes()).hexdigest()
    man_path.write_text(json.dumps(dict(sorted(man.items())), indent=2))
    print(f"ad_space.txt: {m.best_iteration} rounds, {len(cols)} columns; calibration on {len(known)} known + {len(rare)} rare", flush=True)


if __name__ == "__main__":
    main()
