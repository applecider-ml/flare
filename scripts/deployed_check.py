"""Coverage of the shipped (final) classifiers with their cross-conformal calibration on the test split,
which no fit or calibration touched: full-curve model on full-curve features, phase models on the
horizon caches. Writes `deployed_check` into each conformal.json and regenerates the manifests.
Run: python scripts/deployed_check.py"""
import hashlib, json, sys
import os
from pathlib import Path
import numpy as np, pandas as pd
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
from evaluate_unified import context_table, SIX  # noqa: E402
from flare import load_classifier  # noqa: E402
from flare.context import lumdist_mpc  # noqa: E402
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", PKG / "paper_data"))  # python paper/setup_data.py

def check(clf, X, y):
    sets = clf.prediction_sets(X.reindex(columns=clf.feature_names))
    pred = clf.predict(X.reindex(columns=clf.feature_names))
    cov = np.array([SIX[t] in s for t, s in zip(y, sets)]); size = np.array([len(s) for s in sets])
    per = {c: dict(n=int((y == i).sum()), coverage=round(float(cov[y == i].mean()), 3),
                   err=round(float((cov[y == i].mean() * (1 - cov[y == i].mean()) / max((y == i).sum(), 1)) ** .5), 3))
           for i, c in enumerate(SIX)}
    return dict(split="test (1575 objects, untouched by fit and calibration)", marginal=round(float(cov.mean()), 4),
                singleton_rate=round(float((size == 1).mean()), 4), singleton_correct=round(float(((size == 1) & (np.array([SIX.index(p) for p in pred]) == y)).sum() / max((size == 1).sum(), 1)), 4),
                empty_rate=round(float((size == 0).mean()), 4), mean_set_size=round(float(size.mean()), 3), per_class=per)

def main():
    test = pd.read_csv(ROOT / "data/splits/test.csv"); y = np.array([SIX.index(c) for c in test.cls]); n0 = 7347 + 1575
    ctx = context_table(ROOT, test.obj_id.values)
    out = {}
    X5 = pd.read_parquet(ROOT / "results/reconstruction/cache/X_pool_h5.parquet")
    full = pd.read_parquet(ROOT / "data/features_all.parquet").set_index("obj_id").reindex(test.obj_id)
    X = X5.iloc[n0:n0 + len(test)].copy().reset_index(drop=True)
    for col in full.columns: X[col] = full[col].to_numpy()
    peak = 23.9 - 2.5 * full[["g_logflux_peak", "r_logflux_peak", "i_logflux_peak"]].max(axis=1).to_numpy()
    X["M_pseudo"] = peak - 5 * np.log10(lumdist_mpc(ctx.loc[test.obj_id].z_phot.to_numpy()) * 1e5)
    out["full"] = (PKG / "models/bts6", check(load_classifier("bts6"), X, y))
    for h in (5, 10, 20):
        Xh = pd.read_parquet(ROOT / f"results/reconstruction/cache/X_pool_h{h}.parquet").iloc[n0:n0 + len(test)].reset_index(drop=True)
        out[f"h{h}"] = (PKG / f"models/bts6/early/h{h}", check(load_classifier("bts6", horizon_days=h), Xh, y))
    for k, (d, r) in out.items():
        print(f"{k}: marginal {r['marginal']} singleton {r['singleton_rate']} (correct {r['singleton_correct']}) empty {r['empty_rate']} | " +
              " ".join(f"{c} {v['coverage']}±{v['err']}" for c, v in r["per_class"].items()), flush=True)
        c = json.loads((d / "conformal.json").read_text()); c["deployed_check"] = r; (d / "conformal.json").write_text(json.dumps(c, indent=2))
        if (d / "manifest.json").exists():
            (d / "manifest.json").write_text(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.iterdir()) if f.is_file() and f.name != "manifest.json"}, indent=2))
    print("DONE")

if __name__ == "__main__":
    main()
