"""Venn-Abers tables for the shipped classifiers (full-curve and day 5/10/20).

Re-runs the five-fold cross-fit of scripts/train_full_cc.py and scripts/train_early_cc.py
(same pool: train+val, same folds and early-stopping splits; the test split is not read), keeps
the out-of-fold probabilities, and writes for every model a `venn_abers.json` with one step
table per class (flare.venn_abers.build_table). Manifests are regenerated.
The out-of-fold probabilities are also saved (`--oof-dir`) so the tables can be rebuilt quickly.
Run: python scripts/export_venn_abers.py --models <repo>/models/bts6 --threads 16
"""
import argparse, hashlib, json, sys
import os
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
from evaluate_unified import context_table, SIX  # noqa: E402
from train_early_cc import hierarchy_nocal  # noqa: E402
from flare.context import lumdist_mpc  # noqa: E402
from flare.venn_abers import build_table  # noqa: E402
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", PKG / "paper_data"))  # python scripts/setup_data.py


def pool_X(h, pool, n):
    if h is None:
        full = pd.read_parquet(ROOT / "data/features_all.parquet").set_index("obj_id").reindex(pool.obj_id)
        X = pd.read_parquet(ROOT / "results/reconstruction/cache/X_pool_h5.parquet").iloc[:n].copy()
        for col in full.columns:
            X[col] = full[col].to_numpy()
        ctx = context_table(ROOT, pool.obj_id.values)
        peak = 23.9 - 2.5 * full[["g_logflux_peak", "r_logflux_peak", "i_logflux_peak"]].max(axis=1).to_numpy()
        X["M_pseudo"] = peak - 5 * np.log10(lumdist_mpc(ctx.loc[pool.obj_id].z_phot.to_numpy()) * 1e5)
        return X
    return pd.read_parquet(ROOT / f"results/reconstruction/cache/X_pool_h{h}.parquet").iloc[:n].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--oof-dir", type=Path, default=ROOT / "results/venn_abers_export")
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--horizons", nargs="+", default=["full", "5", "10", "20"])
    a = ap.parse_args(); a.oof_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/splits/train.csv"); val = pd.read_csv(ROOT / "data/splits/val.csv")
    pool = pd.concat([train, val], ignore_index=True); n = len(pool)
    y = np.array([SIX.index(c) for c in pool.cls])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    for hs in a.horizons:
        h = None if hs == "full" else int(hs)
        f = a.oof_dir / f"oof_{hs}.npz"
        if f.exists():
            P = np.load(f)["P"]
        else:
            X = pool_X(h, pool, n); P = np.full((n, 6), np.nan)
            for k, (tr0, te) in enumerate(folds):
                tr, es = train_test_split(tr0, test_size=.15, stratify=y[tr0], random_state=k)
                P[te] = hierarchy_nocal(X, y, tr, es, a).predict_proba(X.iloc[te])
                print(f"{hs} fold {k}: acc {np.mean(P[te].argmax(1) == y[te]):.3f}", flush=True)
            np.savez(f, P=P, y=y)
        tables = Parallel(n_jobs=min(6, a.threads))(delayed(build_table)(P[:, i], y == i) for i in range(6))
        out = a.models if h is None else a.models / "early" / f"h{h}"
        (out / "venn_abers.json").write_text(json.dumps(dict(
            method="inductive Venn-Abers, one class against the rest, on the out-of-fold probabilities of the "
                   "five-fold cross-fit over train+val (validity checked empirically, not a finite-sample guarantee)",
            n_cal=n, classes={c: t for c, t in zip(SIX, tables)})))
        man_path = out / "manifest.json"; man = json.loads(man_path.read_text())
        man["venn_abers.json"] = hashlib.sha256((out / "venn_abers.json").read_bytes()).hexdigest()
        man_path.write_text(json.dumps(dict(sorted(man.items())), indent=2))
        print(f"{hs}: wrote {out / 'venn_abers.json'} ({(out / 'venn_abers.json').stat().st_size / 1e3:.0f} kB; "
              f"segments {[len(t['p0']) for t in tables]})", flush=True)


if __name__ == "__main__":
    main()
