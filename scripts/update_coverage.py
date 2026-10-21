"""Measured coverage and out-of-fold performance for the shipped models, written where the API reads
them (conformal.json: empirical_coverage, marginal_coverage; model_card.json: performance_note).

Uses the out-of-fold probabilities that export_venn_abers.py saves for the five-fold cross-fit over
train+val (oof_full.npz, oof_5/10/20.npz): leave-one-fold-out cross-conformal coverage at alpha 0.1 per
class, and the out-of-fold accuracy and macro F1. Replaces the constants that train_bts6.py carries
from earlier evaluations. Manifests are regenerated.
Run: python scripts/update_coverage.py --models models/bts6 --oof-dir <dir>
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
from evaluate_unified import SIX, quantile  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--oof-dir", type=Path, required=True); a = ap.parse_args()
    for hs in ("full", "5", "10", "20"):
        z = np.load(a.oof_dir / f"oof_{hs}.npz"); P, y = z["P"], z["y"]; n = len(y)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
        fold_id = np.empty(n, int)
        for k, (_, te) in enumerate(folds):
            fold_id[te] = k
        cov = np.zeros(n, bool); size = np.zeros(n, int)
        for k in range(5):
            q = np.array([quantile([1 - P[j, i] for j in np.where((fold_id != k) & (y == i))[0]], .9) for i in range(6)])
            mem = (1 - P[fold_id == k]) <= q[None, :]
            cov[fold_id == k] = mem[np.arange(mem.shape[0]), y[fold_id == k]]; size[fold_id == k] = mem.sum(1)
        pred = P.argmax(1)
        per = {c: round(float(cov[y == i].mean()), 3) for i, c in enumerate(SIX)}
        f1 = f1_score(y, pred, average=None, labels=range(6))
        out = a.models if hs == "full" else a.models / "early" / f"h{hs}"
        conf = json.loads((out / "conformal.json").read_text())
        conf.update(empirical_coverage=per, marginal_coverage=round(float(cov.mean()), 4),
                    empirical_coverage_note=(f"leave-one-fold-out cross-conformal coverage at alpha 0.1 over five stratified folds "
                                             f"of train+val ({n} objects), measured by scripts/update_coverage.py"))
        (out / "conformal.json").write_text(json.dumps(conf, indent=2, allow_nan=False))
        card = json.loads((out / "model_card.json").read_text())
        card["performance_note"] = (f"five-fold out-of-fold on train+val ({n} objects), argmax of the out-of-fold probabilities: accuracy {np.mean(pred == y):.3f}, macro F1 "
                                    f"{f1_score(y, pred, average='macro'):.3f}; per-class F1 " + ", ".join(f"{c} {v:.3f}" for c, v in zip(SIX, f1)))
        (out / "model_card.json").write_text(json.dumps(card, indent=2))
        man_path = out / "manifest.json"; man = json.loads(man_path.read_text())
        for f in ("conformal.json", "model_card.json"):
            man[f] = hashlib.sha256((out / f).read_bytes()).hexdigest()
        man_path.write_text(json.dumps(dict(sorted(man.items())), indent=2))
        print(f"{hs}: marginal coverage {cov.mean():.3f}, per class {per}, macro F1 {f1_score(y, pred, average='macro'):.3f}", flush=True)


if __name__ == "__main__":
    main()
