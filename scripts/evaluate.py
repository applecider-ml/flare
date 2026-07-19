"""Evaluate FLARE on a labelled split: bootstrap CIs + conformal coverage.

Run:  python scripts/evaluate.py --data-dir /path/to/photo_events --split test
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flare import FlareClassifier, conformal                  # noqa: E402
from flare import data as D                                   # noqa: E402
from flare.metrics import bootstrap_metrics, format_metrics_table  # noqa: E402
from flare.taxonomy import ID2BROAD_ID                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--horizon", type=float, default=100.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.data_dir:
        D.DATA_DIR = Path(args.data_dir)

    clf = FlareClassifier.from_pretrained()
    df = D.load_manifest(args.split)
    kept, _ = D.filter_manifest_quality(df, horizon_days=args.horizon)
    y = np.array([ID2BROAD_ID[int(l)] for l in kept.label])
    print(f"{args.split}: N={len(kept)}", flush=True)

    X = clf.features_from_files(kept.filepath.tolist(), horizon_days=args.horizon)
    proba = clf.predict_proba(X)

    boot = bootstrap_metrics(y, proba, n_boot=args.n_boot, seed=0)
    print(f"\n## FLARE on the quality-filtered {args.split} split (N={len(y)})\n")
    print(format_metrics_table(boot))

    if clf.qhat is not None:
        sets = conformal.prediction_sets(proba, clf.qhat)
        rep = conformal.coverage_report(sets, y)
        print(f"\n## Conformal coverage (alpha={clf.alpha})\n")
        print("| Class | n | coverage | avg set size |")
        print("|---|---|---|---|")
        for k, v in rep.items():
            print(f"| {k} | {v['n']} | {v['coverage']:.3f} | {v['avg_set_size']:.2f} |")


if __name__ == "__main__":
    main()
