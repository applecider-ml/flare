"""Classify new light curves with a pretrained FLARE model.

Input is either a directory of per-object .npz files, an explicit list of .npz
paths, or a manifest CSV with a `filepath` column. Outputs a CSV with the
predicted class, per-class probabilities, and the conformal prediction set.

Run:
    python scripts/predict.py --npz-dir /path/to/objects --out preds.csv
    python scripts/predict.py --manifest manifest_test.csv --out preds.csv
    python scripts/predict.py obj1.npz obj2.npz
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flare import FlareClassifier, BROAD_CLASSES              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="explicit .npz paths")
    ap.add_argument("--npz-dir", default=None)
    ap.add_argument("--manifest", default=None,
                    help="CSV with a filepath column (optional obj_id)")
    ap.add_argument("--data-dir", default=None,
                    help="prepend to relative filepaths in the manifest")
    ap.add_argument("--model", default=None)
    ap.add_argument("--conformal", default=None)
    ap.add_argument("--horizon", type=float, default=100.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.npz_dir:
        paths = sorted(str(p) for p in Path(args.npz_dir).glob("*.npz"))
        obj_ids = [Path(p).stem for p in paths]
    elif args.manifest:
        mf = pd.read_csv(args.manifest)
        fps = mf["filepath"].astype(str)
        if args.data_dir:
            fps = fps.map(lambda p: str(Path(args.data_dir) / p))
        paths = list(fps)
        obj_ids = list(mf["obj_id"]) if "obj_id" in mf.columns else \
            [Path(p).stem for p in paths]
    else:
        paths = args.files
        obj_ids = [Path(p).stem for p in paths]
    if not paths:
        ap.error("no input: give .npz paths, --npz-dir, or --manifest")

    kw = {}
    if args.model:
        kw["model_path"] = args.model
    if args.conformal:
        kw["conformal_path"] = args.conformal
    clf = FlareClassifier.from_pretrained(**kw)

    X = clf.features_from_files(paths, horizon_days=args.horizon)
    proba = clf.predict_proba(X)
    sets = (clf.prediction_sets(X) if clf.qhat is not None
            else [[] for _ in paths])

    out = pd.DataFrame({"obj_id": obj_ids})
    out["pred"] = [BROAD_CLASSES[i] for i in proba.argmax(1)]
    for c, name in enumerate(BROAD_CLASSES):
        out[f"p_{name}"] = proba[:, c]
    out["conformal_set"] = ["|".join(s) for s in sets]

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"{len(out)} objects -> {args.out}")
    else:
        with pd.option_context("display.width", 200,
                               "display.max_columns", 20):
            print(out.to_string(index=False))


if __name__ == "__main__":
    main()
