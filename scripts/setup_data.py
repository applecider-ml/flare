"""Create the working directory that the training and export scripts read and write.

The scripts expect `data/` (benchmark tables, splits, light curves and the reference six-class
model), `results/` and `figures/`. This script builds it from `benchmark/` in this repository:

    python scripts/setup_data.py                 # creates ./paper_data
    export FLARE_PAPER_DATA=$PWD/paper_data          # the scripts default to this path anyway

The scripts read `FLARE_PAPER_DATA` (default: <repo>/paper_data).
"""
import os, shutil, sys, tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark"


def main():
    root = Path(os.environ.get("FLARE_PAPER_DATA", REPO / "paper_data")).resolve()
    data = root / "data"
    for d in (data, root / "results", root / "figures"):
        d.mkdir(parents=True, exist_ok=True)
    files = [p for p in BENCH.iterdir() if p.is_file() and p.suffix in {".csv", ".parquet"}]
    files += list((BENCH / "reference").iterdir())
    for p in files:
        dst = data / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
    if not (data / "splits").exists():
        shutil.copytree(BENCH / "splits", data / "splits")
    if not (data / "events").exists():
        with tarfile.open(BENCH / "light_curves.tar.gz") as tar:
            tar.extractall(data, filter="data") if sys.version_info >= (3, 12) else tar.extractall(data)
    n = len(list((data / "events").glob("*.npz")))
    print(f"FLARE_DATA = {root}\n  {len(files)} tables, splits, {n} light curves")


if __name__ == "__main__":
    main()
