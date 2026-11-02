"""Run the complete exploratory benchmark or generate a small synthetic archive."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flare.trajectory_benchmark import run_benchmark, synthetic_archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    synthetic = sub.add_parser('synthetic', help='generate synthetic integration inputs')
    synthetic.add_argument('--out', type=Path, required=True)
    synthetic.add_argument('--objects', type=int, default=24)
    synthetic.add_argument('--seed', type=int, default=0)
    run = sub.add_parser('run', help='run all cells of the selected grids')
    run.add_argument('--input', type=Path, required=True)
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--rare-classes', nargs='+', required=True)
    run.add_argument('--mode', choices=['all', 'rare', 'budget'], default='all')
    args = parser.parse_args()
    if args.command == 'synthetic':
        if args.objects < 1:
            parser.error('--objects must be positive')
        synthetic_archives(args.out, n_objects=args.objects, seed=args.seed)
    else:
        run_benchmark(args.input, args.out, rare_classes=args.rare_classes, mode=args.mode)


if __name__ == '__main__':
    main()
