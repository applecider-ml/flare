"""Prepare public BTS prefix features and fit development-only trajectory archives.

First run scripts/setup_data.py with FLARE_PAPER_DATA set to --data-root.
This workflow never reads the test/anomaly split or requires pretrained models.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from evaluate_unified import SIX, context_table
from train_early_cc import hierarchy_nocal
from flare.analysis import feature_frame, observed_events
from flare.data import load_events
from flare.trajectory_benchmark import load_fold, sha, write_json

HORIZONS = (5, 10, 20, 100)
META = {'obj_id', 'split', 'cls', 'bts_type'}


def _feature_row(job):
    root, obj_id, filepath, context = job
    events = load_events(str(Path(root) / 'data' / filepath))
    rows = {}
    for horizon in HORIZONS:
        try:
            frame, _ = feature_frame(observed_events(events, float(horizon)), context)
        except ValueError as exc:
            raise ValueError(f'{obj_id} is ineligible at horizon {horizon}; no silent row dropping') from exc
        rows[horizon] = frame.iloc[0].to_dict()
    return rows


def build_features(data_root, out, *, workers=2):
    """Same prefix/context recipe as inference; train then val order is preserved."""
    root, out = Path(data_root), Path(out)
    if out.exists():
        raise ValueError('cache output already exists; choose a new directory')
    frames = []
    for split in ('train', 'val'):
        frame = pd.read_csv(root / 'data' / 'splits' / f'{split}.csv')
        frame['split'] = split
        frames.append(frame)
    objects = pd.concat(frames, ignore_index=True)
    if objects.obj_id.duplicated().any() or not objects.cls.isin(SIX).all():
        raise ValueError('development objects must have unique IDs and known six-class labels')
    context = context_table(root, objects.obj_id.to_numpy())
    jobs = [(str(root), row.obj_id, row.filepath,
             {key: None if pd.isna(value) else float(value) for key, value in context.loc[row.obj_id].items()})
            for row in objects.itertuples()]
    if workers == 1:
        rows = list(map(_feature_row, jobs))
    else:
        with Pool(workers) as pool:
            rows = pool.map(_feature_row, jobs, chunksize=20)
    out.mkdir(parents=True)
    meta = objects[['obj_id', 'split', 'cls', 'bts_type']].reset_index(drop=True)
    for horizon in HORIZONS:
        features = pd.DataFrame([row[horizon] for row in rows])
        frame = pd.concat([meta, features], axis=1)
        frame.to_parquet(out / f'X_h{horizon}.parquet', index=False)
        print(f'horizon={horizon}: {len(frame)} objects, {len(features.columns)} features', flush=True)
    write_json(out / 'manifest.json', dict(
        class_names=list(SIX), horizons=list(HORIZONS), n_development=len(objects),
        split_order=['train', 'val'], prefix_recipe='feature_frame(observed_events(load_events(path), horizon), context)',
        input_sha256={f'{name}.csv': sha(root / 'data' / f'{name}.csv') for name in
                      ('hosts', 'hosts_photoz', 'context_gaia_wise', 'ps1_position')},
        split_sha256={split: sha(root / 'data' / 'splits' / f'{split}.csv') for split in ('train', 'val')},
        feature_sha256={str(h): sha(out / f'X_h{h}.parquet') for h in HORIZONS}))


def fit_archives(cache, out, *, threads=2, rounds=2000, folds=tuple(range(5))):
    """Twenty original hierarchy fits; ES/calibration/evaluation predictions saved directly."""
    cache, out = Path(cache), Path(out)
    if out.exists():
        raise ValueError('training output already exists; choose a new directory')
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(5) for fold in folds):
        raise ValueError('folds must be unique indices in 0..4')
    data = {h: pd.read_parquet(cache / f'X_h{h}.parquet') for h in HORIZONS}
    master = data[100]
    if master.obj_id.duplicated().any():
        raise ValueError('duplicate development IDs')
    # Accept a reusable cache containing other roles, but never train/predict them.
    ids = master.loc[master.split.isin(['train', 'val']), 'obj_id'].to_numpy(dtype=str)
    features = [column for column in master if column not in META]
    for horizon, frame in data.items():
        dev = frame.loc[frame.split.isin(['train', 'val'])]
        if (frame.obj_id.duplicated().any() or set(dev.obj_id) != set(ids)
                or [column for column in frame if column not in META] != features):
            raise ValueError(f'horizon {horizon} eligibility or ordered feature schema differs')
        data[horizon] = dev.set_index('obj_id').loc[ids]
        if not np.array_equal(data[horizon].cls, master.set_index('obj_id').loc[ids].cls):
            raise ValueError('truth labels differ across horizons')
    y = np.array([SIX.index(name) for name in data[100].cls], dtype=int)
    split = list(KFold(5, shuffle=True, random_state=20261102).split(ids))
    fold_id = np.empty(len(ids), dtype=int)
    for fold, (_, indices) in enumerate(split):
        fold_id[indices] = fold
    out.mkdir(parents=True)
    pd.DataFrame(dict(obj_id=ids, fold=fold_id, cls=[SIX[i] for i in y])).to_csv(out / 'membership.csv', index=False)
    for fold in folds:
        evaluation = split[fold][1]
        calibration = np.flatnonzero(fold_id == (fold + 1) % 5)
        proper = np.flatnonzero((fold_id != fold) & (fold_id != (fold + 1) % 5))
        fit, early_stop = train_test_split(proper, test_size=.15, stratify=y[proper], random_state=100 + fold)
        role_indices = dict(training=early_stop, calibration=calibration, evaluation=evaluation)
        selected = np.concatenate(list(role_indices.values()))
        probabilities = np.empty((len(selected), len(HORIZONS), len(SIX)))
        nobs = np.empty((len(selected), len(HORIZONS)))
        model_dir = out / 'models' / f'fold{fold}'
        model_dir.mkdir(parents=True)
        metadata = {}
        for look, horizon in enumerate(HORIZONS):
            X = data[horizon][features].astype(float)
            model = hierarchy_nocal(X, y, fit, early_stop, SimpleNamespace(threads=threads, rounds=rounds))
            probabilities[:, look] = model.predict_proba(X.iloc[selected])
            nobs[:, look] = X.iloc[selected].n_obs_total
            model.top.save_model(str(model_dir / f'h{horizon}_top.txt'))
            model.branch.save_model(str(model_dir / f'h{horizon}_branch.txt'))
            metadata[str(horizon)] = dict(branch_threshold=float(model.branch_threshold), features=features,
                top_sha256=sha(model_dir / f'h{horizon}_top.txt'), branch_sha256=sha(model_dir / f'h{horizon}_branch.txt'))
            print(f'fold={fold} horizon={horizon}: models and predictions fitted', flush=True)
        arrays = dict(schema_version=np.array(1), class_names=np.array(SIX), horizons=np.array(HORIZONS),
                      training_score_role=np.array('classifier_early_stop'), classifier_fit_ids=ids[fit])
        start = 0
        for role, indices in role_indices.items():
            stop = start + len(indices)
            arrays.update({f'{role}_ids': ids[indices], f'{role}_probabilities': probabilities[start:stop],
                           f'{role}_nobs': nobs[start:stop], f'{role}_y': y[indices]})
            start = stop
        path = out / f'fold{fold}.npz'
        np.savez_compressed(path, **arrays)
        load_fold(path)
        write_json(model_dir / 'metadata.json', metadata)
    write_json(out / 'preparation.json', dict(class_names=list(SIX), horizons=list(HORIZONS),
        kfold_seed=20261102, early_stop_seeds=[100 + fold for fold in folds], rounds=rounds, threads=threads,
        classifier_recipe='train_early_cc.hierarchy_nocal; evaluate_unified.fit',
        training_score_role='classifier_early_stop; these scores were used for early stopping, not untouched OOF',
        test_or_anomaly_predictions=False,
        feature_sha256={str(h): sha(cache / f'X_h{h}.parquet') for h in HORIZONS},
        code_sha256={name: sha(REPO / 'scripts' / name) for name in
                     ('prepare_trajectory_benchmark.py', 'train_early_cc.py', 'evaluate_unified.py')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    features = sub.add_parser('features')
    features.add_argument('--data-root', type=Path, required=True)
    features.add_argument('--out', type=Path, required=True)
    features.add_argument('--workers', type=int, default=2)
    train = sub.add_parser('train')
    train.add_argument('--cache', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--threads', type=int, default=2)
    train.add_argument('--rounds', type=int, default=2000)
    train.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    args = parser.parse_args()
    if args.command == 'features':
        if args.workers < 1:
            parser.error('--workers must be positive')
        build_features(args.data_root, args.out, workers=args.workers)
    else:
        if args.threads < 1 or args.rounds < 1:
            parser.error('--threads and --rounds must be positive')
        fit_archives(args.cache, args.out, threads=args.threads, rounds=args.rounds, folds=tuple(args.folds))


if __name__ == '__main__':
    main()
