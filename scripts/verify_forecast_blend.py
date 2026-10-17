"""Replay saved fixed-blend OOF predictions and audit object partitions."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import improve_supervised as api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    provenance = json.loads((root / 'provenance.json').read_text())
    complete = json.loads((root / 'complete.json').read_text())
    assert api.sha(Path(__file__).with_name('forecast_blend.py')) == complete['script_sha256']
    assert api.sha(api.__file__) == provenance['helper_sha256']
    source = Path(provenance['arguments']['source'])
    assert api.sha(source / 'recency.npz') == provenance['recency_sha256']
    data = Path(provenance['arguments']['data'])
    df, _ = api.population(data)
    checks = []
    for h in [10, 20]:
        x = pd.read_parquet(data / f'results/reconstruction/cache/X_pool_h{h}.parquet').replace([np.inf, -np.inf], np.nan)
        cached = np.load(data / f'results/reconstruction/cache/TL_pool_h{h}.npz', allow_pickle=True)
        state = np.load(source / 'recency.npz')[f'h{h}']
        seen = []
        for fold in range(5):
            out = root / f'forecast_h{h}_fold{fold}'
            parts = np.load(out / 'partitions.npz')
            sets = [set(parts[k]) for k in ['fit', 'early_stop', 'calibration', 'test']]
            assert all(not sets[i] & sets[j] for i in range(4) for j in range(i))
            assert set.union(*sets) == set(range(len(df)))
            seen.extend(parts['test'])
            rows = pd.read_parquet(out / 'points.parquet')
            tables = {m: rows[rows.method == m].reset_index(drop=True)
                      for m in ['baseline', 'real_delta', 'blend']}
            ref = tables['baseline']
            for table in tables.values():
                pd.testing.assert_frame_equal(table[['obj_id', 'day', 'band', 'truth']], ref[['obj_id', 'day', 'band', 'truth']])
                assert set(table.obj_id) <= set(df.obj_id.iloc[parts['test']])
                assert np.isfinite(table[['truth', 'prediction', 'error', 'halfwidth']]).all().all()
                assert (table.day > h).all() and (table.day <= 90).all()
                np.testing.assert_allclose(table.error, abs(table.prediction - table.truth))
            np.testing.assert_allclose(tables['blend'].prediction,
                                       (ref.prediction + tables['real_delta'].prediction) / 2,
                                       rtol=0, atol=1e-12)
            ids = parts['test'][:10]
            baseline = lgb.Booster(model_file=str(out / 'baseline.txt'))
            delta = lgb.Booster(model_file=str(out / 'real_delta.txt'))
            table, truth, oi, bi, pb = api.anchored(baseline, x, ids, h, cached['FP'], cached['L'], 2)
            pr = delta.predict(api.add_recency(table, oi, bi, state), num_threads=2) + cached['L'][oi, bi]
            # real_long preserves object input order; saved rows use the same order.
            for method, values in [('baseline', pb), ('real_delta', pr), ('blend', (pb + pr) / 2)]:
                saved = tables[method][tables[method].obj_id.isin(df.obj_id.iloc[ids])]
                np.testing.assert_array_equal(saved.obj_id, df.obj_id.iloc[oi])
                np.testing.assert_allclose(saved.truth, truth, rtol=0, atol=1e-12)
                np.testing.assert_allclose(saved.prediction, values, rtol=1e-10, atol=1e-10)
            checks.append(dict(horizon=h, fold=fold, replayed_points=int(len(truth)), methods=3))
        assert sorted(seen) == list(range(len(df)))
    report = dict(registered_sources_unchanged=True, disjoint_partitions=True,
                  exactly_one_test_fold_per_object=True, common_evaluation_points=True,
                  fixed_blend_verified_all_points=True, model_replays=checks)
    (root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
