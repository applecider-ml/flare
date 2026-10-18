"""Audit final-fit partitions and replay legacy calibration from cached features.

Current prefix colors intentionally differ from the historical training cache.
Check other raw features, quantify changed colors, and label cached-feature
replay separately from current raw-input inference.
"""
import os
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import improve_supervised as api
sys.path.insert(0, str(api.PKG))
from flare.analysis import feature_frame, observed_events
from flare.config import BTS6
from flare.forecast_blend import BlendForecaster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--data', type=Path, default=Path(os.environ.get('FLARE_PAPER_DATA', api.PKG / 'paper_data')))
    a = parser.parse_args()
    df, _ = api.population(a.data)
    parts = np.load(a.directory / 'partitions.npz')
    train = set(parts['final_train'])
    cal = set(parts['calibration'])
    assert train == set(np.flatnonzero(df.split.to_numpy() == 'train'))
    assert cal == set(np.flatnonzero(df.split.to_numpy() == 'val'))
    assert not train & cal
    assert set(parts['round_fit']) | set(parts['round_stop']) == train
    assert not set(parts['round_fit']) & set(parts['round_stop'])
    model = BlendForecaster(a.directory, num_threads=2)
    for name in model.manifest:
        model.checked_path(name)
    photoz = pd.read_csv(a.data / 'data/hosts_photoz.csv').drop_duplicates('obj_id').set_index('obj_id')
    checks = []
    for h in [10, 20]:
        cached = pd.read_parquet(a.data / f'results/reconstruction/cache/X_pool_h{h}.parquet')
        points = pd.read_parquet(a.directory / f'h{h}_calibration.parquet')
        assert set(points.obj_id) <= set(df.obj_id.iloc[list(cal)])
        ids = parts['calibration'][:20]
        max_error, count, raw_checks, changed_colors = 0., 0, 0, 0
        color_columns = [c for c in cached if c.startswith(('g_r_', 'r_i_'))]
        unchanged_columns = [c for c in cached if c not in color_columns]
        for i in ids:
            rows = points[points.obj_id == df.obj_id[i]]
            if rows.empty:
                continue
            arr = observed_events(api.load_events(a.data / 'data' / df.filepath[i]), h)
            context = cached.iloc[i][BTS6.external_cols].to_dict()
            context['z_phot'] = photoz.loc[df.obj_id[i], 'z_phot'] if df.obj_id[i] in photoz.index else np.nan
            frame, peak = feature_frame(arr, context)
            np.testing.assert_allclose(frame.reindex(columns=unchanged_columns), cached.iloc[[i]][unchanged_columns],
                                       rtol=1e-4, atol=1e-4, equal_nan=True)
            changed_colors += int(not np.isclose(frame[color_columns].to_numpy(),
                cached.iloc[[i]][color_columns].to_numpy(), rtol=1e-4, atol=1e-4, equal_nan=True).all())
            raw_checks += 1
            # Cached columns isolate inference semantics from float32 cache rounding.
            times = np.unique(rows.day)
            pred = model.predict(arr, cached.iloc[[i]], peak, h, days=times)
            for b, label in enumerate(['g', 'r']):
                r = rows[rows.band == b]
                if r.empty:
                    continue
                band = pred['bands'][label]
                values = np.array(band['magnitude'])[1:][np.searchsorted(times, r.day)] - peak
                difference = np.abs(values - r.blend.to_numpy())
                # Historical last-magnitude cache used float32 arithmetic.
                np.testing.assert_allclose(values, r.blend, rtol=0, atol=3e-5)
                max_error = max(max_error, float(difference.max()))
                count += len(r)
        checks.append(dict(horizon=h, noncolor_raw_feature_objects=raw_checks,
                           corrected_color_objects=changed_colors, cached_feature_replayed_points=count,
                           max_replay_rounding_error_mag=max_error))
    report = dict(final_fit_original_train_only=True, original_val_calibration_only=True,
                  original_test_unused_in_final_fit=True, manifests_verified=True,
                  original_population_used_in_development=True,
                  note='Cached-feature replay validates historical weights, not current raw-prefix predictions. Colors are corrected at inference; legacy training/calibration are unchanged.',
                  checks=checks)
    (a.directory / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
