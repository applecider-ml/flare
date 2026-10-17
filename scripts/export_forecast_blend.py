"""Final fit of the fixed blend, allowed only after all registered gates pass.

Original train objects only: select rounds on a fixed 85/15 train carve-out,
then refit all original train objects at those rounds. Original validation
objects calibrate pointwise residual widths; original test objects are unused.
No further hyperparameter or mixture selection. This is a final fit of a
development-selected recipe, not independent confirmation of the OOF gains.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import improve_supervised as api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    a = parser.parse_args()
    result = json.loads((a.evaluation / 'summary.json').read_text())
    verification = json.loads((a.evaluation / 'verification.json').read_text())
    if not result['adopt'] or not verification['fixed_blend_verified_all_points']:
        raise ValueError('The fixed blend must pass its registered gates and verification')
    provenance = json.loads((a.evaluation / 'provenance.json').read_text())
    assert api.sha(api.__file__) == provenance['helper_sha256']
    data = Path(provenance['arguments']['data'])
    source = Path(provenance['arguments']['source'])
    df, y = api.population(data)
    train = np.flatnonzero(df.split.to_numpy() == 'train')
    cal = np.flatnonzero(df.split.to_numpy() == 'val')
    fit, stop = train_test_split(train, test_size=.15, stratify=y[train], random_state=2026)
    a.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.out / 'partitions.npz', obj_id=df.obj_id.to_numpy(dtype=str),
                        final_train=train, round_fit=fit, round_stop=stop, calibration=cal)
    params = json.loads((data / 'results/reconstruction/final_choice.json').read_text())['forecaster']
    for h in [10, 20]:
        x = pd.read_parquet(data / f'results/reconstruction/cache/X_pool_h{h}.parquet').replace([np.inf, -np.inf], np.nan)
        cached = np.load(data / f'results/reconstruction/cache/TL_pool_h{h}.npz', allow_pickle=True)
        state = np.load(source / 'recency.npz')[f'h{h}']
        last, points = cached['L'], cached['FP']
        models, details = {}, {}

        def table(ids, method):
            if method == 'baseline':
                xx, yy, _, _ = api.grid_long(x, cached['T'], ids, h)
                return xx, yy, None
            xx, yy, oi, bi = api.real_long(x, points, last, ids, h)
            return api.add_recency(xx, oi, bi, state), yy - last[oi, bi], api.object_weights(oi)

        for method in ['baseline', 'real_delta']:
            xt, yt, wt = table(fit, method)
            xe, ye, we = table(stop, method)
            selected = api.fit(xt, yt, xe, ye, params, a.threads, wt, we, rounds=4000, patience=100)
            rounds = selected.best_iteration
            del xt, xe, selected
            xx, yy, ww = table(train, method)
            keep = np.isfinite(yy)
            dataset = lgb.Dataset(xx.loc[keep], label=yy[keep], weight=None if ww is None else ww[keep])
            model = lgb.train(dict(params, num_threads=a.threads, verbosity=-1), dataset, num_boost_round=rounds)
            filename = f'h{h}_{method}.txt'
            model.save_model(str(a.out / filename))
            models[method] = model
            details[method] = dict(file=filename, rounds=rounds, features=list(xx.columns))
            print(f'final h{h} {method}: {rounds} rounds on {len(train)} train objects', flush=True)
            del xx, dataset
        xx, truth, oi, bi, pb = api.anchored(models['baseline'], x, cal, h, points, last, a.threads)
        pr = models['real_delta'].predict(api.add_recency(xx, oi, bi, state), num_threads=a.threads) + last[oi, bi]
        residuals = np.abs((pb + pr) / 2 - truth)
        widths = {}
        for level in [.68, .95]:
            ordered = np.sort(residuals)
            index = min(int(np.ceil((len(ordered) + 1) * level)), len(ordered)) - 1
            widths[str(level)] = float(ordered[index])
        calibration = pd.DataFrame(dict(obj_id=df.obj_id.iloc[oi].to_numpy(), cls=y[oi],
            day=xx.fc_day, band=bi, truth=truth, baseline=pb, real_delta=pr, blend=(pb+pr)/2, error=residuals))
        calibration.to_parquet(a.out / f'h{h}_calibration.parquet', index=False)
        meta = dict(horizon_days=h, feature_names=list(x.columns),
            grid_days=[d for d in api.DAYS if d > h], models=details, blend_weight=.5,
            residual_halfwidth=widths,
            calibration=dict(objects=int(len(np.unique(oi))), points=len(oi),
                note='Empirical pointwise residual bands on original validation detections; '
                     'no class-conditional or simultaneous light-curve coverage guarantee. '
                     'Validation population was reused in development; not an external calibration test.'),
            training=dict(objects=len(train), rounds_selected_within_train=True,
                          population='Original BTS train only; validation for residual widths; test unused in final fit'),
            development=dict(evaluation_sha256=api.sha(a.evaluation/'summary.json'),
                             comparison=result['comparisons'][str(h)],
                             note='Same historically inspected BTS population; not external validation.'))
        (a.out / f'h{h}.json').write_text(json.dumps(meta, indent=2) + '\n')
    manifest = {p.name: api.sha(p) for p in sorted(a.out.iterdir()) if p.suffix in ['.json', '.txt']}
    (a.out / 'manifest.json').write_text(json.dumps(dict(algorithm='sha256', files=manifest,
        export_script_sha256=api.sha(__file__), helper_sha256=api.sha(api.__file__),
        evaluation_provenance=provenance), indent=2) + '\n')


if __name__ == '__main__':
    main()
