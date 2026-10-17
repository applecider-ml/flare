"""Optional fixed grid/real-detection blend at days 10 and 20."""
import hashlib
import json
from pathlib import Path

import numpy as np


RECENCY_COLUMNS = ['fc_last_mag', 'fc_age', 'fc_slope_last5', 'fc_scatter_last5']


def recent_state(events, horizon):
    """Band-local state; exact timestamp ties follow the training anchor rule."""
    a = np.asarray(events, dtype=float)
    a = a[(a[:, 0] >= 0) & (a[:, 0] <= horizon) & np.isfinite(a[:, [0, 2, 3]]).all(1)]
    state = np.full((2, 4), np.nan)
    if not len(a):
        return state
    peak = 23.9 - 2.5 * a[:, 3].max()
    for band in [0, 1]:
        v = a[a[:, 2] == band]
        v = v[np.argsort(v[:, 0], kind='stable')]
        if not len(v):
            continue
        v = v[np.r_[True, np.diff(v[:, 0]) > 0]]
        t = v[-5:, 0]
        mag = 23.9 - 2.5 * v[-5:, 3] - peak
        slope = float(np.polyfit(t - t[-1], mag, 1)[0]) if len(t) >= 2 else np.nan
        state[band] = [mag[-1], horizon - t[-1], slope, float(mag.std())]
    return state


class BlendForecaster:
    def __init__(self, directory, num_threads=2):
        self.directory = Path(directory)
        self.num_threads = num_threads
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        if manifest['algorithm'] != 'sha256':
            raise ValueError('unsupported blend manifest hash')
        self.manifest = manifest['files']
        self.models = {}

    def checked_path(self, name):
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError('model path escapes blend directory')
        if name not in self.manifest or hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest[name]:
            raise ValueError(f'blend model manifest mismatch: {name}')
        return path

    def predict(self, events, frame, peak, horizon, days=None):
        import lightgbm as lgb
        import pandas as pd

        if horizon not in (10, 20):
            raise ValueError('blend forecast horizon must be 10 or 20')
        meta = json.loads(self.checked_path(f'h{int(horizon)}.json').read_text())
        grid = np.asarray(meta['grid_days'], dtype=float)
        requested = grid if days is None else np.asarray(days, dtype=float)
        if (requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all()
                or np.any(requested <= horizon) or np.any(requested > 90)
                or np.any(np.diff(requested) <= 0)):
            raise ValueError('forecast days must be a nonempty, increasing sequence in (horizon, 90]')
        x = frame.reindex(columns=meta['feature_names'])
        state = recent_state(events, horizon)
        models = {}
        for name, cfg in meta['models'].items():
            filename = cfg['file']
            if filename not in self.models:
                self.models[filename] = lgb.Booster(model_file=str(self.checked_path(filename)))
            model = self.models[filename]
            if model.feature_name() != cfg['features']:
                raise ValueError(f'blend feature order mismatch: {filename}')
            models[name] = model
        bands = {}
        for band, label in enumerate(['g', 'r']):
            if not np.isfinite(state[band, 0]):
                bands[label] = dict(status='insufficient_data', reason='no detection to anchor this band')
                continue
            base_x = pd.concat([x.assign(fc_day=d, fc_band=float(band)) for d in grid], ignore_index=True)
            base_grid = models['baseline'].predict(base_x, num_threads=self.num_threads)
            baseline = np.interp(requested, np.r_[horizon, grid], np.r_[state[band, 0], base_grid])
            real_x = pd.concat([x.assign(fc_day=d, fc_band=float(band)) for d in requested], ignore_index=True)
            for col, value in zip(RECENCY_COLUMNS, state[band]):
                real_x[col] = value
            real = models['real_delta'].predict(real_x, num_threads=self.num_threads) + state[band, 0]
            mean = np.r_[state[band, 0], (baseline + real) / 2] + peak
            # The first element is the observed anchor, not a future prediction.
            hw68 = np.r_[0., np.full(len(requested), meta['residual_halfwidth']['0.68'])]
            hw95 = np.r_[0., np.full(len(requested), meta['residual_halfwidth']['0.95'])]
            bands[label] = dict(status='ok', days=np.r_[horizon, requested], magnitude=mean,
                lower68=mean-hw68, upper68=mean+hw68, lower95=mean-hw95, upper95=mean+hw95)
        return dict(status='ok' if any(v['status'] == 'ok' for v in bands.values()) else 'insufficient_data',
            model_horizon_days=horizon, observation_cutoff_days=horizon, recipe='blend',
            model_role='fixed_50_50_blend', reference_magnitude=peak, bands=bands,
            units='AB mag', time_origin='first detection', calibration=meta['calibration']['note'],
            development_scope=meta['development']['note'],
            anchor_note='Horizon entry is the last observed band magnitude, shown without a forecast interval.',
            extrapolation='none beyond day 90')
