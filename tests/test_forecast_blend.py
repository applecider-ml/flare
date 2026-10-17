"""Causal inference, artifact integrity and arbitrary-time blend replay."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from flare.analysis import FlareAnalysis, feature_frame, observed_events
from flare.config import PKG_ROOT
from flare.forecast_blend import BlendForecaster, recent_state
from test_analysis import events

sys.path.insert(0, str(PKG_ROOT / 'scripts'))
import improve_supervised as research


class ForecastBlendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FlareAnalysis(num_threads=1)
        cls.folder = PKG_ROOT / 'models/bts6/forecast_blend'

    def test_production_recency_matches_registered_recipe(self):
        a = events()
        a = np.vstack([a, a[8]])
        a[-1, 3] += .25
        for h in [10, 20]:
            np.testing.assert_allclose(recent_state(a, h), research.recency(a, h), equal_nan=True)

    def test_future_changes_do_not_change_predictions(self):
        original = events()
        changed = original.copy()
        changed[changed[:, 0] > 10, 3:] = 900.
        a = self.engine.forecast(original, 10, recipe='blend', days=[10.5, 32.2, 90.])
        b = self.engine.forecast(changed, 10, recipe='blend', days=[10.5, 32.2, 90.])
        self.assertEqual(a, b)
        json.dumps(a, allow_nan=False)

    def test_arbitrary_times_match_registered_anchoring_and_blend(self):
        a = observed_events(events(), 10)
        x, peak = feature_frame(a)
        meta = json.loads((self.folder / 'h10.json').read_text())
        x = x.reindex(columns=meta['feature_names'])
        state = research.recency(a, 10)[None, :, :]
        times = np.array([10.2, 17.3, 50., 89.9])
        import lightgbm as lgb
        baseline = lgb.Booster(model_file=str(self.folder / meta['models']['baseline']['file']))
        delta = lgb.Booster(model_file=str(self.folder / meta['models']['real_delta']['file']))
        points = [np.column_stack([np.tile(times, 2), np.repeat([0, 1], len(times)), np.zeros(2*len(times))])]
        table, _, oi, bi, pb = research.anchored(baseline, x, [0], 10, points, state[:, :, 0], 1)
        pr = delta.predict(research.add_recency(table, oi, bi, state), num_threads=1) + state[oi, bi, 0]
        result = self.engine.forecast(a, 10, recipe='blend', days=times)
        for b, label in enumerate(['g', 'r']):
            band = result['bands'][label]
            np.testing.assert_allclose(band['magnitude'][1:], ((pb+pr)/2+peak)[bi == b], atol=1e-10)
            self.assertEqual(band['days'], [10.] + times.tolist())
            self.assertTrue(np.all(np.array(band['lower95']) <= band['lower68']))

    def test_missing_band_and_invalid_requests(self):
        a = events()
        r = self.engine.forecast(a[a[:, 2] == 0], 10, recipe='blend')
        self.assertEqual(r['bands']['r']['status'], 'insufficient_data')
        for days in [[10], [91], [], [20, 15], [20, 20], [np.nan]]:
            with self.assertRaises(ValueError):
                self.engine.forecast(a, 10, recipe='blend', days=days)
        with self.assertRaises(ValueError):
            self.engine.forecast(a, 5, recipe='blend')
        with self.assertRaises(ValueError):
            self.engine.forecast(a, 10, recipe='unknown')

    def test_manifest_rejects_modified_artifact_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'h10.json').write_text('{}')
            (folder / 'manifest.json').write_text(json.dumps(dict(algorithm='sha256', files={'h10.json': 'wrong'})))
            model = BlendForecaster(folder)
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                model.checked_path('h10.json')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                model.checked_path('../outside.txt')

    def test_unified_analysis_selects_blend_and_renders(self):
        from flare.analysis_report import render_analysis
        result = self.engine.analyze(events(), horizon_days=20, forecast_horizon=10,
                                     forecast_recipe='blend', include_reconstruction=False)
        self.assertEqual(result['forecast']['recipe'], 'blend')
        self.assertIn('Empirical pointwise residual bands', render_analysis({'example': result}))


if __name__ == '__main__':
    unittest.main()
