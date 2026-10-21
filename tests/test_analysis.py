"""Regression checks against shipped models and causal inference boundaries.

Run with: python -m unittest discover -s tests -v
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from flare.analysis import FlareAnalysis, feature_frame, observed_events, reconstruct
from flare.analysis_report import render_analysis
from flare.config import PKG_ROOT


def events():
    a = np.zeros((12, 15))
    a[:, 0] = np.repeat([0., 2., 4., 7., 10., 15.], 2)
    a[:, 1] = np.r_[0, np.diff(a[:, 0])]
    a[:, 2] = np.tile([0, 1], 6)
    a[:, 3] = 1.7 + .04*a[:, 0] - .0008*a[:, 0]**2 + .1*a[:, 2]
    a[:, 4] = .025
    a[np.arange(len(a)), 5+a[:, 2].astype(int)] = 1.
    return a


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FlareAnalysis(num_threads=1)

    def test_manifest(self):
        # every file the shipped manifest lists is present and matches its hash (35 in the push set: the
        # pooled_w role points at pooled and the w075 alternative is a release asset, not in git)
        n = len(json.loads((self.engine.model_dir / 'forecast_pe_manifest.json').read_text()))
        self.assertEqual(self.engine.verify_models()['files'], n)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'forecast').mkdir()
            (root/'forecast/h5.json').write_text('{}')
            (root/'forecast_pe_manifest.json').write_text(json.dumps({'forecast/h5.json': 'incorrect'}))
            other = FlareAnalysis(classifier=self.engine.classifier, model_dir=root)
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                other._metadata('forecast', 5)

    def test_future_detections_cannot_change_forecast_or_pe(self):
        original = events()
        changed = original.copy()
        changed[changed[:, 0] > 5, 3] = 99.
        for method in (self.engine.forecast, self.engine.pe):
            self.assertEqual(method(original, 5), method(changed, 5))

    def test_context_cannot_override_light_curve_and_peak_is_recomputed(self):
        a = observed_events(events(), 5)
        x, peak = feature_frame(a, {'n_obs_total': 999, 'M_pseudo': -999, 'z_phot': .1})
        from flare.context import m_pseudo
        self.assertEqual(x.n_obs_total.iloc[0], len(a))
        self.assertAlmostEqual(x.M_pseudo.iloc[0], m_pseudo(peak, .1))
        x, _ = feature_frame(a, {'M_pseudo': -999})
        self.assertTrue(np.isnan(x.M_pseudo.iloc[0]))

    def test_full_curve_color_cache_cannot_leak_future_counterpart(self):
        from flare.fetch import to_events
        rows = [(60000., 1, 18., .1), (60001., 2, 18.2, .1),
                (60009.5, 1, 17., .1), (60010.5, 2, 16., .1)]
        changed = rows[:-1] + [(60010.5, 2, 23., .1)]
        # Rebuilding the full array changes a PRE-cutoff cached color.
        full = to_events(rows)
        future_changed = to_events(changed)
        self.assertNotEqual(full[2, 8], future_changed[2, 8])
        x, _ = feature_frame(observed_events(full, 10))
        y, _ = feature_frame(observed_events(future_changed, 10))
        early, _ = feature_frame(observed_events(to_events(rows[:-1]), 10))
        pd.testing.assert_frame_equal(x, y)
        pd.testing.assert_frame_equal(x, early)

    def test_low_level_features_ignore_supplied_color_values_and_flags(self):
        from flare.features import extract_from_array
        arr = observed_events(events(), 10)
        changed = arr.copy()
        changed[:, 8:14] = 999.
        before = pd.DataFrame([extract_from_array(arr)])
        after = pd.DataFrame([extract_from_array(changed)])
        pd.testing.assert_frame_equal(before, after)

    def test_saved_feature_order_and_forecast_match_direct_booster(self):
        import lightgbm as lgb
        a = observed_events(events(), 5)
        x, peak = feature_frame(a)
        meta = json.loads((PKG_ROOT/'models/bts6/forecast/h5.json').read_text())
        result = self.engine.forecast(a, 5)
        model = lgb.Booster(model_file=str(PKG_ROOT/'models/bts6/forecast'/meta['models'][result['model_role']]))
        row = x.reindex(columns=meta['feature_names']).assign(fc_day=50., fc_band=0.)
        expected = model.predict(row, num_threads=1)[0] + peak
        band = result['bands']['g']
        self.assertAlmostEqual(band['magnitude'][band['days'].index(50.)], expected, places=10)
        self.assertTrue(np.all(np.array(band['lower95']) <= np.array(band['lower68'])))

    def test_invalid_horizons_and_prior(self):
        for h in [0, -1, np.nan, 101]:
            with self.assertRaises(ValueError): observed_events(events(), h)
        with self.assertRaises(ValueError): self.engine.forecast(events(), 7)
        with self.assertRaises(ValueError):
            self.engine.analyze(events(), 5, forecast_horizon=10, include_reconstruction=False)
        x, _ = feature_frame(observed_events(events(), 5))
        x = x.reindex(columns=self.engine.classifier.feature_names)
        for p in [0, 1, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError): self.engine.classifier.anomaly_probability(x, p)

    def test_gate_abstains_and_results_are_strict_json(self):
        with patch.object(self.engine, '_classifier_at', return_value=self.engine.classifier), patch.object(self.engine.classifier, 'conformal_report', return_value=[{'predicted': 'SN_Ia', 'set': ['SN_Ia', 'SN_CC']}]):
            r = self.engine.analyze(events(), 10, pe_horizon=5, include_reconstruction=False)
        self.assertEqual(r['pe']['status'], 'abstained')
        json.dumps(r, allow_nan=False)
        html = render_analysis({'<script>bad</script>': r})
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>bad</script>', html)
        self.assertIn('alarm calibration at this observation cutoff unvalidated', html)

    def test_matching_early_classifier_is_selected(self):
        result = self.engine.analyze(events(), 10, include_reconstruction=False)
        self.assertEqual(result['classification']['model_horizon_days'], 10)
        self.assertEqual(result['classification']['guaranteed_error'], .1)
        result = self.engine.analyze(events(), 7, include_reconstruction=False)
        self.assertIsNone(result['classification']['guaranteed_error'])

    def test_gp_only_interpolates_and_sparse_data_are_explicit(self):
        a = observed_events(events(), 10)
        r = reconstruct(a)
        for name, b in r['bands'].items():
            bid = ['g', 'r'].index(name)
            t = a[a[:, 2] == bid, 0]
            self.assertGreaterEqual(min(b['days']), t.min())
            self.assertLessEqual(max(b['days']), t.max())
        self.assertEqual(reconstruct(a[:2])['status'], 'insufficient_data')


if __name__ == '__main__':
    unittest.main()
