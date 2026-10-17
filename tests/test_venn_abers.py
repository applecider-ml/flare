"""Venn-Abers tables: the step-table lookup reproduces a direct isotonic fit, and the shipped
classifiers return an interval for every class."""
import unittest
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from flare.venn_abers import build_table, intervals

MODELS = Path(__file__).resolve().parents[1] / "models" / "bts6"


def direct(s_cal, y_cal, s):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    xs = np.append(s_cal, s)
    return (ir.fit(xs, np.append(y_cal, 0.0)).predict([s])[0], ir.fit(xs, np.append(y_cal, 1.0)).predict([s])[0])


class VennAbersTests(unittest.TestCase):
    def test_table_matches_direct_fit(self):
        rng = np.random.default_rng(3)
        s = rng.random(400); y = rng.random(400) < s ** 2
        table = {"c": build_table(s, y)}
        test = rng.random(50)
        lo, hi, merged = intervals(test[:, None], table, ["c"])
        for k, t in enumerate(test):
            d0, d1 = direct(s, y.astype(float), t)
            self.assertAlmostEqual(lo[k, 0], d0, places=5)
            self.assertAlmostEqual(hi[k, 0], d1, places=5)
        self.assertTrue(np.all(lo <= hi))

    @unittest.skipUnless((MODELS / "venn_abers.json").exists(), "Venn-Abers tables not shipped")
    def test_classifier_reports_intervals(self):
        import pandas as pd
        from flare import skyportal as sp
        from flare.config import BTS6
        clf = sp._classifier(None)
        X = pd.DataFrame([{c: np.nan for c in clf.feature_names}])
        out = sp.classify(X, base_rate=0.01, horizon=None)
        iv = out["probability_intervals"]
        self.assertEqual(set(iv), set(BTS6.classes))
        for lo, hi in iv.values():
            self.assertTrue(0.0 <= lo <= hi <= 1.0)


if __name__ == "__main__":
    unittest.main()
