"""Regression checks for calibrated next-observation value and latent-class mixtures."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flare.analysis import FlareAnalysis
from flare.analysis_report import render_analysis
from flare.cli import main as cli_main
from flare.config import PKG_ROOT
from flare.next_observation import NextObservationPredictor, venn_abers_binary
from flare.skyportal import analyze as skyportal_analyze


def make_events() -> np.ndarray:
    times = np.array([0.0, 1.5, 3.0, 4.5, 7.0, 9.5, 13.0])
    a = np.zeros((2 * len(times), 15), dtype=float)
    a[:, 0] = np.repeat(times, 2)
    a[:, 1] = np.r_[0.0, np.diff(a[:, 0])]
    a[:, 2] = np.tile([0.0, 1.0], len(times))
    a[:, 3] = 1.7 + 0.04 * a[:, 0] - 0.0008 * (a[:, 0] ** 2) + 0.1 * a[:, 2]
    a[:, 4] = 0.025
    a[np.arange(len(a)), 5 + a[:, 2].astype(int)] = 1.0
    return a


class NextObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FlareAnalysis(num_threads=1)
        cls.folder = PKG_ROOT / "models" / "bts6" / "next_observation"

    def test_grid_recipe_keeps_legacy_fields_and_adds_venn_abers_calibration(self):
        original = make_events()
        changed = original.copy()
        changed[changed[:, 0] > 5.0, 3:] = 99.0
        r = self.engine.next_observation_value(original, 5, recipe="grid")
        self.assertEqual(r, self.engine.next_observation_value(changed, 5, recipe="grid"))
        for v in r["bands"].values():
            self.assertEqual(v["status"], "ok")
            self.assertTrue(0.0 <= v["p_singleton"] <= 1.0)
            self.assertTrue(0.0 <= v["p_singleton_calibrated"] <= 1.0)
            lo, hi = v["p_singleton_interval"]
            self.assertTrue(0.0 <= lo <= v["p_singleton_calibrated"] <= hi <= 1.0)
        json.dumps(r, allow_nan=False)

    def test_gamut_recipe_is_causal_calibrated_and_handles_single_band_prefixes(self):
        original = make_events()
        changed = original.copy()
        changed[changed[:, 0] > 5.0, 3:] = 999.0
        for h in (5, 10):
            r1 = self.engine.next_observation_value(original, h, recipe="gamut")
            r2 = self.engine.next_observation_value(changed if h == 5 else original, h, recipe="gamut")
            if h == 5:
                self.assertEqual(r1, r2)
            self.assertEqual(r1["recipe"], "gamut")
            self.assertEqual(r1["set_size_before"], len(r1["set_before"]))
            self.assertAlmostEqual(sum(r1["membership_probabilities"].values()), 1.0, places=4)
            for b_name in ("g", "r"):
                v = r1["bands"][b_name]
                self.assertEqual(v["status"], "ok")
                self.assertEqual(v["next_day"], float(h + 1))
                self.assertGreaterEqual(v["expected_set_size"], 0.0)
                self.assertLessEqual(v["expected_set_size"], 6.0)
                self.assertAlmostEqual(v["delta_S"], r1["set_size_before"] - v["expected_set_size"], places=9)
                lo, hi = v["p_singleton_interval"]
                self.assertTrue(0.0 <= lo <= v["p_singleton"] <= hi <= 1.0)
                self.assertTrue(0.0 <= v["p_singleton_raw"] <= 1.0)
            json.dumps(r1, allow_nan=False)

        # Single-band prefix: grid marks missing band insufficient_data, whereas GAMUT predicts both bands
        only_g = original[original[:, 2] == 0.0]
        r_grid = self.engine.next_observation_value(only_g, 5, recipe="grid")
        r_gamut = self.engine.next_observation_value(only_g, 5, recipe="gamut")
        self.assertEqual(r_grid["bands"]["r"]["status"], "insufficient_data")
        self.assertEqual(r_gamut["bands"]["g"]["status"], "ok")
        self.assertEqual(r_gamut["bands"]["r"]["status"], "ok")

    def test_venn_abers_monotone_lookup_and_manifest_guard(self):
        table = {"edges": [0.2, 0.7], "p0": [0.05, 0.3, 0.8], "p1": [0.10, 0.5, 0.9]}
        p_low, lo_l, hi_l = venn_abers_binary(0.1, table)
        p_high, lo_h, hi_h = venn_abers_binary(0.85, table)
        self.assertLess(p_low, p_high)
        self.assertTrue(lo_l <= p_low <= hi_l)
        self.assertTrue(lo_h <= p_high <= hi_h)

        pred = NextObservationPredictor(self.folder, num_threads=1)
        for name in pred.manifest:
            pred.checked_path(name)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "h5.json").write_text("{}")
            (folder / "manifest.json").write_text(json.dumps(dict(algorithm="sha256", files={"h5.json": "bad"})))
            tampered = NextObservationPredictor(folder)
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                tampered.checked_path("h5.json")
            with self.assertRaisesRegex(ValueError, "escapes"):
                tampered.checked_path("../outside.txt")
        with self.assertRaises(ValueError):
            self.engine.next_observation_value(make_events(), 5, recipe="unknown")
        with self.assertRaises(ValueError):
            self.engine.next_observation_value(make_events(), 20, recipe="gamut")

    def test_analyze_cli_html_and_skyportal_next_obs_integration(self):
        ev = make_events()
        out = self.engine.analyze(
            ev, horizon_days=10, include_reconstruction=False, next_obs_horizon=5, next_obs_recipe="gamut"
        )
        self.assertEqual(out["next_observation"]["recipe"], "gamut")
        html = render_analysis({"obj": out})
        self.assertIn("Value of the next observation", html)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ev_path = tmp_dir / "obj.npz"
            out_path = tmp_dir / "out.json"
            np.savez(ev_path, data=ev.astype(np.float32))
            rc = cli_main(
                [
                    "analyze",
                    str(ev_path),
                    "--horizon",
                    "10",
                    "--next-obs-horizon",
                    "5",
                    "--next-obs-recipe",
                    "gamut",
                    "--no-reconstruction",
                    "-o",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            parsed = json.loads(out_path.read_text())
            self.assertEqual(parsed[str(ev_path)]["next_observation"]["recipe"], "gamut")

        t0 = 60500.0
        rows = []
        for dt in [0.0, 0.8, 1.6, 2.4, 3.2, 4.0]:
            for fid, off in ((1, 0.0), (2, 0.15)):
                mag = 19.5 - 1.2 * (1.0 - math.exp(-dt / 4.0)) + off
                rows.append((t0 + dt + 0.01 * fid, fid, mag, 0.05))
        sp = skyportal_analyze(
            sorted(rows),
            None,
            dict(context="none", plot=False, forecast=False, reconstruction=False, next_obs="gamut"),
            resource_id="synthetic_nextobs",
            work_dir=".",
        )
        nx = sp["results"]["next_observation"]
        self.assertEqual(nx["status"], "ok")
        self.assertEqual(nx["recipe"], "gamut")
        flat = sp["annotations_flat"]
        self.assertIn("flare_next_obs_day", flat)
        self.assertIn("flare_next_p_singleton_g", flat)
        self.assertIn("flare_next_p_singleton_r", flat)


if __name__ == "__main__":
    unittest.main()
