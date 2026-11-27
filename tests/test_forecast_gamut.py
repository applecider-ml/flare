"""Regression checks for GAMUT class-conditional forecast bands."""
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
from flare.forecast_gamut import (
    GamutForecaster,
    adaptive_grid,
    apply_temperature,
    extract_components,
)
from flare.skyportal import analyze as skyportal_analyze


def make_events(max_day: float = 35.0) -> np.ndarray:
    times = np.array([0.0, 1.5, 3.0, 4.5, 7.0, 9.5, 13.0, 18.0, 24.0, 29.0, 34.0])
    times = times[times <= max_day]
    a = np.zeros((2 * len(times), 15), dtype=float)
    a[:, 0] = np.repeat(times, 2)
    a[:, 1] = np.r_[0.0, np.diff(a[:, 0])]
    a[:, 2] = np.tile([0.0, 1.0], len(times))
    a[:, 3] = 1.7 + 0.035 * a[:, 0] - 0.0007 * (a[:, 0] ** 2) + 0.1 * a[:, 2]
    a[:, 4] = 0.025
    a[np.arange(len(a)), 5 + a[:, 2].astype(int)] = 1.0
    return a


class ForecastGamutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FlareAnalysis(num_threads=1)
        cls.folder = PKG_ROOT / "models" / "bts6" / "forecast_gamut"

    def test_future_detections_cannot_change_gamut_forecast(self):
        original = make_events()
        changed = original.copy()
        changed[changed[:, 0] > 5.0, 3:] = 999.0
        for mode in ("crossfit", "split"):
            a = self.engine.forecast(original, 5, recipe="gamut", days=[6.5, 22.0, 90.0], gamut_mode=mode)
            b = self.engine.forecast(changed, 5, recipe="gamut", days=[6.5, 22.0, 90.0], gamut_mode=mode)
            self.assertEqual(a, b)
            json.dumps(a, allow_nan=False)

    def test_all_five_horizons_and_band_invariants(self):
        ev = make_events()
        for h in (5, 10, 15, 20, 30):
            for mode in ("crossfit", "split"):
                res = self.engine.forecast(ev, h, recipe="gamut", gamut_mode=mode)
                self.assertEqual(res["status"], "ok")
                self.assertEqual(res["recipe"], "gamut")
                self.assertEqual(res["gamut_mode"], mode)
                self.assertAlmostEqual(sum(res["membership_probabilities"].values()), 1.0, places=4)
                for band_name in ("g", "r"):
                    b = res["bands"][band_name]
                    self.assertEqual(b["status"], "ok")
                    self.assertEqual(b["days"][0], float(h))
                    mag = np.asarray(b["magnitude"], dtype=float)
                    lo68 = np.asarray(b["lower68"], dtype=float)
                    hi68 = np.asarray(b["upper68"], dtype=float)
                    lo95 = np.asarray(b["lower95"], dtype=float)
                    hi95 = np.asarray(b["upper95"], dtype=float)
                    w68 = np.asarray(b["width68"], dtype=float)
                    w95 = np.asarray(b["width95"], dtype=float)
                    # Anchor point has zero interval width
                    self.assertEqual(lo95[0], mag[0])
                    self.assertEqual(hi95[0], mag[0])
                    self.assertEqual(w68[0], 0.0)
                    # Nesting and containment invariants
                    self.assertTrue(np.all(lo95 <= lo68 + 1e-12))
                    self.assertTrue(np.all(lo68 <= mag + 1e-12))
                    self.assertTrue(np.all(mag <= hi68 + 1e-12))
                    self.assertTrue(np.all(hi68 <= hi95 + 1e-12))
                    self.assertTrue(np.all(w95 >= w68 - 1e-10))
                    # Lebesgue measure matches connected interval components
                    for idx, comps in enumerate(b["intervals68"]):
                        comp_len = sum(hi - lo for lo, hi in comps)
                        self.assertAlmostEqual(comp_len, float(w68[idx]), places=9)
                        self.assertLessEqual(comp_len, float(hi68[idx] - lo68[idx]) + 1e-9)

    def test_multimodal_level_set_splits_into_disjoint_intervals(self):
        centers = np.array([[-2.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
        scales = np.full((1, 6), 0.2)
        y_grid, _ = adaptive_grid(np.array([-4.0]), np.array([4.0]), centers, scales)
        bimodal_inside = (np.abs(y_grid + 2.0) <= 0.3) | (np.abs(y_grid - 2.0) <= 0.3)
        lo, hi, width, comps = extract_components(y_grid, bimodal_inside, offset=18.0)
        self.assertEqual(len(comps[0]), 2)
        self.assertLess(width[0], (hi[0] - lo[0]) * 0.5)
        p_scaled = apply_temperature(np.array([0.5, 0.3, 0.05, 0.05, 0.05, 0.05]), 0.7)
        self.assertAlmostEqual(float(p_scaled.sum()), 1.0, places=10)

    def test_missing_band_and_invalid_requests(self):
        ev = make_events()
        only_g = ev[ev[:, 2] == 0.0]
        res = self.engine.forecast(only_g, 10, recipe="gamut")
        self.assertEqual(res["bands"]["g"]["status"], "ok")
        self.assertEqual(res["bands"]["r"]["status"], "insufficient_data")
        for bad_days in [[10.0], [91.0], [], [25.0, 20.0], [20.0, 20.0], [np.nan]]:
            with self.assertRaises(ValueError):
                self.engine.forecast(ev, 10, recipe="gamut", days=bad_days)
        with self.assertRaises(ValueError):
            self.engine.forecast(ev, 7, recipe="gamut")
        with self.assertRaises(ValueError):
            self.engine.forecast(ev, 10, recipe="gamut", gamut_mode="invalid")

    def test_manifest_rejects_modified_artifact_and_path_escape(self):
        model = GamutForecaster(self.folder, num_threads=1)
        for name in model.manifest:
            model.checked_path(name)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "h5.json").write_text("{}")
            (folder / "manifest.json").write_text(json.dumps(dict(algorithm="sha256", files={"h5.json": "wrong"})))
            tampered = GamutForecaster(folder)
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                tampered.checked_path("h5.json")
            with self.assertRaisesRegex(ValueError, "escapes"):
                tampered.checked_path("../outside.txt")

    def test_unified_analysis_cli_and_skyportal_overlay(self):
        ev = make_events()
        result = self.engine.analyze(ev, horizon_days=10, forecast_horizon=5, forecast_recipe="gamut", include_reconstruction=False)
        self.assertEqual(result["forecast"]["recipe"], "gamut")
        html = render_analysis({"obj": result})
        self.assertIn("GAMUT dual-weighted posterior-mixture level sets", html)

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
                    "--forecast-horizon",
                    "5",
                    "--forecast-recipe",
                    "gamut",
                    "--no-reconstruction",
                    "-o",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            parsed = json.loads(out_path.read_text())
            self.assertEqual(parsed[str(ev_path)]["forecast"]["recipe"], "gamut")

        t0 = 60500.0
        rows = []
        for dt in [0.0, 0.9, 2.1, 3.0, 4.2, 5.1, 6.0, 7.2, 8.1, 9.0, 10.2, 11.0]:
            for fid, off in ((1, 0.0), (2, 0.15)):
                mag = 19.5 - 1.2 * (1.0 - math.exp(-dt / 4.0)) + off
                rows.append((t0 + dt + 0.01 * fid, fid, mag, 0.05))
        sp = skyportal_analyze(
            sorted(rows),
            None,
            dict(context="none", plot=False, forecast_recipe="gamut"),
            resource_id="synthetic_gamut",
            work_dir=".",
        )
        self.assertEqual(sp["results"]["forecast"]["status"], "ok")
        self.assertEqual(sp["results"]["forecast"]["recipe"], "gamut")
        for curve in sp["model_lightcurve"].values():
            self.assertTrue(all(p[2] <= p[1] <= p[3] for p in curve))


if __name__ == "__main__":
    unittest.main()
