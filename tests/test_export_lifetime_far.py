"""Lifetime FAR exports must preserve inclusive empirical budgets after JSON loading."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from flare import lifetime as LT
from scripts.export_lifetime_far import LEVELS, level_threshold

ROOT = Path(__file__).resolve().parents[1]


def _export(tmp_path, known, rate=1.0, rare=()):
    replay = tmp_path / "replay.csv"
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text('{"existing.txt": "existing-digest"}')
    with replay.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["population", "trajectory"])
        for population, runs in (("known", known), ("rare", rare)):
            for run in runs:
                trajectory = ";".join(f"{k},0,bts6,{p!r}" for k, p in run.items())
                writer.writerow([population, trajectory])
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_lifetime_far.py"),
         "--replay", str(replay), "--models", str(models), f"--rate={rate}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return result, models


def _check_loaded_export(models, known, rare=()):
    table = json.loads((models / LT.FILENAME).read_text())
    ref = LT.load(models)
    assert ref is not None
    # Compare raw statistics, before the presentation rounding in significance().
    cal = np.sort([
        min((ref.lifetime_p(k, m) for k, m in LT.running_minimum(run).items()), default=1.0)
        for run in known
    ])
    np.testing.assert_array_equal(ref.p_life_ref, cal)
    results = [ref.significance(run) for run in known]
    for name, days in LEVELS.items():
        count = sum(result["levels"][name] for result in results)
        assert count == np.count_nonzero(cal <= ref.levels[name])
        assert table["realised_on_reference"][name] == count / len(known)
        assert count / len(known) * ref.rate_per_day <= 1.0 / days
        assert table["rares_for_the_record"][name] == sum(
            ref.significance(run)["levels"][name] for run in rare
        )
    manifest = json.loads((models / "manifest.json").read_text())
    assert manifest == {
        "existing.txt": "existing-digest",
        LT.FILENAME: hashlib.sha256((models / LT.FILENAME).read_bytes()).hexdigest(),
    }
    return table, ref


@pytest.mark.parametrize(("cal", "rate", "days", "expected"), [
    ([0.1] * 50 + [0.8] * 50, 1.0, 7.0, 0.0),  # first tie exceeds quota
    ([0.1] * 2 + [0.2] * 3 + [0.8] * 5, 0.5, 4.0, 0.2),  # exact quota
    ([0.1] * 2 + [0.2] * 3 + [0.8] * 5, 0.5, 5.0, 0.1),  # omit crossing tie
    ([0.1, 0.2, 0.3, 0.4], 1.0, 2.0, 0.2),  # unchanged without ties
])
def test_threshold_accepts_only_complete_groups(cal, rate, days, expected):
    assert level_threshold(cal, rate, days) == expected


def test_cli_rejects_the_large_tie_group(tmp_path):
    known = [{3: p} for p in [0.01] * 50 + [0.8] * 50]
    result, models = _export(tmp_path, known)
    assert result.returncode == 0, result.stderr
    table, ref = _check_loaded_export(models, known)
    assert set(table["levels"].values()) == {0.0}
    assert not any(ref.significance({3: 0.0})["levels"].values())
    assert ref.lifetime_p(3, 0.0) > 0  # the zero sentinel cannot fire


def test_cli_preserves_running_minimum_precision_and_runtime_far(tmp_path):
    values = [0.01000001 + i * 1e-9 for i in range(100)]
    assert len({round(p, 6) for p in values}) == 1
    known = [{3: p} for p in values]
    rare = [{3: 0.0}, {3: 0.9}]
    result, models = _export(tmp_path, known, rare=rare)
    assert result.returncode == 0, result.stderr
    table, ref = _check_loaded_export(models, known, rare)
    assert table["running_min"]["3"] == values
    np.testing.assert_array_equal(ref.p_life_ref, np.arange(2, 102) / 101)
    for name, days in LEVELS.items():
        assert sum(ref.significance(run)["levels"][name] for run in known) == int(100 / days)
    # The named level uses count/N; runtime separately reports the +1 estimate.
    at_quarter = ref.significance(known[0])
    assert at_quarter["levels"]["1 per quarter"]
    assert at_quarter["lifetime_false_alarm_fraction"] == round(2 / 101, 5)
    assert at_quarter["false_alarms_per_day"] > 1 / 91


def test_cli_preserves_distinct_lifetime_statistics_below_six_decimal_precision(tmp_path):
    known = [{3: 0.01}, {3: 0.2, 4: 0.01}] + [{3: 0.8, 4: 0.8}] * 2498
    result, models = _export(tmp_path, known, rate=300.0)
    assert result.returncode == 0, result.stderr
    table, ref = _check_loaded_export(models, known)
    first, second = table["p_life_reference"][:2]
    assert first < second and round(first, 6) == round(second, 6)
    assert ref.levels["1 per week"] == first
    assert ref.significance(known[0])["levels"]["1 per week"]
    assert not ref.significance(known[1])["levels"]["1 per week"]
    assert ref.significance(known[0])["lifetime_false_alarm_fraction"] == round(2 / 2501, 5)


@pytest.mark.parametrize("rate", [np.finfo(float).max, np.nextafter(0.0, 1.0)])
def test_cli_handles_extreme_positive_rates(tmp_path, rate):
    known = [{3: 0.01}] * 50
    result, models = _export(tmp_path, known, rate=rate)
    assert result.returncode == 0, result.stderr
    table, ref = _check_loaded_export(models, known)
    expected = 0.0 if rate > 1 else 1.0
    assert set(table["levels"].values()) == {expected}
    assert all(ref.significance(known[0])["levels"].values()) == (rate < 1)


@pytest.mark.parametrize("rate", [0, -1, "nan", "inf", "-inf"])
def test_cli_rejects_invalid_stream_rates_without_writing(tmp_path, rate):
    result, models = _export(tmp_path, [{3: 0.1}] * 50, rate=rate)
    assert result.returncode != 0
    assert "--rate must be finite and strictly positive" in result.stderr
    assert not (models / LT.FILENAME).exists()
    assert json.loads((models / "manifest.json").read_text()) == {"existing.txt": "existing-digest"}


@pytest.mark.parametrize(("known", "message"), [
    ([], "no known-class objects"),
    ([{3: 0.1}] * 49, "no analysis index has at least 50 known-class objects"),
    ([{}] * 50, "no analysis index has at least 50 known-class objects"),
])
def test_cli_rejects_missing_references_without_writing(tmp_path, known, message):
    result, models = _export(tmp_path, known)
    assert result.returncode != 0 and message in result.stderr
    assert not (models / LT.FILENAME).exists()
    assert json.loads((models / "manifest.json").read_text()) == {"existing.txt": "existing-digest"}


def test_empty_individual_trajectory_matches_runtime(tmp_path):
    known = [{3: 0.1}] * 50 + [{}]
    result, models = _export(tmp_path, known, rate=0.01)
    assert result.returncode == 0, result.stderr
    table, ref = _check_loaded_export(models, known)
    assert table["n_known"] == 51
    assert ref.significance({})["lifetime_p"] == 1.0
