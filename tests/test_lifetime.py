"""flare.lifetime: replay schedule, lifetime significance and the shipped reference."""
import json
from pathlib import Path

import numpy as np

from flare import lifetime as LT

MODELS = Path(__file__).resolve().parents[1] / "models" / "bts6"


def test_schedule_every_detection_then_every_fifth_and_nested():
    assert LT.steps(5) == [3, 4, 5]
    s = LT.steps(47)
    assert s[:38] == list(range(3, 41)) and s[38:] == [45]
    for n in range(3, 120):     # a new detection only adds analyses
        assert LT.steps(n + 1)[:len(LT.steps(n))] == LT.steps(n)


def test_an_alarm_is_never_lost_when_a_detection_arrives():
    ref = _ref()
    p = {k: 0.5 for k in range(3, 60)}; p[47] = 1e-4     # a single very significant analysis at 47
    seen = [ref.significance({k: p[k] for k in LT.steps(n)})["lifetime_p"] for n in range(40, 60)]
    assert all(b <= a for a, b in zip(seen, seen[1:]))


def test_running_minimum_is_monotone():
    run = LT.running_minimum({3: 0.5, 4: 0.2, 5: 0.4, 10: 0.1})
    assert list(run.values()) == [0.5, 0.2, 0.2, 0.1]


def _ref():
    rng = np.random.default_rng(0)
    table = {"running_min": {k: sorted(rng.uniform(0, 1, 500) ** (1 + 0.05 * k)) for k in range(3, 21)},
             "stream_rate_per_day": 7.0, "levels": {"1 per week": 0.02, "1 per month": 0.005},
             "p_life_reference": sorted(rng.uniform(0, 1, 500) ** 3)}
    return LT.LifetimeReference(table)


def test_first_analysis_carries_no_trials_factor_and_levels_are_ordered():
    ref = _ref()
    one = ref.significance({3: 0.001})
    assert one["n_analyses"] == 1 and one["lifetime_p"] == round(ref.lifetime_p(3, 0.001), 5)
    assert one["false_alarms_per_day"] == round(one["lifetime_false_alarm_fraction"] * 7.0, 4)
    assert one["lifetime_false_alarm_fraction"] >= one["lifetime_p"] * 0.5
    many = ref.significance({k: 0.3 for k in range(3, 21)})
    assert many["lifetime_p"] >= one["lifetime_p"] and many["level"] is None and one["level"] in ("1 per week", "1 per month")
    assert ref.lifetime_p(40, 0.01) == ref.lifetime_p(20, 0.01)     # beyond the populated indices the last one is used


def test_shipped_reference_is_consistent():
    ref = LT.load(MODELS)
    assert ref is not None and ref.rate_per_day > 0 and set(ref.levels) >= {"1 per week", "1 per month"}
    assert ref.levels["1 per month"] < ref.levels["1 per week"]
    man = json.loads((MODELS / "manifest.json").read_text())
    assert LT.FILENAME in man
    for k in ref.ks:
        assert np.all(np.diff(ref.ref[int(k)]) >= 0) and len(ref.ref[int(k)]) >= 50
