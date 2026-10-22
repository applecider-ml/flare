"""False-alarm rate per unit time for a service that re-analyses every source as its photometry grows.

The per-analysis conformal p-value of the consensus (flare.consensus) controls the false-alarm rate of
one analysis. A saved source is analysed again at every new detection, about fifty times over a
hundred days on Fritz, so the chance that a known-class object is flagged at some point in its life is
much larger than the per-analysis rate (15% of the test objects are flagged sometime at the nominal 1%).

Lifetime significance. At the k-th analysis of an object, the statistic is m_k, the smallest per-analysis
p-value it has produced so far. Its lifetime p-value at k is the conformal p-value of m_k against the
running minima of the known-class objects at their own k-th analysis, from a replay of the known
population detection by detection, and P_life is the smallest lifetime p-value over the analyses so
far. P_life is a statistic, and the minimum over analyses is not itself a valid p-value under repeated
testing (an anytime-valid construction would be needed for that, e.g. Johari et al. 2022); its
false-alarm rate is therefore set empirically: the fraction of the replayed known objects whose own
P_life, over their whole life under the same schedule and horizon, is at least as small, times the
rate at which sources enter the service. That is the rate at which the known population produces an
object at least this significant, in the sense of a gravitational-wave search, for this horizon,
schedule and stream rate. An object flagged at its first analysis carries no trials factor. The reference (one sorted array per analysis
index, the stream rate and the thresholds of the reported levels) ships with the models as
``lifetime_far.json``; it is built by replaying the known test objects with the deployed models on the
photometry that the service receives (scripts/export_lifetime_far.py).

At analysis time the prefixes of the light curve are replayed with the same schedule as the reference,
so the service keeps no state between analyses. The schedule is nested (every detection count up to
MAX_EVERY, then every STRIDE-th, independently of the length of the light curve), so that a new
detection only adds analyses and an alarm reached at an earlier analysis is never lost; beyond
MAX_EVERY the newest detections enter the lifetime statistic at the next multiple of STRIDE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FILENAME = "lifetime_far.json"
MAX_EVERY = 40      # every detection up to this count, then every fifth
STRIDE = 5


def on_schedule(k: int) -> bool:
    """Whether the analysis at detection count k belongs to the replay schedule."""
    return k >= 3 and (k <= MAX_EVERY or k % STRIDE == 0)


def steps(n: int) -> list[int]:
    """The analysis indices (detection counts) at which a light curve of n detections is replayed;
    steps(n) is a prefix of steps(n + 1)."""
    return [k for k in range(3, n + 1) if on_schedule(k)]


def running_minimum(p_by_k: dict[int, float]) -> dict[int, float]:
    m, out = 1.0, {}
    for k in sorted(p_by_k):
        m = min(m, float(p_by_k[k])); out[k] = m
    return out


class LifetimeReference:
    """The replayed known population: sorted running minima per analysis index, stream rate, levels."""

    def __init__(self, table: dict):
        self.ref = {int(k): np.asarray(v, dtype=float) for k, v in table["running_min"].items()}
        self.ks = np.array(sorted(self.ref))
        self.rate_per_day = float(table["stream_rate_per_day"])
        self.levels = {name: float(v) for name, v in table["levels"].items()}     # name -> P_life threshold
        self.p_life_ref = np.sort(np.asarray(table["p_life_reference"], dtype=float))  # P_life of the replayed known objects
        self.meta = {k: v for k, v in table.items() if k not in ("running_min", "levels", "p_life_reference")}

    def lifetime_p(self, k: int, m: float) -> float:
        """Conformal p-value of a running minimum m at analysis index k; beyond the indices that enough
        known objects reach, the last populated index is the reference."""
        kk = self.ks[self.ks <= k]
        s = self.ref[int(kk[-1])] if len(kk) else self.ref[int(self.ks[0])]
        return float((1 + np.searchsorted(s, m, side="right")) / (len(s) + 1))

    def significance(self, p_by_k: dict[int, float]) -> dict:
        """P_life and the false-alarm rate per unit time for the per-analysis p-values of one object."""
        run = running_minimum(p_by_k)
        life = {k: self.lifetime_p(k, m) for k, m in run.items()}
        p_life = min(life.values()) if life else 1.0
        # the lifetime false-alarm fraction: known objects whose whole-life P_life is at least as small
        frac = float((1 + np.searchsorted(self.p_life_ref, p_life, side="right")) / (len(self.p_life_ref) + 1))
        far_day = frac * self.rate_per_day
        level = None
        for name, thr in sorted(self.levels.items(), key=lambda kv: kv[1]):     # strictest first
            if p_life <= thr:
                level = name; break
        return {
            "lifetime_p": round(p_life, 5),
            "lifetime_false_alarm_fraction": round(frac, 5),
            "false_alarms_per_day": round(far_day, 4),
            "one_false_alarm_per_days": round(1.0 / far_day, 1) if far_day > 0 else None,
            "level": level,
            "levels": {name: bool(p_life <= thr) for name, thr in self.levels.items()},
            "n_analyses": len(run),
            "first_analysis_at_level": {name: next((k for k in sorted(life) if life[k] <= thr), None)
                                        for name, thr in self.levels.items()},
        }


def load(directory: str | Path) -> LifetimeReference | None:
    path = Path(directory) / FILENAME
    if not path.exists():
        return None
    return LifetimeReference(json.loads(path.read_text()))
