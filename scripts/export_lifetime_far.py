"""Build models/bts6/lifetime_far.json (flare.lifetime) from a detection-by-detection replay of the known
test objects with the deployed models on the photometry that the service receives.

Input: a CSV with one row per object (columns population, trajectory), where trajectory is
"k,day,model,p;..." for the analysis indices of flare.lifetime.steps (fritz_dataset/scripts/
sequential_replay.py produces it); analyses off the schedule of flare.lifetime are dropped. Only known-class objects enter the reference; the out-of-taxonomy
objects are reported for the record. The stream rate is the number of sources saved per day in the
Fritz groups that dispatch the service (measured over the stated window).
Run: python scripts/export_lifetime_far.py --replay <csv> --rate 6.8 --rate-note "..." --models models/bts6
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from flare.lifetime import FILENAME, on_schedule, running_minimum  # noqa: E402

LEVELS = {"1 per week": 7.0, "1 per month": 30.0, "1 per quarter": 91.0}
MIN_REF = 50


def traj(s):
    out = {}
    for item in str(s).split(";"):
        if item and item != "nan":
            k, _day, _model, p = item.split(",")
            if on_schedule(int(k)):     # replays made with the earlier schedule also hold the last detection count
                out[int(k)] = float(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", type=Path, required=True); ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--rate", type=float, required=True); ap.add_argument("--rate-note", default="")
    a = ap.parse_args()
    d = pd.read_csv(a.replay)
    known = [running_minimum(traj(t)) for t in d[d.population == "known"].trajectory]
    rare = [running_minimum(traj(t)) for t in d[d.population == "rare"].trajectory]
    ks = sorted({k for r in known for k in r})
    ref = {k: np.sort([r[k] for r in known if k in r]) for k in ks}
    ref = {k: v for k, v in ref.items() if len(v) >= MIN_REF}
    kk = np.array(sorted(ref))

    def p_life(run):
        best = 1.0
        for k, m in run.items():
            c = kk[kk <= k]; s = ref[int(c[-1])] if len(c) else ref[int(kk[0])]
            best = min(best, (1 + np.searchsorted(s, m, side="right")) / (len(s) + 1))
        return best

    cal = np.sort([p_life(r) for r in known])
    levels = {}
    for name, days in LEVELS.items():
        frac = 1.0 / (days * a.rate); n_allowed = int(np.floor(frac * len(cal)))
        levels[name] = float(cal[n_allowed - 1]) if n_allowed >= 1 else 0.0
    pr = np.array([p_life(r) for r in rare])
    table = dict(
        running_min={int(k): [round(float(x), 6) for x in v] for k, v in ref.items()},
        stream_rate_per_day=a.rate, stream_rate_note=a.rate_note, levels=levels,
        p_life_reference=[round(float(x), 6) for x in cal],
        n_known=len(known), n_analyses_median=float(np.median([len(r) for r in known])),
        realised_on_reference={name: float(np.mean(cal <= thr)) for name, thr in levels.items()},
        rares_for_the_record={name: int((pr <= thr).sum()) for name, thr in levels.items()}, n_rares=len(rare),
        note=("P_life = min over analyses of the conformal p-value of the running-minimum per-analysis p against "
              "the known test objects replayed detection by detection with the deployed models on the photometry "
              "that Fritz supplies to the service (ZTF rows); thresholds are the quantiles of P_life on the same "
              "objects, so the realised rates above are in-sample (split-half check in the paper); the false-alarm "
              "rate per unit time is the fraction of these objects whose P_life is at least as small, times the stream rate"))
    out = a.models / FILENAME
    out.write_text(json.dumps(table))
    man_path = a.models / "manifest.json"; man = json.loads(man_path.read_text())
    man[FILENAME] = hashlib.sha256(out.read_bytes()).hexdigest()
    man_path.write_text(json.dumps(dict(sorted(man.items())), indent=2))
    print(f"{out}: {len(ref)} analysis indices, {len(known)} known objects; levels {levels}; "
          f"realised {table['realised_on_reference']}; rares {table['rares_for_the_record']} of {len(rare)}; {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
