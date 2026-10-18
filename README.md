# Early-time anomaly alarm calibrated per number of detections (2026-10-18)

Changed files only, against `applecider-ml/flare` main `029a9e9`. The folder `flare/` mirrors
the repository; every file goes to the same path.

## What it fixes

With a single reference, the day-5 consensus alarm fired on 10% of the known objects with nine
or more detections and on none of those with three or fewer, at a nominal 1%. For the day-5, 10
and 20 models, the p-values are now computed separately for groups of calibration objects with a
similar number of detections (`n_obs_total`), each group holding at least 100 objects so that a
1% alarm stays attainable. These models also calibrate on twice as many objects as before: the
p-values come from the whole validation split, which no fit touched, and the percentile ranks are
frozen on the test split. **The full-curve model is unchanged** (it showed no dependence on the
number of detections, 0.3–2.2% across groups, within noise).

Checked on test objects that the calibration does not use (ranks from half of the test split,
p-values from the validation split, evaluation on the other half):

| model | 1% alarm on known objects, by detections | overall at 1% / 5% / 10% |
|---|---|---|
| day 5, before | 0–3: 0.0%, 4–5: 1.1%, 6–8: 2.0%, 9+: 10.0% | 1.1 / 5.1 / 10.4% |
| day 5, now | 0–3: 0.3%, 4–5: 2.2%, 6–8: 0.0%, 9+: 0.0% | 0.9 / 5.7 / 10.9% |
| day 10, before → now | 9+: 3.1% → 1.0% | 1.1 → 0.8% at 1% |
| day 20, before → now | 9+: 0.9% → 0.5% | 0.8 → 0.5% at 1% |

Hold-out (37 novae + 16 without a parent class), detected at 1% / 5% / 10%:

| model | before | now |
|---|---|---|
| full | 10 / 25 / 33 | unchanged |
| day 5 | 6 / 26 / 32 | 6 / 27 / 33 |
| day 10 | 13 / 25 / 31 | 14 / 26 / 33 |
| day 20 | 13 / 28 / 35 | 11 / 29 / 36 |

The GRB ZTF26abzxgsr is still flagged at 1% (p = 0.003, was 0.004).

## Files

| file | change |
|---|---|
| `flare/consensus.py` | Mondrian p-values by number of detections at run time; files without the groups still work |
| `scripts/export_consensus.py` | builds the groups for day 5/10/20; the full-curve export is unchanged (same thresholds and p-values as the shipped file) |
| `models/bts6/early/h5`, `h10`, `h20`: `consensus.pkl.gz` | re-exported |
| `models/bts6/early/h5`, `h10`, `h20`: `manifest.json` | new sha256 of the three files |
| `tests/test_consensus.py` | one more test (50 pass) |

The three `consensus.pkl.gz` and the three `manifest.json` must go in together: FLARE refuses a
consensus file whose sha256 does not match its manifest.

```bash
cd flare-repo && git checkout main && git pull       # must be at 029a9e9
cp -R ../push_mondrian_20261018/flare/. ./
git status                                           # 9 modified files
git add -A && git commit -m "Early-time consensus alarm: p-values per number of detections"
git push
```

No change is needed in osg-skyportal-plugin beyond the earlier patch; the next image build picks
this up.
