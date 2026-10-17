# Follow-up: classification ensemble and forecast over time

**2026-09-17 correction:** the forecast tables below predate the discovery of
future-counterpart leakage in cached early colors. The chronological models
have not been rescored here. Do not interpret these tables as fully causal
validation. Full-curve classification is unaffected by this specific cutoff
issue. See the [correction and external test](external_forecast_and_colors.md).

The existing optional forecast blend retains its aggregate benefit when fitted
on older catalog-peak cohorts and tested on the 2024 and 2025 cohorts. This is
additional evidence for the same recipe, not another change to its weights.
The three-seed classifier improves point estimates but fails its registered
uncertainty requirement, so no new classifier is adopted.

## Forecast

| Catalog peak cohort | Issued at day10: error reduction | Issued at day20: error reduction |
|---|---|---|
| 2024 | 7.0% | 8.7% |
| 2025 | 6.5% | 6.9% |
| Both cohorts | **6.9%** | **8.2%** |

Pooled median object MAE is 0.2603 -> 0.2424 mag at day10 (2,434 objects), and
0.1764 -> 0.1620 at day20 (2,315 objects). The paired 95% gain intervals are
[0.0127, 0.0235] and [0.0108, 0.0178] mag. The existing fixed 50/50 blend and
baseline are both refitted on historical objects; shipped checkpoints are not
scored on objects that they have seen. All methods share identical future
detections. Every annual/horizon aggregate gain is positive.

All four preregistered temporal criteria pass: minimum pooled improvements,
positive intervals, no annual aggregate regression, bounded pooled rare-class
cost, and no substantial point-median regression. The per-class bounds apply
to the **pooled** cohorts, not to every annual rare-class subset.

Tradeoffs remain: pooled AGN error rises 0.3629 -> 0.3871 at day10 and
0.2963 -> 0.3385 at day20; TDE rises 0.5997 -> 0.6182 and 0.3156 -> 0.3378.
The 2024 day10 SLSN subset worsens 0.6150 -> 0.6668 (19 objects), even though
pooled SLSN day10 improves. Pooled empirical coverage of the nominal68% bands
is 65.0%/66.4%; in2024 it is 62.6%/65.6%. Better mean predictions do not
establish calibrated uncertainty for a shifted population or each class.

The cohorts are defined using [BTS catalog peak dates](https://sites.astro.caltech.edu/ztf/bts/explorer_info.html),
expressed as JD−2458000. For each test year, historical peak dates must fall
more than100 days before Jan1. The historical pool splits76/12/12 into
fit/stopping/calibration objects. It contains7,670 objects for the2024 test and
8,812 for2025; the original pre-eligibility test cohorts have1,229 and1,217.

This is a peak-date stress test on the same historically inspected population.
Local event archives contain relative times, without absolute start timestamps,
and historical context/label snapshots are unavailable. A100-day peak-date
buffer does not prove all training observations or labels were available at
that date. This is neither an external test nor a fully prospective backtest.

## Classification

Three equal-weight hierarchies use fixed seeds42,17,83 with independent .30
training context-dropout masks, the same fitting/stopping/test object splits,
and the same stopping inputs. Top and branch probabilities are averaged
separately. The branch threshold is chosen on stopping objects only, using
the previous grid; outer predictions never choose seeds, weights or decoder.

| Metric, full context | Paired baseline | Three-seed ensemble |
|---|---|---|
| Pooled macro F1 | 0.8241 | 0.8322 |
| Accuracy | 0.9226 | 0.9251 |
| Mean fold F1 | 0.8221 | 0.8296 |
| SLSN F1 | 0.5192 | 0.5307 |
| TDE F1 | 0.7874 | 0.8000 |

All six class-F1 point estimates increase, but the paired macro-F1 gain
interval is **[-0.0024, +0.0190]**, so E1 fails. E2 and E3 pass. Removing all
external context worsens macro F1 from0.6250 to0.6167. The promising full-context
point estimate does not justify replacing the calibrated classifier. The
historical paper's0.832 mean is not comparable to the new pooled0.8322;
exact stopping partitions differ, and the new fold mean is0.8296.

## Data availability and artifacts

The local audit found10,497 known-class feature objects plus the existing52
anomaly objects. The2,126 unused core-manifest objects all fail the original
quality criteria. The catalog has10,050 entries outside the known population,
but catalog rows alone do not supply a ready independent photometry/feature
validation set. These counts do not imply that all catalog-only objects are
usable or unseen in every historical study.

Full metrics, confusion counts, cohort membership, model checkpoints, hashes,
prediction replays, plots and source snapshots are in
[`results/robustness_followup_20260917`](../../bts_dataset/results/robustness_followup_20260917/).
The original experiments and final optional forecast weights remain documented
in [the supervised improvements guide](supervised_improvements.md).

```bash
python flare/scripts/robustness_followup.py \
  --source /tmp/flare-supervised-improvements-20260916 \
  --out /tmp/flare-robustness-new-run --threads 4 --workers 3
python flare/scripts/summarize_robustness.py /tmp/flare-robustness-new-run
```

All output directories must be new. Bootstrap intervals use 2,000 paired
object draws and exclude retraining and historical selection uncertainty.
All 34 project tests pass, including the new calendar-buffer boundaries and
hierarchical decoding checks. Saved-model replay verifies classification
probabilities for 200 object/context cases across three members and all three
forecast methods at 1,046 sampled future detections. All baseline classification
labels and all saved mixture calculations are also checked.
