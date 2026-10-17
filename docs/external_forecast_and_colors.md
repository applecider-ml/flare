# External forecast evaluation and the prefix-color correction

An audit found that historical early-feature caches could use a future
observation when computing an early g−r/r−i color. The unified inference API
now rebuilds these color channels from the supplied observed prefix. The
new independent-object evaluation uses this correction and frozen weights.

## The bug and its scope

Truncating rows at day 10 is insufficient when a color stored at day 9.5 was
computed using the other band's detection at day 10.5. The event builder pairs
bands within 1.5 days and historically computed colors before early truncation.
Changing future flux without rebuilding the cached colors did not expose this
failure in the original boundary tests.

The shared `data.color_series()` now discards supplied color values/flags and
reconstructs them from the available rows. This protects both low-level feature
extraction and classification, forecast and PE through the unified analysis API.
A new regression test changes future
photometry, rebuilds the entire event/color cache, and verifies unchanged early
features. Eight real-object/horizon checks verify unchanged forecast outputs.

Of 10,497 known objects, **903 at day 10 and 776 at day 20** had an early color
matched to an observation after the cutoff. Historical early-feature cache
results must therefore not be called fully causal validation. Full-curve
classification comparisons do not have this particular early-cutoff issue.

The original cache files, registered scripts and archived metrics are retained
as historical records. Corrected color-feature tables are separate artifacts.
The shipped forecast weights were trained with the historical caches; **this
work corrects inference and rescores frozen models, not training**. Fully
prefix-causal retraining, early-classification calibration and PE reassessment
remain separate work. No claim is made that the old early calibration is
validated for the changed feature construction.

## Rescoring the same out-of-fold forecast models

The previously evaluated fixed blend retains its aggregate benefit when its
held-out prefixes use corrected colors. Other input features, targets, models,
folds and residual widths are held fixed.

| Horizon | Paired grid | Fixed blend | Reduction | 95% gain interval |
|---|---|---|---|---|
| 10 | 0.2601 mag | 0.2398 mag | 7.82% | [0.0180, 0.0224] mag |
| 20 | 0.1835 mag | 0.1696 mag | 7.57% | [0.0123, 0.0160] mag |

These remain development results on the previously inspected population and
legacy-trained models. The earlier chronological study has not been rescored
here and retains its historical preprocessing limitation.

## Newly downloaded, object-disjoint curves

The 2026-09-17 BTS snapshot contains 20,599 objects. Only 10 classified objects
without local event files have one of the six benchmark types, mostly older
objects; that is insufficient for a new six-class classification evaluation.
The available larger population consists of unclassified objects. Forecast
error can be measured using actual future photometry without class labels.

Before downloading photometry, the experiment froze 200 unclassified ZTF25/26
objects in a deterministic hash order, excluding all local event-file,
feature-table and split IDs. No replacements were made. Public ALeRCE
photometry was retrieved for all 200; raw MJD and JSON are archived. The sample
was not used to refit, tune or recalibrate any model.

Both methods have **all external context absent**. The comparator is the grid
component in the blend bundle, fitted on the same original 7,347 training
objects. This is a new deployment condition, not a replacement estimate for
the labeled, context-rich benchmark. The target is actual future `magpsf`,
at matching times through day 90, in bands with an observed early anchor.

Eligibility: first detection in 2025 or later, at least 90 elapsed days before
the frozen snapshot, and the original full-curve quality cuts (at least 8
detections and 2 in each of g/r within 100 days). Of 200 objects, 125 pass;
56 fail photometric quality and 19 are too recent. A further 3/19 have no
eligible anchored future points at day 10/20, leaving 122/106 scored objects.

| Horizon | Paired grid | Fixed blend | Reduction | 95% gain interval |
|---|---|---|---|---|
| 10 | 0.2194 mag | 0.1929 mag | **12.09%** | **[0.0036, 0.0415]** mag |
| 20 | 0.1824 mag | 0.1704 mag | 6.54% | [-0.0035, 0.0283] mag |

The day 10 gain has a positive interval. Day20 remains uncertain; therefore
**the registered joint confirmation criterion fails**. The minimum sample and
point-median safeguards pass. The failed joint verdict is retained; the sample
is not expanded after inspecting results to obtain a passing interval.

No class labels are available, so this cannot establish rare-class forecast
performance, classification accuracy, AD performance or label-conditional
coverage. Residual-band coverage is descriptive with the original fixed widths.

Sources: [BTS sample documentation](https://sites.astro.caltech.edu/ztf/bts/explorer_info.html)
and [ALeRCE public API](https://science.alerce.online/services/ztf-database-api/).

## Artifacts and reproduction

See [`results/external_forecast_20260917`](../../bts_dataset/results/external_forecast_20260917/)
for the external report, color audit, every eligibility decision, frozen IDs,
model/source/raw-data hashes, corrected OOF predictions and artifact archive.
All 38 project tests pass. The external run first used the correction in the
unified API; it was then centralized in shared feature extraction. Exact
feature equality was verified on all250 eligible object/horizon cases, with
the original evaluation source snapshot retained in `frozen_source/`.

```bash
python flare/scripts/external_forecast.py freeze \
  --catalog snapshot.csv --out /tmp/flare-external-new-run
curl --parallel --parallel-max 4 --config /tmp/flare-external-new-run/downloads.curl
python flare/scripts/external_forecast.py evaluate --out /tmp/flare-external-new-run
python flare/scripts/summarize_external_forecast.py /tmp/flare-external-new-run

python flare/scripts/audit_causal_colors.py \
  --source /tmp/flare-forecast-blend-20260916 \
  --initial /tmp/flare-supervised-improvements-20260916 \
  --out /tmp/flare-colors-new-run
```

Raw data and source snapshots must be restored to the registered state to
replay the original external run. A new catalogue or changed model/code is a
new run; the verifier rejects modified frozen inputs.
