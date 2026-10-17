# Class-conditional predictive anomaly detection

Research entry point: `scripts/predictive_ad.py`. This tests whether every known
class fails to predict withheld observations, relative to a separate reference
model. It is independent of production FLARE inference.

## Protocol

The experiment uses original BTS objects, up to day 100. Three contiguous
intervals partition each observed time span. Each interval is predicted from
the remaining observations. Hidden magnitudes never enter context summaries,
centering, colors or model inputs. Times, filters and reported uncertainties
are conditioning variables. This is blocked interpolation on full curves;
causal early-alert performance is not evaluated.

A small conditional neural process encodes actual observations, pools context
and attends to it at each query time. A shared decoder receives a candidate
class and predicts Student-t location and scale. Known labels condition the
decoder only during training. At evaluation, all known classes are tried and
the same candidate class must explain all withheld blocks of an object.

A separate reference network receives per-band magnitude moments and counts,
context time limits, and query time/filter/error. It cannot access the temporal
encoder. This is a fitted nuisance reference, not a physical model or a
guarantee of invariance to cadence. The primary score compares predictive NLL
with reference NLL. Photometric error enters the Student-t scale in quadrature;
this is an approximate observation model, not an exact convolution.

Before reading final validation/test/novel curves, six leave-one-class-out
experiments use **original training objects only**. Fitting, early stopping,
calibration and the excluded class are disjoint. These select between mean
and upper-quartile aggregation of pointwise NLL differences, with equal block
weights and macro averaging of recall across excluded classes. Ties choose the
mean. The class choice is made after aggregation across blocks. Selection is
saved before final scoring. LOCO results are development selection evidence.

Final fits use the original 6,244 fit / 1,103 stopping objects and three seeds
(17, 42, 83). The 1,575 validation objects calibrate the nominal 1% alarm level.
The 1,575 known test objects and 52 familiar novelties evaluate the locked rule.
Energy controls reuse the verified paired LightGBM fits from the SSL experiment,
with identical object partitions and inputs. These are not production weights.
Raw best-class NLL and reference-only NLL are controls, not fallback candidates.

## Registered decision

The exact script hash and predictions are appended to the experiment ledger
before fitting. Primary adoption requires all of:

- At least +5 percentage points mean recall over paired energy, with positive
  lower endpoint of the paired 95% object-bootstrap interval.
- Observed known test FAR no greater than 2% for every seed, at thresholds
  calibrated for nominal 1% on validation.
- At least +5 percentage points recall over the reference-only control.

At least five extra discoveries in the first 100 reviewed is a separate
secondary hypothesis. Repeated seeds are averaged per object in bootstraps.
Missing predictions are reported and count as misses for all-object recall.
The existing benchmark has been examined in previous experiments: even a
positive result needs confirmation on a fresh temporal/external population.

## Completed experiment: 2026-09-16

**The registered adoption criterion failed.** Six training-only LOCO fits
selected mean aggregation (macro recall 0.0343 versus 0.0312 for the tail rule).
All three final seeds completed, with predictions for all 1,575 known test
objects and all 52 novelties.

| Method | Mean novel recall at nominal 1% | Measured known FAR | Novel in first 100 |
|---|---:|---:|---:|
| Paired energy refit | 6.41% | 1.02% | 12.0 |
| Reference-only NLL | 7.05% | 0.68% | 15.7 |
| Best-class NLL | 1.92% | 0.47% | 15.7 |
| Selected predictive ratio | 3.21% | 0.80% | 5.0 |

The primary detected 2, 2 and 1 novel objects, with 12, 14 and 12 false alarms.
Energy detected 3, 4 and 3. The paired recall difference is **−3.21 percentage
points**, with 95% object-bootstrap interval **[−8.97, +1.92] points**. P1, P3
and P4 failed; P2 held. The interval includes zero, so this does not establish
a statistically resolved decrease either. It does establish that the required
improvement was not demonstrated.

The reference-only control's small recall difference and better top100 ranking
are exploratory observations, not a replacement selection rule. The ratio's
AUROC was 0.416 versus 0.759 for energy: reference subtraction did not isolate
useful novelty in this implementation. No score direction was reversed and no
post-hoc combination was adopted. Production AD remains unchanged.

Artifacts: [full report](../../bts_dataset/results/predictive_ad_20260916/summary.md),
[metrics](../../bts_dataset/results/predictive_ad_20260916/metrics.csv),
[registered verdict](../../bts_dataset/results/predictive_ad_20260916/summary.json),
[verification](../../bts_dataset/results/predictive_ad_20260916/verification.json).
Twenty tests passed, and independent checkpoint replay matched scores and
calibration ranks for a known and a novel object in each seed.

![Predictive AD comparison](../../bts_dataset/results/predictive_ad_20260916/predictive_ad_comparison.png)

A project inode-quota error interrupted output after the second seed had saved
its completed checkpoint. Scoring resumed from that checkpoint, followed by
the third seed, using the unchanged registered functions and locked selection.
The recovery driver and hashes are saved in the result directory. The disposable
FLARE build tree was relocated to `/tmp/flare-build-quota-backup-20260916`.

## Reproduce

From the parent workspace, with FLARE's optional research dependencies:

```bash
python flare/scripts/predictive_ad.py \
  --out bts_dataset/results/predictive_ad_new_run --threads 4
python flare/scripts/summarize_predictive_ad.py \
  bts_dataset/results/predictive_ad_new_run
```

Output must be a new directory. The default budget is 40 epochs with patience 6
per fit, six LOCO fits and three final fits. Checkpoints, exact partitions,
input hashes, selection, calibration scores and training histories are saved.

Replay a completed seed on an object's local event file:

```bash
python flare/scripts/score_predictive_ad.py \
  --run bts_dataset/results/predictive_ad_20260916 --seed 17 object.npz
```

The output is explicitly experimental. Its novelty p-value is a rank among
known calibration objects, not the probability that an object is anomalous.
Objects without a valid context/target block return an unavailable result.

Boundary checks are in `tests/test_predictive_ad.py`, including hidden-target
exclusion, same-time grouping, padding invariance, finite optimization and
class-balanced LOCO rule selection.

## Motivation

The local SSL representation experiment failed its adoption rule. This tests
a different objective: class-conditional predictive compatibility. It also
differs from the earlier global forecast-residual combination, which did not
compare every class against a nuisance reference.

Relevant background: [ParSNIP](https://arxiv.org/abs/2109.13999),
[predictive transient AD](https://academic.oup.com/mnras/article/517/1/393/6705438)
and [likelihood-ratio OOD detection](https://arxiv.org/abs/1906.02845).
This implementation is a local research design, not a reproduction of those
papers and not evidence that their results transfer to BTS.
