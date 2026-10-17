# Self-supervised learning for blind AD

Research experiment: `scripts/ssl_ad.py`. It learns a representation from the
existing BTS training light curves, then compares anomaly scores on embeddings,
FLARE features, and their combination. It does not change the production AD
scorer or import PyTorch into the normal inference path.

## Fixed design

The original 7,347 training objects supply an 85% fitting / 15% early-stopping
split. The original 1,575 validation objects are used only to calibrate the 1%
false-alarm operating point. The 1,575 test objects and 52 out-of-taxonomy
objects never enter pretraining, fitted preprocessing or checkpoint selection.
The three seeds (17, 42, 83) share the same object partitions.

This is a local pretraining experiment. It neither uses nor evaluates external
million-object pretrained weights. The anomalies were inspected during earlier
research, so these results remain development evidence. No fresh temporal
holdout is available in this experiment.

The encoder consumes up to 96 real observations through day 100, with actual
times, centered magnitudes, errors and bands. No interpolated regular grid is
used. Long curves are deterministically subsampled by observation index,
including their first and last observations. A two-layer Transformer with a
48-dimensional representation learns from masked magnitudes and consistency
between two independently thinned, noise-perturbed views. It receives no class
loss. Selection uses only a fixed self-supervised early-stopping loss.

The fixed primary score is **kNN distance on equal-weight blocks of FLARE
features and learned embeddings**. Controls are energy, tabular-only kNN,
embedding-only kNN, embedding Isolation Forest and an untrained encoder.
Imputation/scaling and neighbor reference sets use fitting objects only.
Every rule is calibrated independently using known validation objects.

The numerical hypotheses and adoption rule were recorded in the script
docstring and `bts_dataset/results/registered_predictions_ledger.txt` before
running. Primary adoption requires all of:

* At least +0.05 mean novel recall at nominal 1% FAR over paired energy, with
  positive lower endpoint of a paired 95% object-bootstrap interval.
* Measured known test FAR at most 0.02 for each seed.
* At least +0.05 mean recall over tabular-only kNN, demonstrating that the
  embeddings add information to the existing feature-distance baseline.

The change in discoveries among the first 100 reviewed objects is a separate
secondary measure. The script does not choose an alternative winner after
seeing anomaly outcomes. It reports individual families and known-class false
alarms, since aggregate gains may hide poor behavior on rare known classes.

## Completed experiment: 2026-09-16

**The registered adoption criterion failed.** All three seeds completed. The
primary hybrid detected 0/52 novel objects in every seed at the independently
calibrated nominal 1% level, with 30, 30 and 31 false alarms among 1,575 known
test objects (1.90–1.97%). The paired energy baseline detected 3, 4 and 3 novel
objects, with 15, 18 and 15 false alarms (0.95–1.14%). Energy here is a refit
using the same fitting/early-stopping objects and existing model-card
parameters, not the shipped production checkpoint.

Mean recall difference versus energy was **−6.41 percentage points**, with
paired 95% object-bootstrap interval **[−11.54, −1.92] points**. The hybrid
added no recall over tabular-only kNN and found 6 novel objects among the first
100 reviewed, versus 12 on average for energy. S1, S3 and S4 failed; only S2
(each seed's observed FAR at most 2%) held. Its higher overall AUROC (0.810
versus 0.759 for energy) did not translate into better detection at the chosen
false-alarm level.

Embedding-only kNN reached 1.28% mean recall and embedding Isolation Forest
5.13%, versus 6.41% for energy. These controls do not justify changing the
preselected primary or deploying this encoder. Production AD remains unchanged.
This result concerns this local training corpus and design; large-corpus
pretraining remains untested.

The [full report](../../bts_dataset/results/ssl_ad_20260916/summary.md) contains
per-seed counts, family breakdowns, confidence intervals and all six controls.
Machine-readable results are in
[summary.json](../../bts_dataset/results/ssl_ad_20260916/summary.json) and
[metrics.csv](../../bts_dataset/results/ssl_ad_20260916/metrics.csv).

![Three-seed AD comparison](../../bts_dataset/results/ssl_ad_20260916/ssl_ad_comparison.png)

## Run

From the parent directory, using an environment with the normal FLARE
dependencies and PyTorch (`pip install -e './flare[research]'` if needed):

```bash
python flare/scripts/ssl_ad.py \
  --data bts_dataset --out bts_dataset/results/ssl_ad_new_run --threads 8
python flare/scripts/summarize_ssl_ad.py bts_dataset/results/ssl_ad_new_run
```

The output directory must be new. The default is 30 epochs maximum with
patience 7 and three seeds. Everything runs on CPU. The existing experiment
uses `bts_dataset/results/ssl_ad_20260916/`.

Artifacts include input hashes, exact object partitions, feature order,
preprocessing parameters, encoder checkpoints, reference embeddings,
calibration and evaluation scores, energy weights and training curves. A
completion marker is written only after all requested seeds finish. The
summary script produces metrics, family tables, a figure and the registered
hypothesis verdicts. Repeated seeds are averaged per novel object in the
bootstrap, never counted as additional independent discoveries.

For reproducibility, a separate research command can replay the fixed primary
score from a completed seed, without changing normal FLARE inference:

```bash
python flare/scripts/score_ssl_ad.py \
  --run bts_dataset/results/ssl_ad_20260916 --seed 17 object.npz
```

Optional `--context` uses a JSON mapping from file stem to external context.
The result is explicitly experimental. Its novelty p-value describes rank
among known calibration objects; it is not an anomaly probability. Early-time
operation has not been evaluated.

## Tests

```bash
cd flare
python -m unittest discover -s tests -p test_ssl_ad.py -v
```

Checks cover label exclusion, irregular-time preservation, correct masking,
padding-invariant embeddings, finite training gradients, fitting-only
preprocessing, finite-sample calibration ranks and object-level bootstrap
invariance to duplicated seeds. These tests need the
optional PyTorch dependency; normal FLARE inference does not.
