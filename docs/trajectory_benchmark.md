# Trajectory sets and budgeted continuation

This opt-in research workflow combines finite-rank trajectory class sets, a causal
checkpoint allocator, and small learned continuation rankings. It exposes complete
policy grids, per-object decisions and action ledgers, class-specific errors, emitted
label precision, and realized resource use. The [empirical report](trajectory_benchmark/results.md)
contains both useful scheduling results and counterexamples. Shipped classifiers,
calibration files, and service decisions are unchanged.

## Reproduce from the public BTS benchmark

Install the package and development checks. Matplotlib is optional for the figure.
Run these commands from the repository root, choosing writable directories outside
the checkout for raw data, caches, models and action arrays:

```bash
python -m pip install -e '.[figures]' pytest
export FLARE_PAPER_DATA="$HOME/flare-benchmark-data"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/setup_data.py
python scripts/prepare_trajectory_benchmark.py features \
  --data-root "$FLARE_PAPER_DATA" --out "$HOME/flare-prefix-cache" --workers 2
python scripts/prepare_trajectory_benchmark.py train \
  --cache "$HOME/flare-prefix-cache" --out "$HOME/flare-folds" --threads 2
python scripts/run_trajectory_benchmark.py run \
  --input "$HOME/flare-folds" --out "$HOME/flare-replay" --rare-classes SLSN TDE
python scripts/plot_trajectory_benchmark.py \
  --results "$HOME/flare-replay/results.json" --out "$HOME/flare-frontier.png"
python -m pytest tests -q
```

Every output directory must be new. `setup_data.py` uses the benchmark tables and
light curves bundled with FLARE; no unpublished models, prediction archives,
network service, protocol file, or research checkout is needed. The feature command
reads only train/validation objects. It uses the inference path
`feature_frame(observed_events(load_events(path), horizon), context)` at days
5/10/20/100, recomputing colors and peak-dependent features from each observed prefix.
Ineligible rows cause an explicit error; horizons cannot silently use different objects.

Training preserves train-then-validation row order, the five-fold `KFold` seed
20261102, next-fold calibration, and the remaining three folds as the proper-training
pool. A stratified 85/15 fit/early-stop split uses seed `100 + fold`. All four
classifiers use the existing `train_early_cc.hierarchy_nocal` and `evaluate_unified.fit`
recipe, 2,000 maximum rounds, and the same ordered features. Early-stop,
calibration and evaluation predictions are saved directly. The continuation-training
rows are classifier early-stop objects: they belong to proper training and influenced
classifier fitting, so their scores are **not untouched out-of-fold predictions**.
`--folds 0` can check one complete four-horizon fold; omitting it fits all twenty
classifiers. `--rounds` supports development smoke runs, but changing it changes the study.

A small synthetic end-to-end example requires no light curves or model fitting:

```bash
python scripts/run_trajectory_benchmark.py synthetic --out "$HOME/flare-synthetic"
python scripts/run_trajectory_benchmark.py run --input "$HOME/flare-synthetic" \
  --out "$HOME/flare-synthetic-results" --rare-classes rare
```

The runner's `--mode rare` or `--mode budget` runs the complete selected grid.
Default `all` runs both. Evaluation truth enters summaries only; calibration,
continuation fitting and scheduling do not receive evaluation labels.

## Portable fold archive schema

An input directory contains `fold*.npz`, loaded with `allow_pickle=False`. All folds
must use identical class and schedule order. IDs are nonempty Unicode strings,
unique within each role; training/calibration/evaluation roles are disjoint within
a fold. Evaluation IDs are unique across folds, and repeated objects have consistent
truth labels. Fold training/calibration populations may overlap across refits.

| Field | Shape / meaning |
|---|---|
| `schema_version` | Scalar integer `1` |
| `class_names` | `(K,)` unique nonempty Unicode names in probability-column order, `K >= 2` |
| `horizons` | `(T,)` finite positive strictly increasing frozen times |
| `training_score_role` | Scalar `classifier_early_stop`, `held_out_training`, or `synthetic` |
| `{role}_ids` | `(N_role,)`, for `role` = `training`, `calibration`, `evaluation` |
| `{role}_probabilities` | `(N_role,T,K)`, finite values in `[0,1]`, rows summing to one |
| `{role}_nobs` | `(N_role,T)`, finite nonnegative nondecreasing cumulative observation counts |
| `{role}_y` | `(N_role,)` integer dataset truth in `[0,K)`, never prediction labels |
| `classifier_fit_ids` | Optional gradient-fitting membership, unique and disjoint from all three prediction roles |

`training` denotes continuation-training examples, which may be early-stop rows;
`classifier_fit_ids` is the separate gradient-fitting subset. This distinction is
recorded rather than inferred from a filename. The public preparation command also
saves feature names, memberships, model hashes, branch thresholds, seeds, and source
hashes. `--rare-classes` explicitly selects the indices used by the rare-mass baseline.

Outputs include complete fold/pooled `results.json`, its `results.json.gz` copy,
`COMPLETE` containing the uncompressed JSON SHA-256, all-cell CSVs and class counts.
The CSVs contain every fold cell plus each pooled cell, including sufficient class
counts to recover error and precision rates. Full JSON retains confusion matrices,
retention rates, class-specific resources and all other diagnostics. Candidate paths,
thresholds, training targets, fitted regressions, per-object decisions and every
query action remain in the run directory. Pickled models are local outputs.
Strict JSON encodes infinite calibration
thresholds as `null` with an explicit positive-infinity mask.

## APIs and statistical meaning

- `flare.trajectory_sets`: `calibrate_sets(..., class_names=..., horizons=...)`,
  `nested_candidate_sets(prefix, calibration)`, `first_singleton(..., horizons=...)`,
  `summarize_sets(..., class_names=...)`, and calibration JSON round-tripping.
- `flare.checkpoint_budget`: `allocate_query_budget(...)`, including seeded random,
  five heuristics and a custom prefix-only scorer. A custom callback receives copied,
  read-only histories in `PrefixObservation`; deterministic ties use object ID.
- `flare.continuation`: prefix features, future-training closure targets,
  `fit_continuation_models(..., class_names=..., horizons=...)`, and `continuation_scorer`.
- `flare.trajectory_benchmark`: schema validation, complete-grid execution and count pooling.

For a frozen schedule and frozen scorers, define each object's true-label score
`S = max_t(1-p_t(Y))`. The finite-rank threshold uses
`ceil((n+1)*(1-alpha))`, with infinity when the requested rank exceeds the sample
size; ties remain included. Under exchangeability of complete calibration/query
trajectories this gives ordinary marginal split-conformal inclusion. Calibrating
within true class gives the Mondrian statement under within-class exchangeability.
A prefix maximum cannot exceed the complete-trajectory maximum, so the nested
prefix set contains the complete-trajectory set. Thus any wrong singleton implies
true-class exclusion somewhere on the frozen path. This is an implementation of
established conformal constructions, not a new theorem. See
[Vovk et al., *Algorithmic Learning in a Random World* (2005)](https://doi.org/10.1007/b106715)
and [Angelopoulos and Bates (2023)](https://doi.org/10.1561/2200000101).

These statements are marginal over calibration/query sampling, not conditional on
the realized calibration table. They require the same trajectory/eligibility rule
and frozen horizon count. A shorter query prefix is supported without reallocating
Bonferroni alpha or extending the schedule. Separate full-alpha fixed-look controls
are valid at their prespecified deadline, not for adaptive choice among deadlines.
No assertion concerns lifetime anomaly false alarms or emitted-label precision.

All objects start with one common initial checkpoint. Each extra token reveals one
object's next **archived checkpoint**, independent of its future observation count.
This is an archive-query cost, not a telescope observation or acquisition intervention.
The allocator closes first singletons automatically, empty sets as referrals, and
terminal ambiguous sets as exhausted. Fixed-deadline diagnostics instead charge
unresolved objects through their deadline, including empty sets. Realized `nobs`
and consumed days are measured after revelation.

Continuation targets inspect future **training** paths only. Each eligible object's
row weights sum to one. Aggregate reward counts correct future closures; macro reward
weights class `c` by `N/(K*n_c)`, with frequencies from all training objects. Ridge
reward/cost pairs use fixed alpha 1; the ranking is nonnegative predicted reward
per clipped predicted query cost. Outputs are utility rankings, not calibrated probabilities.
The quality variant adds current cumulative observations and previous-look probability
and label changes to the same known time basis.

Metrics distinguish wrong/all within a true class, wrong/selected within that class,
and correct/all emissions of a predicted label. Zero denominators are `None`.
Pool counts before calculating rates. Empirical macro correct/wrong rates average
only represented true classes and report the included names and count. Random-seed
ranges reflect algorithmic variation, not confidence intervals.
