# Supervised improvements against the adopted FLARE recipes

**2026-09-24 prefix-causal retrain of the blend:** both components were retrained on the
feature cache rebuilt from truncated rows (colour leak fixed 2026-09-17), same script,
partitions and gates. B1-B3 hold: day 10 8.22% (0.2602 -> 0.2388 mag, 95% [0.0194, 0.0241]),
day 20 7.62% (0.1837 -> 0.1697, [0.0125, 0.0159]); SLSN cost 0.023 / 0.005 mag. The exported
train-only bundle replaces `models/bts6/forecast_blend/`. The frozen 200-object external sample
rescored with it: day 10 12.4% ([0.0052, 0.0501]), day 20 9.5% ([0.0008, 0.0369]), X1-X3 met
(a rescoring of the same sample, not a new blind test). Results:
`bts_dataset/results/forecast_blend_causal_20260924/` and `external_forecast_rescore_20260924/`.

**2026-09-23 learned embeddings:** [three-seed evaluation](../../bts_dataset/results/embedding_seed_check_20260923/README.md)
finds 8.6% lower h10 median object MAE for a fixed three-seed ensemble against
the stronger matched grid/real-delta recipe (0.2432 -> 0.2224 mag). Each seed
passes the development gate. The [external transfer check](../../bts_dataset/results/external_embedding_forecast_20260923/README.md)
fails: on 122 unclassified, object-disjoint objects with no external context,
error rises 0.1945 -> 0.2273 mag. These are research artifacts, not a new
production default or a general performance claim.

**2026-09-22 follow-up:** [physics and neural benchmark report](../../bts_dataset/results/physics_followup_review_20260922/README.md).
The peak-only PE hybrid improves t0/mB on a disjoint 766-object development
confirmation. A new attentive forecasting model does not beat a causal refit
of the stronger grid/real-delta blend, despite beating its simpler comparator.
These research artifacts do not replace production models.

**2026-09-17 correction:** historical early color caches could use a detection
after the cutoff. Inference now rebuilds colors from the observed prefix.
Frozen-model OOF rescoring retains 7.82%/7.57% aggregate gains; training still
used the historical caches. The original early-cache tables below are retained
as historical results, not fully causal validation. See the
[audit and independent-object evaluation](external_forecast_and_colors.md).

The fixed 50/50 forecast blend passes its registered development criteria.
Classification with complete context does not improve significantly.

The [subsequent robustness follow-up](robustness_followup.md) confirms the
forecast recipe's aggregate benefit in older-to-newer peak-date cohorts
(6.9%/8.2% lower error at day10/20). A fixed three-seed classifier raises the
paired F1 point estimate to0.8322, but its gain interval includes zero and
it is not adopted. These remain development results on familiar objects.

| Forecast issued at | Paired grid baseline | Fixed blend | Reduction |
|---|---|---|---|
| Day 10 | 0.2592 mag | 0.2393 mag | 7.70% |
| Day 20 | 0.1833 mag | 0.1694 mag | 7.57% |

These are median object MAEs on actual future detections, from five folds over
the same historically inspected BTS population. The paired 95% bootstrap
intervals for the gains are [0.0180, 0.0225] and [0.0121, 0.0159] mag.
This is a new gain against paired refits of the adopted forecast recipe, not
the earlier comparison with a Bazin extrapolation or a full-curve classifier.

The tradeoffs remain visible: SLSN error rises by 0.0103/0.0202 mag at day10/20;
TDE improves by 0.0730 at day10 but worsens by 0.0208 at day20. CV worsens by
0.0147 at day20, and SN_CC by 0.0051/0.0011. Passing the preset 0.05 mag rare
class tolerance does not mean improvement for every class. The new models are
available explicitly as `recipe="blend"`; the existing default remains `grid`.

Full results, class/lead tables and the complete reproducibility archive are in
[`bts_dataset/results/supervised_improvements_20260916`](../../bts_dataset/results/supervised_improvements_20260916/).

## Using the optional forecast

```python
import flare

engine = flare.load_analysis()
forecast = engine.forecast(events, 10, context=context, recipe="blend")
# Evaluate exactly at requested future timestamps instead of the display grid:
forecast = engine.forecast(events, 10, context=context,
                           recipe="blend", days=[12.5, 25., 50., 90.])
result = engine.analyze(events, horizon_days=20, forecast_horizon=10,
                        forecast_recipe="blend", context=context)
```

```bash
flare analyze object.npy --horizon 20 --forecast-horizon 10 \
  --forecast-recipe blend --html -o analysis.html
```

Only horizons 10 and 20 are supported by the blend. Future timestamps must be
strictly increasing and lie in `(horizon, 90]`. The baseline component is
interpolated from its anchored grid; the real-detection component is evaluated
directly at each requested timestamp. No predicted class routes the mixture.

The final four models fit original train objects only. A fixed 85/15 carve-out
within train selects boosting rounds, then each model refits the complete
train split. Original validation objects supply empirical 68%/95% residual
widths. Neither validation nor original test objects train these final weights;
both populations have nevertheless been used in historical development.
The intervals are pointwise residual bands, not class-conditional or whole-curve
coverage guarantees. The horizon entry is the last observed magnitude, with
no forecast interval. Model/metadata SHA256 hashes are checked at load time.

## Experiment sequence and verdicts

The original `real_delta` primary improves aggregate error by 12.48%/10.33%,
but its day10 SLSN cost is 0.0772 mag, failing its adoption rule. It is not
silently relabeled as a successful primary. Its fixed `grid_delta` ablation
also cannot replace it after viewing results.

The follow-up in `scripts/forecast_blend.py` records a fixed arithmetic 50/50
mixture before evaluating blend predictions. Both components retrain in five
new partitions (seed2026; inner fit/stop/calibration seed100+fold), on the same
objects. B1 requires >=5%/3% improvements and positive paired lower bounds;
B2 limits SLSN/TDE costs to 0.05 mag; B3 limits point-median cost to 0.01 mag.
All three pass. New partitions do not provide an independent test population.

Classification complete-context macro F1 is 0.8241 -> 0.8243, delta +0.0002,
95% interval [-0.0134, +0.0135]: the primary criterion fails. Complete context
removal improves 0.6250 -> 0.6925 (delta interval [+0.0447, +0.0906]), a useful
secondary stress-test gain, but not superiority over a dedicated LC-only model.
No new classifier is adopted.

## Original registered designs

Entry point: `scripts/improve_supervised.py`. Two fixed primary experiments
test new training recipes against paired refits of the adopted models. The
script and numerical hypotheses were recorded in the prediction ledger before
execution. These are development experiments on the previously inspected
10,497 known BTS objects, not an untouched external validation population.

## Classification

The reference is the full-curve config F hierarchy: 190 LC/context features,
five-way top classifier, binary SLSN branch, existing class weights and
hyperparameters, and stopping-only branch-threshold selection. Five outer folds
use seed 0. Both variants share fitting/stopping/test object partitions.

The baseline masks each external block independently on 30% of its training
rows, as in the adopted recipe. The candidate uses four views of each fitting
object: one complete view and three independently masked at rate 40%. Expected
overall missingness remains 30%. Each view carries one-quarter of the object's
weight; minimum leaf sample count scales by four. No new features are added.
Stopping inputs are identical between methods.

Primary adoption requires at least +0.005 pooled macro F1 with a positive
paired 95% object-bootstrap lower bound, no accuracy loss above 0.002, no class
F1 loss above 0.03, and no absent-context macro-F1 loss above 0.01. Complete
external-context removal is a prespecified secondary stress test, not an
alternative adoption target or a comparison against a dedicated LC-only model.

These are paired recipe refits, not the shipped weight files. The saved
stopping partitions use numerical-label stratification; historical code used
string-label ordering, which changes the exact draws. The paper's 0.832 mean
F1 must not be substituted for the new paired baseline's measured F1.

## Forecast

Horizons 10 and 20, five paired object folds with seed 42; each outer train
splits 76/12/12 into fitting/stopping/calibration objects. No observation or
augmented view of a test object enters fitting or stopping.

- `baseline`: adopted tuned pooled L1 model, original interpolated grid
  targets, original anchor when interpolating forecasts to real detection times.
- `real_delta`, the fixed primary: train directly on actual future detections
  through day 90, predict change from the last observed band magnitude, and give
  each fitting object equal total weight. Add four band-local early inputs:
  last magnitude, age of last observation, last-five-point slope and scatter.
- `grid_delta`, a fixed control: same delta target and early recency features,
  trained on the original grid with uniform row weights.

All early features and centering use observations at or before the horizon.
The new anchor matches the original cached last magnitude for every object.
Only the target uses later detections. All methods are scored on identical
eligible observed detections, never on their own selected subsets.

The primary metric is mean absolute detection error for each object, then the
median over objects. Adoption requires at least 5% reduction at day 10 and 3%
at day 20, with positive paired-bootstrap gain lower bounds at both horizons.
SLSN/TDE median object errors may worsen by at most 0.05 mag, and global
point-median error by at most 0.01 mag. Day-50 grid errors and lead-bin results
are secondary. An ablation cannot replace the primary after seeing outcomes.

Plain residual intervals are separately calibrated for each model on real
calibration detections. They are descriptive controls, not the original
adaptive forecast bands and not a claim to fix rare-class undercoverage.

## Reproduction and verification

From the workspace parent:

```bash
python flare/scripts/improve_supervised.py \
  --out /tmp/flare-supervised-new-run --threads 4 --workers 3
python flare/scripts/summarize_supervised.py /tmp/flare-supervised-new-run
python flare/scripts/forecast_blend.py --source /tmp/flare-supervised-new-run \
  --out /tmp/flare-blend-new-run --threads 4 --workers 3
python flare/scripts/summarize_forecast_blend.py /tmp/flare-blend-new-run
python flare/scripts/verify_forecast_blend.py /tmp/flare-blend-new-run
python flare/scripts/export_forecast_blend.py --evaluation /tmp/flare-blend-new-run \
  --out /tmp/flare-blend-final-new-run --threads 4
python flare/scripts/verify_forecast_export.py /tmp/flare-blend-final-new-run
```

The output directory must be new. Local scratch avoids the project filesystem's
known inode-quota problem. Results and a complete artifact archive are copied
back to the project after verification. Three independent fold jobs run at
once; each uses four CPU threads. No GPU or new dependency is needed.

The tests check cutoff invariance, exact-timestamp ties, preservation of the
original context view, missingness, per-object weighting and target alignment.
All 32 project tests pass, including the optional inference integration's
artifact-integrity and exact-timestamp replay checks. Final inference also
reproduces 1,077 calibration-point predictions to floating-point precision;
raw-input feature agreement is checked on 38 object/horizon cases.
The summarizer uses
2,000 paired bootstrap draws over unique objects, class-stratified for F1.
Intervals are conditional on fitted folds and exclude retraining uncertainty
and the effect of historical reuse of the benchmark.
