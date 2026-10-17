# Offline analysis: classification, AD, reconstruction, forecast and Ia PE

The Python API and `flare analyze` read local FLARE event arrays. They do not
fetch photometry or context. The existing `predict`/`report` commands retain
their online behavior. All outputs in this new interface refer to an explicit
observation cutoff. At days 5, 10 and 20, classification selects a matching
early-time hierarchy and its separate calibration. AD still uses the full-curve
model and explicitly marks its early-time calibration as unvalidated.

Early color features are recomputed from the observed prefix, including when
the supplied arrays contain colors computed from a longer curve. A September
17 audit found that historical early training/evaluation caches could include
a future counterpart in an early color. Current weights retain that training
history; early calibration after the preprocessing correction needs separate
reassessment. See [the audit and external forecast test](external_forecast_and_colors.md).

```python
import flare
from flare.data import load_events

engine = flare.load_analysis()
result = engine.analyze(
    load_events("object.npz"),
    horizon_days=10,
    forecast_horizon=10,
    pe_horizon=10,
    context={"z_phot": 0.08},  # other external columns are optional
)
```

```bash
flare analyze object.npz --horizon 10 --forecast-horizon 10 --pe-horizon 10 -o result.json
flare analyze object.npz --horizon 10 --forecast-horizon 10 --pe-horizon 10 --html -o report.html
```

For multiple objects, `--context context.json` accepts a mapping from each
file's stem to its external context. Omitted context remains missing. Passing
only a precomputed `M_pseudo` is insufficient: provide `z_phot`, so the
brightness reference can be recomputed at each requested horizon. External
context cannot overwrite light-curve features.

## Outputs and applicability

* `classification`: probabilities, decoded class, conformal candidates and
  calibration scope. At days 5, 10 and 20 a matching early classifier is used;
  at other cutoffs the full-curve model is used, with its guarantee and measured
  coverage cleared when the cutoff does not match its training horizon.
* `anomaly`: energy, novelty p-value, likelihood ratio and anomaly probability
  with the assumed prevalence. `--base-rate` must lie strictly between 0 and 1.
* `reconstruction`: a Matérn-3/2 GP fitted only within each observed g/r span.
  Requires three distinct epochs per band. Predictive standard deviations and
  empirical 68%/95% intervals are returned. Multipliers 1.211/3.222 come from
  the original 412-object study; early-time and per-class coverage are not
  established. They are not simultaneous curve-wide confidence bands.
* `forecast`: saved model at exactly day 5, 10, 15, 20 or 30, with g/r means and
  intervals through day 90. No automatic horizon rounding or extrapolation.
  The value at the model horizon is anchored to the last observation in each
  band. Missing anchor bands are reported as insufficient data. The reference
  magnitude is the brightest detection available by the model horizon.
  At days 10 and 20, `forecast_recipe="blend"` (CLI `--forecast-recipe blend`)
  selects the [fixed 50/50 blend](supervised_improvements.md), with 7.7%/7.6%
  lower median object error in paired development evaluation. Its intervals
  are empirical pointwise residual bands with no class-conditional guarantee.
  The original `grid` recipe remains the default. For arbitrary future times,
  use `engine.forecast(events, 10, recipe="blend", days=[12.5, 25., 50.])`.
* `pe`: SALT3 t0, x1, c and mB estimates at day 5, 10 or 20, conditional on Ia.
  The unified API abstains unless the conformal set is exactly `{SN_Ia}` at
  the **PE horizon**. This gate is a conservative product rule, not a validated
  gate purity guarantee. PE intervals were calibrated on true,
  quality-selected Ia and approximate full-curve SALT3 fits. Weak early stretch
  constraints are retained explicitly in the metadata.

Choose model horizons no later than the observation cutoff. A forecast from
day 10 with a day-17 observation cutoff uses detections through day 10, and is
labelled accordingly. Both forecast and PE are disabled unless their horizons
are selected. `--no-reconstruction` disables the GP fit.

For a confirmed Ia, `engine.pe(events, 10, context=...)` explicitly requests an
Ia-conditional estimate without the classifier gate. `engine.forecast(...)`
and `engine.verify_models()` are also available independently. MD5 checks
against the bundled manifest detect inconsistent forecast/PE assets before
loading; the manifest is not a cryptographic authenticity signature.

The early classifiers can also be loaded with
`flare.load_classifier('bts6', horizon_days=10)`. They expose classification and
conformal prediction; they do not contain a calibrated anomaly probability.
They live under `models/bts6/early/h{5,10,20}/`, with SHA256 manifests, and are
trained by `scripts/train_early.py`. Fitting/early stopping use only the original
train split, and calibration uses only the original validation split. Passing
an explicit classifier to `FlareAnalysis` disables automatic classifier selection.

## Development benchmark

`scripts/evaluate_unified.py` evaluates all four tasks with shared five-fold
object partitions and separate fit, early-stopping, calibration and test sets.
Its docstring fixes the comparisons and numerical hypotheses. Run from the
repository parent with the existing BTS caches:

```bash
python flare/scripts/evaluate_unified.py \
  --data bts_dataset --out bts_dataset/results/unified_new_run --threads 2
```

The output directory must not already exist. It contains the script hash and
arguments, exact split indices with object IDs, per-fold metrics, classification,
PE and AD predictions, and a completion marker only after all requested folds
finish. The forecaster is retrained on each fold; errors are evaluated against
real later detections. Fixed lead/cadence groups are calibrated independently
of test labels. Forecast residual AD becomes available only after subsequent
observations (here through day 50); energy is available at the early horizon.

These are development comparisons on repeatedly inspected BTS data. New
generalization claims require a fresh population. Repeated evaluations of the
same 52 novel objects across folds are not independent anomaly samples. No
experimental model replaces the deployed weights automatically.

## Verification

```bash
cd flare
python -m unittest discover -s tests -v
```

The regression suite covers cutoff isolation, saved feature ordering, direct
booster parity, context handling, manifest mismatch, Ia abstention, JSON
serialization, HTML escaping, invalid priors/horizons and GP interpolation span.
