# Unified FLARE: measured follow-up

The API, local-file CLI and HTML report now combine classification, AD,
observed-span GP reconstruction, future forecasts and conditional SN Ia PE.
See [usage](unified_analysis.md) and [a real-object report](unified_example.html).

The development benchmark finished all 5 folds at days 5, 10 and 20 on 10,497
known-class objects. A shared object partition is used for every task, with
separate fitting, early stopping, calibration and test sets. The script's
predictions and hash were recorded before execution in the existing ledger.
This is further development on a previously inspected sample, not an external
generalization test.

| Day | Full-trained model on early inputs, macro F1 | Horizon-trained macro F1 | Horizon-trained conformal coverage |
|---|---|---|---|
| 5 | 0.415 | 0.633 | 0.904 |
| 10 | 0.470 | 0.680 | 0.901 |
| 20 | 0.549 | 0.768 | 0.901 |

**Adopted:** separate early classifiers and calibrators at days 5, 10 and 20,
automatically selected by the offline analysis API. The baseline above is a
separately refitted full-curve model receiving truncated inputs, not the
historical early-time baseline or the headline full-curve F1 of 0.832. These
numbers demonstrate why matching the training horizon is necessary; they do
not establish a new state of the art against the earlier horizon-specific
experiments. Final exported early weights train/early-stop on original train
and calibrate on original validation; their test labels are not used.

**PE:** at day 10 the singleton-Ia gate has 95.4% purity and retains 48.2% of
true Ia. Among gated true Ia with usable SALT3 labels, median peak-time error
is 0.968 d and peak-B magnitude error is 0.114 mag. The full Ia population
has errors 1.103 d and 0.130 mag. This is a selection effect, not a better
regressor. About 4.6% of selected objects are non-Ia and have no comparable
SALT3 truth. Stretch remains weak and its gated 68% interval coverage at day
10 is 0.657. Existing PE weights are retained with these applicability limits.

**Forecast:** grouping interval calibration by lead and observed count failed
to fix rare-class undercoverage. At day 5, SLSN pointwise 68% coverage changed
0.551 -> 0.545 and TDE 0.608 -> 0.611. Keep the original forecast models and
state the limitation alongside the band. Reconstruction uses the original GP
recipe and empirical calibration; no new interpolation gain is claimed.

**AD:** at day 10, on novel objects with sufficient later observations,
energy flags 1.2%, residual-only flags 4.3%, and the fixed combination flags
3.5%; the combination flags 0.9% of eligible known objects. Residuals require
future detections through day 50. The gain fails the predeclared 5 percentage
point criterion, so no new anomaly scorer is deployed. These 51 eligible novel
objects are averaged across repeated fold evaluations, not counted as 255
independent examples. Early use of the shipped AD calibration remains marked
as unvalidated.

Registered outcomes: U1/U2 hold; U3/U4 fail. Full per-class results, interval
widths, paired F1 bootstrap intervals, PE selection counts and definitions:
[benchmark summary](../../bts_dataset/results/unified_20260915/summary.md).
The same directory holds per-object OOF CSVs, exact fold assignments,
provenance and a completion marker.

Validation: eight regression tests passed, all 45 forecast/PE assets match
their manifest, nine object/horizon feature extractions match cached LC/context
features (photo-z numerical integration differs by <0.000002 mag), and a built
wheel includes all early-classifier, forecast and PE assets. Installed-package
inference is checked independently of the source checkout.
