# FLARE

**Feature-based Light-curve Aggregated Ranking Ensemble**: a photometric classifier for ZTF
transients with calibrated uncertainty, an anomaly layer for what lies outside its taxonomy, and a
forecast of the future light curve from its first days.
FLARE computes 160 physically motivated features from each light curve, joins the catalogue
context that is available when an alert is issued (Pan-STARRS host photometry, a Legacy Surveys
photometric redshift, Gaia DR3 and AllWISE, the Pan-STARRS pre-outburst counterpart), and
classifies with gradient-boosted trees. It needs no GPU and trains in minutes.

Project page with live demos: https://applecider-ml.github.io/flare/

## Install

```bash
pip install -e .                 # add [consensus] for the blind-consensus anomaly alarm (LUNA)
```

## Quick start

```python
import flare
clf = flare.load_classifier()                        # six classes: SN Ia, SN CC, SLSN, AGN, TDE, CV
probs = clf.predict_proba_from_files(["ZTF21abcdxyz.npz"])
sets  = clf.prediction_sets_from_files(["ZTF21abcdxyz.npz"])   # 90% conformal sets
early = flare.load_classifier(horizon_days=5)        # models trained on the first 5, 10 or 20 days
```

```bash
flare predict ZTF19acbzgog                           # JSON for one object (BOOM if configured, else ALeRCE)
flare report ZTF19acbzgog -o report.html             # a self-contained report page
flare analyze events/*.npz --html                    # classification, anomaly score, forecast, SN Ia PE
flare analyze events/*.npz --horizon 5 --forecast-horizon 5 --pe-horizon 5 --next-obs-horizon 5
```

A light curve needs at least eight detections, with at least two in each of g and r, for the
classifier and the forecast; the i band is used where present but is not required.

External context is optional: the models are trained with each block randomly removed, so pass
NaN for anything that is missing and never impute it.

## Performance

Five-fold cross-validation on 10,497 spectroscopically confirmed BTS transients:

| inputs | accuracy | balanced accuracy | macro F1 |
|---|---|---|---|
| light curve only | 90% | 71% | 0.72 |
| + host, M_pseudo, Gaia/WISE, PS1 counterpart | 92% | 83% | 0.83 |

Per-class F1: SN Ia 0.96, SN CC 0.85, SLSN 0.53, AGN 0.90, TDE 0.81, CV 0.94. The shipped models
for the first 5, 10 and 20 days are trained on the BTS benchmark together with the spectroscopically
confirmed Fritz sources, and reach an out-of-fold macro F1 of 0.66, 0.71 and 0.77 on that pool of
11,113 objects (`models/bts6/early/report.json`).

## Uncertainty and anomalies

- **Prediction sets:** class-conditional conformal sets at 90%, calibrated cross-conformally;
  every output also reports the measured coverage of the predicted class, the credibility and
  the confidence.
- **Probability intervals:** a Venn-Abers interval for every class probability
  (`flare/venn_abers.py`), wider for rare classes and for objects unlike the training data.
- **Anomaly alarm:** the mean percentile rank of each object over fourteen LUNA scorers
  (`flare/consensus.py`), with a conformal p-value and alarms at false-alarm rates of 1%, 5%
  and 10%, calibrated separately for every model. Without `luna-ood`, the anomaly energy (the
  log-sum-exp of the tree scores) gives the p-value and the alarms.

- **Lifetime false-alarm rate:** a source on Fritz is analysed again at every new detection, so
  `flare/lifetime.py` turns the smallest per-analysis p-value so far into a false-alarm rate per
  unit time, calibrated by replaying the known test objects detection by detection
  (`models/bts6/lifetime_far.json`), with levels of one false alarm per week, month and quarter.
- **Review priority:** once operators have confirmed members of several new families,
  `flare/review.py` combines the anomaly scorers into a supervised review ranking
  (`scripts/review_nightly.py` refits it from the Fritz classifications); the blind alarm is unchanged.

## Forecast, SN Ia parameters and the next observation

```python
import flare
from flare.data import load_events
engine = flare.load_analysis()
events = load_events("ZTF21abcdxyz.npz")
fc = engine.forecast(events, horizon_days=5)                    # g and r to day 90 with 68% and 95% bands
fc = engine.forecast(events, horizon_days=10, recipe="blend")   # optional blend at day 10 or 20
pe = engine.pe(events, horizon_days=10)                         # SALT3 t0, mB, x1, c, assuming an SN Ia
nx = engine.next_observation_value(events, horizon_days=5)      # one more g or r point tomorrow
```

- **Forecast:** gradient-boosted regressors on the same features predict the g and r magnitude on
  a fixed grid of days, from detections up to day 5, 10, 15, 20 or 30, with conformal 68% and 95%
  bands per predicted class. The median error of the day-50 magnitude is 0.43 mag from day 5 and
  0.16 mag from day 30, lower than Bazin and Gaussian-process extrapolations at every horizon. The
  bands cover 67-69% and 94-97% of the future measurements over the population; they are too narrow
  for SLSNe and, at late horizons, for CVs.
- **SN Ia parameters:** the full-curve SALT3 values from the first 5, 10 or 20 days; the peak time
  falls from 1.5 to 0.6 d error and the peak magnitude from 0.25 to 0.09 mag between day 5 and day
  20, while the stretch stays close to the population median until the decline is observed.
  `analyze()` abstains unless SN Ia is the only class in the 90% set.
- **Next observation:** the expected reduction of the 90% prediction set, and the probability that
  it closes to one class, after one more g or r point on the next night, drawn from the forecast.
  It ranks objects by their gain (Spearman 0.58 at day 5 against the real next point); the closing
  probability is a ranking score, not a calibrated probability.

## SkyPortal / Fritz

`flare.skyportal.analyze` is the analysis service used on Fritz. It routes each light curve to
the model trained at its phase (day 5, 10 or 20, otherwise the full-curve model), posts a
classification only when the 90% set holds a single class (or "Supernova" when every class in
the set is a supernova type), and returns the probabilities, intervals, prediction set, anomaly
p-value with the smallest false-alarm rate that flags the object, the lifetime false-alarm rate,
the review priority and a triage verdict as annotations. A source reaches anomaly review when its
lifetime level is one false alarm per week or rarer. The service also returns the g and r forecast
as an overlay on the photometry plot, at the smallest shipped horizon that contains the detections,
for light curves with at least eight detections (two in each of g and r) spanning less than 90 days.

## Repository

```
flare/       the package (features, models, conformal sets, Venn-Abers, anomaly layer, SkyPortal)
models/      trained weights with manifests (full-curve, day 5/10/20, forecast, SN Ia PE, lifetime FAR)
benchmark/   the BTS benchmark: light curves, splits, features and catalogue context (DATASET.md)
scripts/     training and export scripts for the shipped models, and the scripts that build the benchmark
notebooks/   tutorials: flare_tutorial, why_flare_works, similarity_search, the Rubin lesson, ztf_summer_school_v2
docs/        the analysis guide (unified_analysis.md) and research notes (research_notes.md)
```

## Citation

If you use FLARE, please cite Sasli et al. 2026, *Photometric classification and out-of-taxonomy
detection of ZTF transients with FLARE*, [arXiv:2610.30656](https://arxiv.org/abs/2610.30656)
(see `CITATION.cff`). For the forecast, the SN Ia parameters and the value of the next observation,
please also cite Sasli et al. (in prep.), *What the first days of a transient predict*.

## License

MIT, see `LICENSE`.
