# FLARE

**Feature-based Light-curve Aggregated Ranking Ensemble**: a photometric classifier for ZTF
transients with calibrated uncertainty and an anomaly layer for what lies outside its taxonomy.
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
```

External context is optional: the models are trained with each block randomly removed, so pass
NaN for anything that is missing and never impute it.

## Performance

Five-fold cross-validation on 10,497 spectroscopically confirmed BTS transients:

| inputs | accuracy | balanced accuracy | macro F1 |
|---|---|---|---|
| light curve only | 90% | 71% | 0.72 |
| + host, M_pseudo, Gaia/WISE, PS1 counterpart | 92% | 83% | 0.83 |

Per-class F1: SN Ia 0.96, SN CC 0.85, SLSN 0.53, AGN 0.90, TDE 0.81, CV 0.94. The models trained
on the first 5, 10 and 20 days reach a macro F1 of 0.56, 0.63 and 0.73 on the same benchmark.

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

## SkyPortal / Fritz

`flare.skyportal.analyze` is the analysis service used on Fritz. It routes each light curve to
the model trained at its phase (day 5, 10 or 20, otherwise the full-curve model), posts a
classification only when the 90% set holds a single class (or "Supernova" when every class in
the set is a supernova type), and returns the probabilities, intervals, prediction set, anomaly
p-value with the smallest false-alarm rate that flags the object, and a triage verdict as
annotations.

## Repository

```
flare/       the package (features, models, conformal sets, Venn-Abers, anomaly layer, SkyPortal)
models/      trained weights with manifests (full-curve, day 5/10/20, forecast, SN Ia PE)
benchmark/   the BTS benchmark: light curves, splits, features and catalogue context (DATASET.md)
scripts/     training and export scripts for the shipped models, and the scripts that build the benchmark
notebooks/   tutorials: flare_tutorial, why_flare_works, similarity_search, the Rubin lesson
docs/        the analysis guide (unified_analysis.md) and research notes (research_notes.md)
```

## Citation

If you use FLARE, please cite the accompanying paper (Sasli et al., in prep.).

## License

MIT, see `LICENSE`.
