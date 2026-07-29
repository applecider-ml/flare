# FLARE

**Feature-based Light-curve Aggregated Ranking Ensemble** — a physics-features +
gradient-boosted-trees classifier for ZTF photometric transients
(SNI · SNII · CV · AGN · TDE), with class-conditional conformal uncertainty.

FLARE extracts 160 physically-motivated features from each multi-band light curve
(variability statistics, Bazin transient fits, colour evolution, cadence) and
classifies them with a LightGBM ensemble. It is designed for production use:
well-calibrated probabilities, guaranteed per-class conformal coverage,
interpretable decisions, CPU-minutes to train, and a single model file to deploy.

## Performance

Five-fold cross-validation over 13,153 quality-filtered ZTF objects:

| Metric | FLARE |
|---|---|
| accuracy | 0.949 ± 0.005 |
| balanced accuracy | 0.848 ± 0.031 |
| macro F1 | 0.863 ± 0.025 |
| macro AUPRC | 0.905 ± 0.032 |
| ECE (calibration) | 0.013 ± 0.005 |

Per-class (same protocol): SNI F1 0.96 · SNII 0.84 · CV 0.88 · AGN 0.98 ·
TDE precision 0.73 / recall 0.62 (65 events). Conformal prediction sets meet the
90% coverage target for **every** class, including TDE. A class-weighted variant
raises TDE recall to 0.74 for rare-transient-focused deployments.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
```

Requires `light-curve`, `lightgbm`, `numpy/pandas/scipy/scikit-learn`. No GPU, no
deep-learning frameworks.

## Predict

```python
from flare import FlareClassifier

clf = FlareClassifier.from_pretrained()          # bundled weights + conformal
proba = clf.predict_proba_from_files(["ZTF20xxx.npz"])   # (N, 5)
label = clf.predict_from_files(["ZTF20xxx.npz"])         # ['TDE']
sets  = clf.prediction_sets_from_files(["ZTF20xxx.npz"]) # [['TDE','AGN']] (90% coverage)
```

Command line:

```bash
python scripts/predict.py --npz-dir path/to/objects --out preds.csv
python scripts/predict.py --manifest manifest_test.csv --data-dir path/to/data --out preds.csv
```

Each input `.npz` holds a `(n_events, 15)` `data` array with columns
`dt, dt_prev, band_id, logflux, logflux_err, band_g/r/i, g_r(±err/flag),
r_i(±err/flag), label`. FLARE reconstructs per-band light curves and extracts
its features automatically.

## Train / reproduce

```bash
export FLARE_DATA=/path/to/photo_events      # manifest_{train,val,test}.csv + .npz + feature_stats_day100.npz
python scripts/train.py    --trials 100      # -> models/flare_lgbm.txt + conformal.json
python scripts/evaluate.py --split test      # bootstrap CIs + conformal coverage
python scripts/train.py    --horizon 10      # early-time (day-10) model
```

## Learn

Two executable, student-level lessons (with exercises and baked outputs):

- `notebooks/flare_tutorial.ipynb` — from raw light curves to features, trees,
  calibration, conformal prediction and interpretability.
- `notebooks/similarity_search.ipynb` — the FLARE features as an interpretable
  embedding: k-NN retrieval ("find more like this one"), a 2-D map of the
  transient sky, k-NN classification, anomaly detection via neighbour distance,
  and scaling with approximate NN. Companion to the
  [SimilaritySearch](https://github.com/asasli/SimilaritySearch) lecture series.

## What's in the box

```
flare/
  __init__.py      FlareClassifier
  model.py         classifier API (predict / predict_proba / prediction_sets)
  features.py      160 physics features from a light curve
  data.py          light-curve reconstruction + quality filter
  conformal.py     Mondrian (class-conditional) conformal calibrator
  taxonomy.py      5 broad classes + subclass mapping
  metrics.py       metrics + stratified bootstrap CIs
models/            flare_lgbm.txt, conformal.json, model_card.json (+ day-10)
scripts/           train.py, predict.py, evaluate.py
notebooks/         flare_tutorial.ipynb (teaching notebook)
examples/          quickstart.py
```

## Why it works

SHAP attribution on the trees recovers textbook transient physics: TDEs are
identified by a **blue, near-constant g−r colour and a smooth decline**, AGN by
stochastic variability and a poor transient fit, SNe by their rise/fade shape and
reddening. Engineered physics features plus tree-based ranking, with honest
calibration on top.

## Uncertainty

`prediction_sets` returns, per object, the set of classes guaranteed to contain
the truth with probability ≥ 1−α (α=0.10 by default), **per class** (Mondrian
conformal prediction). Easy objects get a single-class set; genuinely ambiguous
ones (typically TDE/AGN) get a two-class shortlist — ideal for spectroscopic
follow-up triggering.

## Citation

If you use FLARE, please cite the accompanying paper (in prep.).

## License

MIT — see `LICENSE`.
