# FLARE

**Feature-based Light-curve Aggregated Ranking Ensemble** — a physics-features +
gradient-boosted-trees classifier for ZTF photometric transients, with
class-conditional conformal uncertainty and an anomaly layer for what falls
outside the taxonomy.

FLARE extracts 160 physically-motivated features from each multi-band light
curve (variability statistics, Bazin transient fits, colour evolution, cadence),
optionally joins external context available at alert time (Pan-STARRS host
photometry; a Legacy-Surveys photo-z pseudo-absolute magnitude; Gaia DR3 and
AllWISE point-source context), and classifies
with LightGBM. Designed for production: calibrated probabilities, guaranteed
per-class conformal coverage, interpretable decisions, CPU-minutes to train,
plain model files to deploy.

## Two schemes, one switch

| scheme | classes | model | when |
|---|---|---|---|
| **`bts6`** (default) | SN Ia · SN CC · SLSN · AGN · TDE · CV | 5-way top level + SLSN head; host, photo-z, Gaia/WISE, and PS1-counterpart aware | the paper's headline; new work |
| `broad5` | SNI · SNII · CV · AGN · TDE | flat LightGBM | legacy AppleCiDEr taxonomy; tutorial notebooks |

Selection: `flare.load_classifier("bts6")`, the `FLARE_SCHEME` environment
variable, or the default in `flare/config.py`.

## Performance (bts6, five-fold out of fold, 10,497 BTS objects)

| Metric | light curve only | + host | + Gaia/WISE | + PS1 counterpart |
|---|---|---|---|---|
| accuracy | 0.899 ± 0.005 | 0.918 ± 0.005 | 0.922 ± 0.005 | **0.924 ± 0.001** |
| balanced accuracy | 0.713 ± 0.034 | 0.799 ± 0.021 | 0.811 ± 0.026 | **0.826 ± 0.033** |
| macro F1 | 0.723 ± 0.034 | 0.804 ± 0.012 | 0.820 ± 0.017 | **0.832 ± 0.016** |

Per-class F1 in the adopted configuration: SN Ia 0.96 · SN CC 0.85 · SLSN 0.53 ·
AGN 0.90 · TDE 0.81 · CV 0.94. The anomaly layer deliberately reads a narrower
context (features + host + Gaia/WISE; `models/bts6/ad_space.txt`) — M_pseudo
and the counterpart block help classification but harm blind detection. Mondrian conformal sets hold the 90% target
(marginal 0.910; the 68-object TDE class sits at 0.853, within binomial noise),
delivering the entangled classes as small candidate sets
(SLSN ≈ 2.1 labels) rather than silent misclassifications. Supervised detection
of a trained rare class reaches 94% (TDE) / 86% (novae) at a 1% false-alarm
budget. The `broad5` numbers of the original release are in `models/README.md`.

## Command line

```bash
flare report ZTF19acbzgog ZTF22abkfhua -o report.html   # a self-contained console page
flare predict ZTF19acbzgog                              # the same as JSON
```

Photometry comes from BOOM when `BOOM_URL` and credentials are set and from
ALeRCE otherwise; the host, photo-z, Gaia/WISE and pre-outburst-counterpart
blocks are queried per position. Anything unavailable stays missing — the
models are trained with each block randomly blanked, so absence degrades
gracefully instead of being misread. `docs/flare_console.html` is what the
report looks like.

## Repository layout

    flare/        the package (features, models, conformal, fetch, context)
    models/       pretrained weights for both schemes
    benchmark/    the BTS benchmark: splits, features, external context
                  (light curves via Zenodo or BOOM re-fetch — see
                  benchmark/DATASET.md)
    paper/        every analysis script behind the paper's numbers and figures
    notebooks/    tutorials, including the ZTF summer-school session

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
```

Requires `light-curve`, `lightgbm`, `numpy/pandas/scipy/scikit-learn`. No GPU, no
deep-learning frameworks.

## Predict

```python
import flare
clf = flare.load_classifier()                    # bts6 by default

labels = clf.predict_from_files(["ZTF21abcdxyz.npz"])
sets   = clf.prediction_sets_from_files(["ZTF21abcdxyz.npz"])   # conformal
energy = clf.anomaly_energy(clf.features_from_files([...]))     # OOD score

# everything from an object id (BOOM if configured, else public ALeRCE):
from flare.fetch import fetch_events
from flare.context import context_features        # host + photo-z + Gaia/WISE + PS1 counterpart
```

All external columns are optional at inference: the models are trained with the
external blocks randomly blanked, so a hostless or uncovered object degrades
gracefully. Never impute those columns — pass NaN.

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
