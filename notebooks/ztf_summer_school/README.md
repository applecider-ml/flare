# ZTF Summer School — session materials (v2)

Two notebooks, one session, one hackathon.

| Notebook | What it covers | Time |
|---|---|---|
| `deep_architecture.ipynb` | build a deep sequence classifier on **raw** light curves from scratch — architecture, training, full evaluation report. CPU-only, every line visible. | ~30 min |
| `flare_pipeline.ipynb` | use the **pretrained FLARE** pipeline: classification, conformal prediction sets, then anomaly detection with LUNA and an active-learning step on TDEs the model never saw. No training. | ~30 min |

## Setup

```bash
git clone https://github.com/applecider-ml/flare.git
cd flare
pip install -e .
pip install luna    # anomaly-detection package (github.com/asasli/luna)
export FLARE_DATA=/path/to/photo_events
```

## Hackathon

- **Anomaly-detection track** starts from `flare_pipeline.ipynb`, section 7:
  score the 8 real out-of-taxonomy intruders in `rare_transients/`, find which
  ones hide, improve the detector, and map the value-of-labels curve. Tasks
  are listed at the end of the notebook.
- **Hyrax track** starts from `deep_architecture.ipynb`: wrap the trained
  `LightCurveGRU` as a Hyrax model, following the patterns from the Hyrax
  session.

`train_no_tde.py` rebuilds the 4-class no-TDE model from scratch (~1 minute)
if you want to change what counts as "known"; `fetch_rare_transients.py`
re-downloads the intruders from public brokers.
