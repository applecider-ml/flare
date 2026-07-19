# FLARE inference weights

Everything needed to classify a light curve. Pure LightGBM + JSON — no torch.

| File | What |
|---|---|
| `flare_lgbm.txt` | LightGBM booster, 160 physics features, 1140 trees — the day-100 model |
| `conformal.json` | Mondrian LAC per-class quantiles (α=0.10), calibrated on val (N=1991) |
| `model_card.json` | hyperparameters, class-weight β, Optuna val score |
| `flare_lgbm_day10.txt` | early-time model (features from the first 10 days) |

Load: `FlareClassifier.from_pretrained()` (defaults to these files).
Retrain/recalibrate: `python scripts/train.py --data-dir <photo_events>`.

The reported numbers were produced by exactly these weights (see
`scripts/evaluate.py`).
