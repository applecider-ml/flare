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

## bts6 (the paper's headline scheme)

`models/bts6/` carries the hierarchical classifier of the BTS paper:

| file | role |
|---|---|
| `top.txt` | five-way top level (SN_Ia, SN_CC+, AGN, TDE, CV), 171 features (160 light-curve + host block + M_pseudo) |
| `slsn_branch.txt` | binary SLSN head inside SN_CC+, threshold in the model card |
| `conformal.json` | Mondrian per-class thresholds, alpha = 0.10, calibrated on a validation half untouched by early stopping |
| `model_card.json` | branch threshold, training provenance, and the five-fold numbers to quote |

Trained by `scripts/train_bts6.py` on the benchmark's training split with the
external blocks blanked on 30% of rows (the missingness guard). Performance
claims should always come from the paper's five-fold cross-evaluation, not
from any single split.

The `broad5` files above remain the original AppleCiDEr-era release used by
the tutorial notebooks (including `flare_lgbm_no_tde.txt`, the teaching model
with TDEs deliberately removed).
