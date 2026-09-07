# ZTF Summer School — FLARE tutorial

Materials for a hands-on session on machine learning for ZTF transient
classification, built around the FLARE architecture. The session has two acts:

**Act 1 — supervised classification (the known unknowns).**
The ML parts of the FLARE pipeline, using the pretrained 5-class weights
shipped with the package (`models/flare_lgbm.txt`): light curves to physics features, gradient-boosted trees, calibration, conformal prediction sets, feature attribution. Follows the flow of `../flare_tutorial.ipynb`.

**Act 2 — anomaly detection (the unknown unknowns).**
A special teaching model, `models/flare_lgbm_no_tde.txt`, was trained with **every TDE removed** from both training and validation (4 classes: SNI, SNII, CV, AGN; val accuracy 0.955). To this model, tidal disruption events do not exist. Pushing the real test TDEs through it:

| behaviour | value |
|---|---|
| where TDEs land | mostly AGN (7/9), some SN |
| mean max-probability on TDEs | 0.72 |
| mean max-probability on known classes | 0.96 |

The classifier cannot name what it has never seen — but it is measurably
*less confident*. Act 2 uses this model plus the **[LUNA](https://github.com/asasli/luna)**
anomaly-detection package. Headline numbers from the executed tutorial:
The discovery pool holds **56 hidden TDEs** (9 test + 47 train-excluded)
among ~2,000 objects — all unseen by the model, the scorers and the combiner.
Blind rank-average consensus of 14 scorers: AUC 0.72, 6/56 TDEs in the top-40
(6x chance). Adding just the 9 validation TDEs via LUNA's supervised combiner
(fitted on val only — fully independent of the evaluation pool): AUC **0.91**,
18/56 in the top-40, 20/56 inside 49 flagged at a 1% false-alarm rate — the
active-learning loop of a real survey.

## Hackathon

Track briefs for the hands-on day (anomaly detection + Hyrax): see
[`HACKATHON.md`](HACKATHON.md) — three tracks from first-detection to
open research questions.

## Files

| File | Purpose |
|---|---|
| `train_no_tde.py` | reproduces the 4-class no-TDE model (uses the published FLARE hyperparameters, ~1 min) |
| `../..​/models/flare_lgbm_no_tde.txt` | the no-TDE weights (820 trees) |
| `../../models/model_card_no_tde.json` | its model card |
| `tutorial.ipynb` | the session notebook (executed; both acts) |
| `beyond_the_catalog.ipynb` | bonus: 8 real out-of-taxonomy transients (FBOT, Ca-rich, ILRT, SLSN-I, SN Iax) scored as intruders — categories mirroring ASTRANet (arXiv:2607.08044) |
| `fetch_rare_transients.py` | downloads + converts the external objects (`rare_transients/`) — **BOOM** backend (BOOM_URL + token/credentials) with public ALeRCE fallback |

## Loading the no-TDE model

```python
from flare import FlareClassifier
clf = FlareClassifier.from_pretrained(
    model_path="models/flare_lgbm_no_tde.txt",
    conformal_path="/nonexistent")        # no conformal for the 4-class model
proba = clf.predict_proba_from_files(files)   # (N, 4): SNI, SNII, CV, AGN
```

`predict` maps argmax over the 4 columns to the correct class names
(TDE is the 5th class and can never be predicted).
