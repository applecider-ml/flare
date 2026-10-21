# FLARE weights

Plain LightGBM boosters and JSON, checked against SHA-256 manifests at load time; no torch.
The one exception is `consensus.pkl.gz`, a pickled LUNA pipeline that is read only when
`luna-ood` 0.3 or later is installed, and only after its SHA-256 matches the manifest.

| directory | content |
|---|---|
| `bts6/` | the six-class classifier of the paper (SN Ia, SN CC, SLSN, AGN, TDE, CV): five-way top level (`top.txt`), SLSN head (`slsn_branch.txt`), cross-conformal calibration (`conformal.json`), anomaly space and calibration (`ad_space.txt`, `anomaly_calibration.json`), Venn-Abers tables (`venn_abers.json`), blind-consensus anomaly alarm at 1%, 5% and 10% false-alarm rates (`consensus.pkl.gz`), model card and manifest |
| `bts6/early/h5`, `h10`, `h20` | the same classifier trained on light curves truncated at 5, 10 and 20 days, each with its own calibration |
| `bts6/forecast/`, `bts6/forecast_blend/`, `bts6/pe_ia/` | forecast and SN Ia parameter-estimation models of `flare.load_analysis()` |
| `bts6/onnx/` | ONNX exports of the top level and the SLSN head |

Load: `flare.load_classifier()` (full curve) or `flare.load_classifier(horizon_days=5)`.
The full-curve and early models are fitted on the benchmark's train and validation splits
(`benchmark/`), so the test split gives an honest evaluation. Training scripts: `scripts/train_bts6.py`,
`scripts/train_early_cc.py`, `scripts/export_venn_abers.py`, `scripts/export_consensus.py`.

The five-class weights of the original release (`broad5`) are no longer shipped; train them on
the benchmark with `python scripts/train.py` if needed.
