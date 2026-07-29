# FusionNet — pretrained weights

The neural companion to FLARE: a FiLM-GRU sequence encoder that also receives
the same 160 physics features FLARE uses (projected through a 96-wide block and
concatenated with the pooled sequence representation before the classifier).
3-seed ensemble + a single temperature parameter tuned on the validation set.

On the held-out test split (N=1,979) the ensemble reaches accuracy 0.953,
balanced accuracy 0.849, macro F1 0.856, ECE 0.009 — leading the feature-tree
model on macro F1 / balanced accuracy / TDE recall, trailing it on AUPRC.
(Single fixed split; treat rare-class numbers with caution.)

## Files

| File | Contents |
|---|---|
| `fusion_d96_seed{42,123,456}.pt` | encoder / classifier / feature-projector weights per seed |
| `config.json` | full architecture + training configuration |
| `feature_scaler.npz` | median / IQR / names for the 160 features (train-only statistics) |
| `feature_stats_day100.npz` | per-channel flux normalisation for the sequence input |
| `ensemble.json` | seed list + temperature (0.538) |

## Usage

The `flare` package itself stays torch-free; the loader lives with the teaching
notebooks and needs `torch` plus the pulsar architecture module
(`PULSAR_PATH` env var; defaults to the cluster copy).

```python
import sys; sys.path.insert(0, "notebooks/ztf_summer_school")
from fusion_net import FusionNet

net = FusionNet.from_pretrained()            # CPU is fine
proba  = net.predict_proba_from_files(files) # (N, 5) — same API as FLARE
labels = net.predict_from_files(files)       # ['TDE', ...]
```

Verified against the training-run probabilities: identical labels; probability
drift ≤ 5e-3 (GPU-trained vs CPU-inference float32 — expected and benign).
