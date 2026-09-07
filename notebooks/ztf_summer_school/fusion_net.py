"""FusionNet — the neural companion to FLARE, with the same predict API.

Loads the pretrained fusion FiLM-GRU (sequence encoder + the same 160 physics
features FLARE uses, d_global=96, 3-seed ensemble + temperature) from
``flare/models/fusion_net/``.

The flare *package* stays torch-free on purpose; this wrapper lives with the
teaching notebooks and needs two extras at runtime:
  * torch
  * the pulsar research module (architecture definition), located via the
    PULSAR_PATH env var (default: the beat_tempo vendored copy on the cluster).

Usage:
    from fusion_net import FusionNet
    net = FusionNet.from_pretrained()
    proba  = net.predict_proba_from_files(files)   # (N, 5) — same as FLARE
    labels = net.predict_from_files(files)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
FLARE_ROOT = _HERE.parents[2]                      # .../flare
MODEL_DIR = FLARE_ROOT / "models" / "fusion_net"
PULSAR_PATH = Path(os.environ.get(
    "PULSAR_PATH", "/projects/bcrv/asasli/later/beat_tempo/external"))

sys.path.insert(0, str(FLARE_ROOT))
from flare.features import extract_from_array           # noqa: E402
from flare.taxonomy import BROAD_CLASSES, NUM_CLASSES   # noqa: E402


class FusionNet:
    def __init__(self, models, collate, scaler, cfg, temperature):
        self._models = models          # list of (encoder, clf, proj)
        self._collate = collate
        self._med, self._iqr, self._names = scaler
        self._cfg = cfg
        self.temperature = float(temperature)

    # ------------------------------------------------------------- loading --
    @classmethod
    def from_pretrained(cls, model_dir=MODEL_DIR, pulsar_path=PULSAR_PATH,
                        device="cpu"):
        import torch
        sys.path.insert(0, str(pulsar_path))
        import pulsar

        cfg_d = json.load(open(Path(model_dir) / "config.json"))
        cfg = pulsar.Config()
        for k, v in cfg_d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, tuple(v) if isinstance(v, list) else v)

        sc = np.load(Path(model_dir) / "feature_scaler.npz", allow_pickle=True)
        names = [str(n) for n in sc["names"]]
        n_feat = len(names)

        ens = json.load(open(Path(model_dir) / "ensemble.json"))
        dev = torch.device(device)
        models = []
        for s in ens["seeds"]:
            encoder = pulsar._build_encoder(cfg).to(dev)
            clf_in = cfg.d_model + cfg.d_global
            clf = pulsar.CosineClassifier(clf_in, NUM_CLASSES,
                                          cfg.cosine_scale).to(dev)
            proj = pulsar.GlobalFeatProjector(n_feat, cfg.d_global,
                                              cfg.dropout).to(dev)
            ck = torch.load(Path(model_dir) / f"fusion_d96_seed{s}.pt",
                            map_location=dev)
            encoder.load_state_dict(ck["enc"])
            clf.load_state_dict(ck["clf"])
            proj.load_state_dict(ck["proj"])
            for m in (encoder, clf, proj):
                m.eval()
            models.append((encoder, clf, proj))

        mean, std = pulsar.load_stats(Path(model_dir) / cfg.stats_file)
        collate = pulsar._collate(mean, std)
        obj = cls(models, collate, (sc["median"], sc["iqr"], names),
                  cfg, ens["temperature"])
        obj._pulsar = pulsar
        obj._torch = torch
        obj._device = dev
        return obj

    # ----------------------------------------------------------- features --
    def _globals(self, arr):
        try:
            f = extract_from_array(arr)
            v = np.array([f.get(k, np.nan) for k in self._names], np.float64)
        except Exception:
            v = np.full(len(self._names), np.nan)
        v = np.where(np.isfinite(v), v, self._med)
        return np.clip((v - self._med) / self._iqr, -10, 10).astype(np.float32)

    # ---------------------------------------------------------- inference --
    def predict_proba_from_files(self, files, batch_size=256):
        torch = self._torch
        pulsar = self._pulsar
        items = []
        for fp in files:
            raw = np.load(fp, allow_pickle=False)
            arr = np.asarray(raw["data"] if hasattr(raw, "files") else raw,
                             np.float32)
            arr = arr[arr[:, 0] <= self._cfg.horizon_days]
            if len(arr) == 0:
                arr = np.zeros((1, 15), np.float32)
            seq = pulsar.build_event_tensor(arr)
            gf = torch.from_numpy(self._globals(arr))
            items.append((seq, gf, 0))          # label unused at inference

        probs = np.zeros((len(items), NUM_CLASSES))
        with torch.no_grad():
            for lo in range(0, len(items), batch_size):
                x, _, mask, lengths, gf = self._collate(items[lo:lo + batch_size])
                x, mask, lengths, gf = (t.to(self._device)
                                        for t in (x, mask, lengths, gf))
                seed_ps = []
                for encoder, clf, proj in self._models:
                    h = encoder(x, mask, lengths)
                    h = torch.cat([h, proj(gf)], -1)
                    seed_ps.append(torch.softmax(clf(h), -1).cpu().numpy())
                probs[lo:lo + len(seed_ps[0])] = np.mean(seed_ps, 0)

        # temperature on the ensemble mean (tuned on val NLL)
        lp = np.log(probs + 1e-12) / self.temperature
        e = np.exp(lp - lp.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict_from_files(self, files):
        p = self.predict_proba_from_files(files)
        return [BROAD_CLASSES[i] for i in p.argmax(1)]
