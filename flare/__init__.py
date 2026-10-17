"""FLARE — Feature-based Light-curve Aggregated Ranking Ensemble.

A physics-features + gradient-boosted-trees classifier for ZTF photometric
transients with class-conditional conformal uncertainty. Well-calibrated,
interpretable, CPU-light, and deployable as plain model files.

Two schemes ship (selected by config, see flare.config):

    bts6    the default: SN_Ia | SN_CC | SLSN | AGN | TDE | CV, predicted as a
            five-way top level with a dedicated SLSN head, optionally consuming
            host-galaxy photometry and a photo-z pseudo-absolute magnitude
    broad5  the original AppleCiDEr taxonomy (SNI | SNII | CV | AGN | TDE),
            one flat model; kept for the tutorial notebooks

Quick start:

    import flare
    clf = flare.load_classifier()                  # scheme from config/env
    labels = clf.predict_from_files(["obj.npz"])
    sets = clf.prediction_sets_from_files(["obj.npz"])

    # everything from an object id, photometry via BOOM (or public ALeRCE):
    from flare.fetch import fetch_events
    from flare.context import context_features
"""
from typing import Optional

from .config import DEFAULT_SCHEME, SCHEMES, resolve_scheme
from .model import FlareClassifier
from .taxonomy import BROAD_CLASSES, NUM_CLASSES

__version__ = "2.0.0"


def load_classifier(scheme: Optional[str] = None, horizon_days: Optional[int] = None):
    """Load the pretrained classifier for a scheme (default: config/env)."""
    cfg = resolve_scheme(scheme)
    if horizon_days not in (None, 100):
        if cfg.name != "bts6" or horizon_days not in (5, 10, 20):
            raise ValueError("early classifiers support bts6 at days 5, 10 or 20")
        from dataclasses import replace
        from pathlib import Path
        import hashlib
        import json
        root = Path(cfg.model_paths['top']).parent / 'early' / f'h{int(horizon_days)}'
        manifest = json.loads((root/'manifest.json').read_text())
        if not {'top.txt', 'slsn_branch.txt', 'conformal.json', 'model_card.json'} <= manifest.keys():
            raise ValueError("early classifier manifest is incomplete")
        for filename, digest in manifest.items():
            path = (root/filename).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"early classifier manifest mismatch: {filename}")
        paths = {k:root/name for k,name in [('top','top.txt'),('slsn_branch','slsn_branch.txt'),
                                           ('conformal','conformal.json'),('card','model_card.json')]}
        cfg = replace(cfg, model_paths=paths, horizon_days=float(horizon_days))
    if cfg.name == "broad5":
        return FlareClassifier.from_pretrained()
    from .hierarchical import HierarchicalFlare
    return HierarchicalFlare.from_pretrained(cfg)


def load_analysis(**kwargs):
    """Load the offline BTS6 classification/AD/reconstruction/forecast/PE API."""
    from .analysis import FlareAnalysis
    return FlareAnalysis(**kwargs)


__all__ = ["FlareClassifier", "load_classifier", "load_analysis", "resolve_scheme",
           "SCHEMES", "DEFAULT_SCHEME", "BROAD_CLASSES", "NUM_CLASSES"]


def _cli():                       # console_scripts entry point: `flare`
    from .cli import main
    raise SystemExit(main())
