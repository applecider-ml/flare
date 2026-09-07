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


def load_classifier(scheme: Optional[str] = None):
    """Load the pretrained classifier for a scheme (default: config/env)."""
    cfg = resolve_scheme(scheme)
    if cfg.name == "broad5":
        return FlareClassifier.from_pretrained()
    from .hierarchical import HierarchicalFlare
    return HierarchicalFlare.from_pretrained(cfg)


__all__ = ["FlareClassifier", "load_classifier", "resolve_scheme",
           "SCHEMES", "DEFAULT_SCHEME", "BROAD_CLASSES", "NUM_CLASSES"]


def _cli():                       # console_scripts entry point: `flare`
    from .cli import main
    raise SystemExit(main())
