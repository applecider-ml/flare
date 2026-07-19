"""FLARE — Feature-based Light-curve Aggregated Ranking Ensemble.

A physics-features + gradient-boosted-trees classifier for ZTF photometric
transients (SNI, SNII, CV, AGN, TDE) with class-conditional conformal
uncertainty. Well-calibrated, interpretable, CPU-light, and deployable as a
single model file.
"""
from .model import FlareClassifier
from .taxonomy import BROAD_CLASSES, NUM_CLASSES

__version__ = "1.0.0"
__all__ = ["FlareClassifier", "BROAD_CLASSES", "NUM_CLASSES"]
