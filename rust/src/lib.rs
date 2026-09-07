//! FLARE for alert brokers.
//!
//! The photometric module of the AppleCiDEr pipeline (Sasli et al.), as a
//! dependency-light Rust crate that a broker such as BOOM can call per alert:
//!
//! * [`lightcurve`]  — detections (mjd, band, mag, mag_err) → the event
//!   representation FLARE was trained on (flux space, per-band curves,
//!   g−r / r−i colour series within a 1.5-day window, 100-day horizon).
//! * [`features`]    — the 160 physics features: cadence, peak, mean SNR,
//!   26 `light-curve-feature` statistics on log-flux, a Bazin fit in flux
//!   space, colour statistics and cross-band peak relations.
//! * [`lgbm`]        — an exact evaluator for LightGBM text models
//!   (double thresholds, LightGBM's missing-value semantics), so the
//!   probabilities that feed the conformal p-values are bit-for-bit those
//!   of the Python model.
//! * [`calib`]       — class-conditional (Mondrian) conformal sets and the
//!   prevalence-aware anomaly probability from the shipped calibration.
//! * [`model`]       — the hierarchical classifier: five-way top level,
//!   SLSN branch inside SN CC⁺, anomaly energy in its own feature space.
//!
//! Every numeric convention mirrors the Python package `flare` that lives
//! beside this crate in <https://github.com/applecider-ml/flare>; the parity
//! tests under `rust/parity/` compare the two on the benchmark objects.
//!
//! Brokers that already depend on another crate called `flare` (BOOM does,
//! for `boom-astro/flare`) import this one under a different name:
//! `applecider-flare = { package = "flare", git = "https://github.com/applecider-ml/flare", version = "2" }`.

pub mod calib;
pub mod features;
pub mod lgbm;
pub mod lightcurve;
pub mod model;

#[derive(thiserror::Error, Debug)]
pub enum FlareError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("LightGBM model parse error: {0}")]
    ModelParse(String),
    #[error("model artefact missing or inconsistent: {0}")]
    Artefact(String),
    #[error("light curve fails the quality cut (need ≥8 detections, ≥2 in g and ≥2 in r)")]
    QualityCut,
}
