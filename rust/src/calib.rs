//! Conformal and anomaly calibration, read from the artefacts the Python
//! trainer ships (`conformal.json`, `anomaly_calibration.json`).

use std::collections::HashMap;

use serde::Deserialize;

use crate::FlareError;

/// Class-conditional (Mondrian) conformal calibration.
#[derive(Clone, Debug)]
pub struct Conformal {
    pub alpha: f64,
    pub classes: Vec<String>,
    /// Per-class nonconformity quantile at `alpha`; membership is 1 − P(c) ≤ q̂_c.
    pub qhat: Vec<f64>,
    /// Per-class calibration nonconformity scores, sorted ascending.
    pub cal_scores: Vec<Vec<f64>>,
    /// Out-of-fold coverage measured per class by the trainer, when shipped.
    pub empirical_coverage: Vec<Option<f64>>,
    /// Calibration members per class, when shipped.
    pub n_cal_per_class: Vec<Option<u64>>,
    /// Smallest expressible p-value per class, 1/(n_c + 1), when shipped.
    pub p_value_floor: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct ConformalJson {
    alpha: f64,
    classes: Vec<String>,
    qhat: HashMap<String, f64>,
    #[serde(default)]
    cal_scores: HashMap<String, Vec<f64>>,
    #[serde(default)]
    empirical_coverage: HashMap<String, f64>,
    #[serde(default)]
    n_cal_per_class: HashMap<String, u64>,
    #[serde(default)]
    p_value_floor: HashMap<String, f64>,
}

impl Conformal {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, FlareError> {
        Self::from_str(&std::fs::read_to_string(path)?)
    }

    /// From the text of `conformal.json` (also stored verbatim as ONNX metadata `conformal`).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Self, FlareError> {
        let j: ConformalJson = serde_json::from_str(text)?;
        let mut qhat = Vec::new();
        let mut cal = Vec::new();
        let (mut cov, mut ncal, mut floor) = (Vec::new(), Vec::new(), Vec::new());
        for c in &j.classes {
            qhat.push(*j.qhat.get(c).ok_or_else(|| FlareError::Artefact(format!("qhat missing {c}")))?);
            let mut s = j.cal_scores.get(c).cloned().unwrap_or_default();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            cal.push(s);
            cov.push(j.empirical_coverage.get(c).copied());
            ncal.push(j.n_cal_per_class.get(c).copied());
            floor.push(j.p_value_floor.get(c).copied());
        }
        Ok(Conformal {
            alpha: j.alpha,
            classes: j.classes,
            qhat,
            cal_scores: cal,
            empirical_coverage: cov,
            n_cal_per_class: ncal,
            p_value_floor: floor,
        })
    }

    /// p_c(x) = (1 + #{s ∈ cal_c : s ≥ 1 − P(c)}) / (n_c + 1); NaN when no scores shipped.
    pub fn p_values(&self, p6: &[f64]) -> Vec<f64> {
        self.cal_scores
            .iter()
            .zip(p6)
            .map(|(cal, &p)| {
                if cal.is_empty() {
                    return f64::NAN;
                }
                let s = 1.0 - p;
                // ge = n - searchsorted(cal, s, side="left") = #{cal >= s}
                let lt = cal.partition_point(|&v| v < s);
                (1.0 + (cal.len() - lt) as f64) / (cal.len() as f64 + 1.0)
            })
            .collect()
    }

    /// Prediction set at the bundled alpha, thresholding the stored quantiles.
    pub fn prediction_set(&self, p6: &[f64]) -> Vec<String> {
        self.classes
            .iter()
            .zip(p6)
            .zip(&self.qhat)
            .filter(|((_, &p), &q)| 1.0 - p <= q)
            .map(|((c, _), _)| c.clone())
            .collect()
    }
}

/// Energy → probability, with the base rate made explicit.
#[derive(Clone, Debug, Deserialize)]
pub struct AnomalyCalibration {
    pub coef: f64,
    pub intercept: f64,
    pub base_rate: f64,
    #[serde(default)]
    pub known_energies: Vec<f64>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct AnomalyReport {
    pub energy: f64,
    pub likelihood_ratio: f64,
    pub p_anomaly: f64,
    pub base_rate: f64,
    /// Fraction of known-class objects at least this extreme (distribution-free).
    pub novelty_p: Option<f64>,
    pub energy_percentile: Option<f64>,
}

impl AnomalyCalibration {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, FlareError> {
        Self::from_str(&std::fs::read_to_string(path)?)
    }

    /// From the text of `anomaly_calibration.json` (also stored as ONNX metadata `anomaly_calibration`).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Self, FlareError> {
        let mut c: AnomalyCalibration = serde_json::from_str(text)?;
        c.known_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(c)
    }

    pub fn report(&self, energy: f64, base_rate: Option<f64>) -> AnomalyReport {
        let pi = base_rate.unwrap_or(self.base_rate);
        let lr = (self.coef * energy + self.intercept).exp();
        let odds = lr * pi / (1.0 - pi);
        let p = odds / (1.0 + odds);
        let (novelty_p, pct) = if self.known_energies.is_empty() {
            (None, None)
        } else {
            let n = self.known_energies.len();
            let ge = n - self.known_energies.partition_point(|&v| v < energy);
            (Some((1.0 + ge as f64) / (n as f64 + 1.0)), Some(100.0 * (1.0 - ge as f64 / n as f64)))
        };
        AnomalyReport { energy, likelihood_ratio: lr, p_anomaly: p, base_rate: pi, novelty_p, energy_percentile: pct }
    }
}
