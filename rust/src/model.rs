//! The hierarchical FLARE classifier (`flare.hierarchical.HierarchicalFlare`).
//!
//! Five-way top level over {SN Ia, SN CC⁺, AGN, TDE, CV} with SN CC⁺ = SN CC ∪ SLSN,
//! a binary SLSN head inside SN CC⁺, class-conditional conformal sets over the six
//! reported classes, and the anomaly energy −logsumexp(raw) of a separate booster
//! trained in the anomaly space (features + host + Gaia/WISE).
//!
//! Two ways to run it:
//!
//! * [`Flare`] evaluates the LightGBM text models with the crate's own exact
//!   evaluator ([`crate::lgbm`]) — self-contained, no runtime dependency.
//! * [`Head`] holds only the hierarchy and the calibration, and turns *raw scores*
//!   produced elsewhere (an ONNX runtime, for a broker that keeps every model as
//!   ONNX) into the same [`Report`]. `scripts/export_onnx.py` writes those ONNX
//!   files with the calibration in their metadata, so [`Head::from_metadata`] can
//!   be built straight from the top model's `metadata_props`.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::calib::{AnomalyCalibration, AnomalyReport, Conformal};
use crate::features;
use crate::lgbm::{logsumexp, softmax, LgbmModel, Objective};
use crate::lightcurve::{Detection, LightCurve, HORIZON_DAYS};
use crate::FlareError;

pub const SIX: [&str; 6] = ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"];
pub const FIVE: [&str; 5] = ["SN_Ia", "SN_CC+", "AGN", "TDE", "CV"];

#[derive(Deserialize)]
struct Card {
    branch_threshold: f64,
    #[serde(default)]
    classes: Vec<String>,
}

/// Everything the per-alert card shows.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    /// Factorised probabilities over `SIX`.
    pub proba: [f64; 6],
    /// Hierarchical argmax with the tuned branch threshold.
    pub label: String,
    pub p_values: [f64; 6],
    /// Conformal set at the bundled alpha.
    pub set: Vec<String>,
    pub alpha: f64,
    /// Largest p-value (how typical the object is of its best class).
    pub credibility: f64,
    /// 1 − second-largest p-value.
    pub confidence: f64,
    pub anomaly: AnomalyReport,
    /// The argmax class is absent from its own conformal set: route to review.
    pub argmax_excluded: bool,
    pub n_features_present: usize,
    /// Out-of-fold coverage the trainer measured for the predicted class, if shipped.
    pub coverage_measured: Option<f64>,
    /// Calibration members of the predicted class, if shipped.
    pub coverage_n: Option<u64>,
    /// Smallest expressible p-value for the predicted class, if shipped.
    pub p_value_floor: Option<f64>,
}

/// Hierarchy + calibration, independent of how the raw scores were produced.
pub struct Head {
    pub branch_threshold: f64,
    pub conformal: Conformal,
    pub anomaly: AnomalyCalibration,
}

impl Head {
    /// From the artefacts' JSON texts (`model_card.json`, `conformal.json`,
    /// `anomaly_calibration.json`) — or the identical strings stored as ONNX
    /// `metadata_props` (`model_card`, `conformal`, `anomaly_calibration`).
    pub fn from_metadata(card_json: &str, conformal_json: &str, anomaly_json: &str) -> Result<Self, FlareError> {
        let card: Card = serde_json::from_str(card_json)?;
        if !card.classes.is_empty() && card.classes != SIX {
            return Err(FlareError::Artefact(format!("unexpected class order {:?}", card.classes)));
        }
        let conformal = Conformal::from_str(conformal_json)?;
        if conformal.classes != SIX {
            return Err(FlareError::Artefact("conformal classes differ from SIX".into()));
        }
        let anomaly = AnomalyCalibration::from_str(anomaly_json)?;
        Ok(Head { branch_threshold: card.branch_threshold, conformal, anomaly })
    }

    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, FlareError> {
        let d = dir.as_ref();
        Self::from_metadata(
            &std::fs::read_to_string(d.join("model_card.json"))?,
            &std::fs::read_to_string(d.join("conformal.json"))?,
            &std::fs::read_to_string(d.join("anomaly_calibration.json"))?,
        )
    }

    /// Factorised six-class probabilities from the top raw scores (5) and the branch logit.
    pub fn proba(&self, top_raw: &[f64], branch_raw: f64) -> ([f64; 6], f64) {
        let p5 = softmax(top_raw);
        let pb = 1.0 / (1.0 + (-branch_raw).exp());
        let mut p6 = [0.0; 6];
        for (j, c) in SIX.iter().enumerate() {
            p6[j] = match *c {
                "SN_CC" => p5[1] * (1.0 - pb),
                "SLSN" => p5[1] * pb,
                _ => p5[FIVE.iter().position(|f| f == c).unwrap()],
            };
        }
        (p6, pb)
    }

    /// The full card from raw scores: top (5), SLSN branch (1 logit), anomaly space (6).
    pub fn report(
        &self,
        top_raw: &[f64],
        branch_raw: f64,
        ad_raw: &[f64],
        n_features_present: usize,
        base_rate: Option<f64>,
    ) -> Report {
        let p5 = softmax(top_raw);
        let (p6, pb) = self.proba(top_raw, branch_raw);
        let top = (0..5).max_by(|&a, &b| p5[a].partial_cmp(&p5[b]).unwrap()).unwrap();
        let label = if FIVE[top] == "SN_CC+" {
            if pb > self.branch_threshold { "SLSN" } else { "SN_CC" }
        } else {
            FIVE[top]
        }
        .to_string();
        let pv = self.conformal.p_values(&p6);
        let mut p_values = [0.0; 6];
        p_values.copy_from_slice(&pv);
        let set = self.conformal.prediction_set(&p6);
        let mut sorted = pv.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let energy = -logsumexp(ad_raw);
        let li = SIX.iter().position(|c| *c == label).unwrap_or(0);
        Report {
            proba: p6,
            argmax_excluded: !set.iter().any(|c| *c == label),
            coverage_measured: self.conformal.empirical_coverage[li],
            coverage_n: self.conformal.n_cal_per_class[li],
            p_value_floor: self.conformal.p_value_floor[li],
            label,
            p_values,
            set,
            alpha: self.conformal.alpha,
            credibility: sorted[0],
            confidence: 1.0 - sorted[1],
            anomaly: self.anomaly.report(energy, base_rate),
            n_features_present,
        }
    }
}

/// Order a name → value map into a model's input vector (missing → NaN).
pub fn vector_from_map(feature_names: &[String], m: &HashMap<String, f64>) -> Vec<f64> {
    feature_names.iter().map(|n| *m.get(n).unwrap_or(&f64::NAN)).collect()
}

/// Light-curve features + caller-supplied context columns → one feature map.
pub fn feature_map(dets: &[Detection], context: &HashMap<String, f64>) -> Result<HashMap<String, f64>, FlareError> {
    let lc = LightCurve::from_detections(dets, HORIZON_DAYS).ok_or(FlareError::QualityCut)?;
    if !lc.passes_quality() {
        return Err(FlareError::QualityCut);
    }
    let mut f = features::extract(&lc);
    for (k, v) in context {
        f.insert(k.clone(), *v);
    }
    Ok(f)
}

/// The self-contained classifier: text models evaluated by [`crate::lgbm`].
pub struct Flare {
    pub top: LgbmModel,
    pub branch: LgbmModel,
    pub ad_space: LgbmModel,
    pub head: Head,
}

impl Flare {
    /// Load `top.txt`, `slsn_branch.txt`, `ad_space.txt`, `conformal.json`,
    /// `anomaly_calibration.json`, `model_card.json` from one directory.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, FlareError> {
        let d = dir.as_ref();
        let top = LgbmModel::from_file(d.join("top.txt"))?;
        let branch = LgbmModel::from_file(d.join("slsn_branch.txt"))?;
        let ad_space = LgbmModel::from_file(d.join("ad_space.txt"))?;
        if top.objective != Objective::Multiclass || top.num_class != 5 {
            return Err(FlareError::Artefact("top.txt must be a 5-class multiclass booster".into()));
        }
        if branch.objective != Objective::Binary {
            return Err(FlareError::Artefact("slsn_branch.txt must be a binary booster".into()));
        }
        if top.feature_names != branch.feature_names {
            return Err(FlareError::Artefact("top and branch feature orders differ".into()));
        }
        Ok(Flare { top, branch, ad_space, head: Head::from_dir(d)? })
    }

    /// Feature names the classifier expects (light curve + context columns).
    pub fn feature_names(&self) -> &[String] {
        &self.top.feature_names
    }

    pub fn features(&self, dets: &[Detection], context: &HashMap<String, f64>) -> Result<HashMap<String, f64>, FlareError> {
        feature_map(dets, context)
    }

    pub fn proba_from_map(&self, f: &HashMap<String, f64>) -> ([f64; 6], f64) {
        let x = self.top.vector_from_map(f);
        self.head.proba(&self.top.predict_raw(&x), self.branch.predict_raw(&x)[0])
    }

    pub fn energy_from_map(&self, f: &HashMap<String, f64>) -> f64 {
        -logsumexp(&self.ad_space.predict_raw(&self.ad_space.vector_from_map(f)))
    }

    pub fn report_from_map(&self, f: &HashMap<String, f64>, base_rate: Option<f64>) -> Report {
        let x = self.top.vector_from_map(f);
        let n_present = x.iter().filter(|v| v.is_finite()).count();
        let ad = self.ad_space.predict_raw(&self.ad_space.vector_from_map(f));
        self.head.report(&self.top.predict_raw(&x), self.branch.predict_raw(&x)[0], &ad, n_present, base_rate)
    }

    /// One call per alert: detections + context → report.
    pub fn predict(&self, dets: &[Detection], context: &HashMap<String, f64>, base_rate: Option<f64>) -> Result<Report, FlareError> {
        let f = self.features(dets, context)?;
        Ok(self.report_from_map(&f, base_rate))
    }
}
