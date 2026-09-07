//! Exact evaluator for LightGBM text models (`Booster.save_model`, format v3/v4).
//!
//! Only numerical splits are supported (FLARE has no categorical features).
//! Thresholds stay in `f64` and the missing-value semantics follow LightGBM's
//! `Tree::NumericalDecision`, so raw scores equal `Booster.predict(raw_score=True)`
//! to floating-point round-off.

use std::collections::HashMap;

use crate::FlareError;

const CATEGORICAL_MASK: u8 = 1;
const DEFAULT_LEFT_MASK: u8 = 2;
const ZERO_THRESHOLD: f64 = 1e-35;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissingType {
    None,
    Zero,
    NaN,
}

#[derive(Clone, Debug)]
struct Tree {
    num_leaves: usize,
    split_feature: Vec<usize>,
    threshold: Vec<f64>,
    decision_type: Vec<u8>,
    left_child: Vec<i32>,
    right_child: Vec<i32>,
    leaf_value: Vec<f64>,
}

impl Tree {
    #[inline]
    fn missing_type(dt: u8) -> MissingType {
        match (dt >> 2) & 3 {
            0 => MissingType::None,
            1 => MissingType::Zero,
            _ => MissingType::NaN,
        }
    }

    fn predict(&self, x: &[f64]) -> f64 {
        if self.num_leaves == 1 {
            return self.leaf_value[0];
        }
        let mut node: i32 = 0;
        loop {
            let i = node as usize;
            let dt = self.decision_type[i];
            let mut fval = x[self.split_feature[i]];
            let mt = Self::missing_type(dt);
            if fval.is_nan() && mt != MissingType::NaN {
                fval = 0.0;
            }
            let go_left = if (mt == MissingType::Zero && fval.abs() <= ZERO_THRESHOLD)
                || (mt == MissingType::NaN && fval.is_nan())
            {
                dt & DEFAULT_LEFT_MASK != 0
            } else {
                fval <= self.threshold[i]
            };
            node = if go_left { self.left_child[i] } else { self.right_child[i] };
            if node < 0 {
                return self.leaf_value[(-node - 1) as usize];
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Objective {
    Multiclass,
    Binary,
    Other(String),
}

/// A parsed LightGBM model.
#[derive(Clone, Debug)]
pub struct LgbmModel {
    pub feature_names: Vec<String>,
    pub num_class: usize,
    pub objective: Objective,
    trees: Vec<Tree>,
    num_tree_per_iteration: usize,
}

fn parse_vec<T: std::str::FromStr>(s: &str) -> Result<Vec<T>, FlareError> {
    s.split_whitespace()
        .map(|t| t.parse::<T>().map_err(|_| FlareError::ModelParse(format!("bad number '{t}'"))))
        .collect()
}

impl LgbmModel {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, FlareError> {
        Self::from_str(&std::fs::read_to_string(path)?)
    }

    pub fn from_str(text: &str) -> Result<Self, FlareError> {
        let mut header: HashMap<&str, &str> = HashMap::new();
        let mut trees = Vec::new();
        let mut blocks = text.split("\nTree=");
        let head = blocks.next().ok_or_else(|| FlareError::ModelParse("empty model".into()))?;
        for line in head.lines() {
            if let Some((k, v)) = line.split_once('=') {
                header.insert(k.trim(), v.trim());
            }
        }
        let num_class: usize = header.get("num_class").unwrap_or(&"1").parse().unwrap_or(1);
        let num_tree_per_iteration: usize =
            header.get("num_tree_per_iteration").unwrap_or(&"1").parse().unwrap_or(1);
        let objective = match header.get("objective") {
            Some(o) if o.starts_with("multiclass") => Objective::Multiclass,
            Some(o) if o.starts_with("binary") => Objective::Binary,
            Some(o) => Objective::Other(o.to_string()),
            None => Objective::Other("unknown".into()),
        };
        let feature_names: Vec<String> = header
            .get("feature_names")
            .ok_or_else(|| FlareError::ModelParse("no feature_names".into()))?
            .split_whitespace()
            .map(str::to_string)
            .collect();

        for block in blocks {
            let block = match block.split("\nend of trees").next() {
                Some(b) => b,
                None => block,
            };
            let mut kv: HashMap<&str, &str> = HashMap::new();
            for line in block.lines().skip(1) {
                if let Some((k, v)) = line.split_once('=') {
                    kv.insert(k.trim(), v.trim());
                }
            }
            if kv.is_empty() {
                continue;
            }
            let num_leaves: usize = kv
                .get("num_leaves")
                .ok_or_else(|| FlareError::ModelParse("tree without num_leaves".into()))?
                .parse()
                .map_err(|_| FlareError::ModelParse("bad num_leaves".into()))?;
            let num_cat: usize = kv.get("num_cat").unwrap_or(&"0").parse().unwrap_or(0);
            if num_cat > 0 {
                return Err(FlareError::ModelParse("categorical splits are not supported".into()));
            }
            let leaf_value: Vec<f64> = parse_vec(kv.get("leaf_value").unwrap_or(&""))?;
            let tree = if num_leaves == 1 {
                Tree {
                    num_leaves,
                    split_feature: vec![],
                    threshold: vec![],
                    decision_type: vec![],
                    left_child: vec![],
                    right_child: vec![],
                    leaf_value,
                }
            } else {
                let decision_type: Vec<u8> = parse_vec(kv.get("decision_type").unwrap_or(&""))?;
                if decision_type.iter().any(|d| d & CATEGORICAL_MASK != 0) {
                    return Err(FlareError::ModelParse("categorical decision type".into()));
                }
                Tree {
                    num_leaves,
                    split_feature: parse_vec(kv.get("split_feature").unwrap_or(&""))?,
                    threshold: parse_vec(kv.get("threshold").unwrap_or(&""))?,
                    decision_type,
                    left_child: parse_vec(kv.get("left_child").unwrap_or(&""))?,
                    right_child: parse_vec(kv.get("right_child").unwrap_or(&""))?,
                    leaf_value,
                }
            };
            trees.push(tree);
        }
        if trees.is_empty() {
            return Err(FlareError::ModelParse("no trees".into()));
        }
        Ok(LgbmModel { feature_names, num_class, objective, trees, num_tree_per_iteration })
    }

    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Raw scores: one value per class (multiclass) or a single logit (binary).
    /// `x` must be ordered like `feature_names`; missing values are `NaN`.
    pub fn predict_raw(&self, x: &[f64]) -> Vec<f64> {
        let k = self.num_tree_per_iteration.max(1);
        let mut out = vec![0.0; k];
        for (i, t) in self.trees.iter().enumerate() {
            out[i % k] += t.predict(x);
        }
        out
    }

    /// Probabilities: softmax (multiclass) or sigmoid (binary, returned as a single value).
    pub fn predict_proba(&self, x: &[f64]) -> Vec<f64> {
        let raw = self.predict_raw(x);
        match self.objective {
            Objective::Multiclass => softmax(&raw),
            Objective::Binary => vec![1.0 / (1.0 + (-raw[0]).exp())],
            Objective::Other(_) => raw,
        }
    }

    /// Reorder a name → value map into this model's feature order (missing → NaN).
    pub fn vector_from_map(&self, m: &HashMap<String, f64>) -> Vec<f64> {
        self.feature_names.iter().map(|n| *m.get(n).unwrap_or(&f64::NAN)).collect()
    }
}

pub fn softmax(raw: &[f64]) -> Vec<f64> {
    let mx = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = raw.iter().map(|r| (r - mx).exp()).collect();
    let s: f64 = ex.iter().sum();
    ex.iter().map(|e| e / s).collect()
}

pub fn logsumexp(raw: &[f64]) -> f64 {
    let mx = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    mx + raw.iter().map(|r| (r - mx).exp()).sum::<f64>().ln()
}
