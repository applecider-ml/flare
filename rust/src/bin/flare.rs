//! CLI for parity tests and ad-hoc use.
//!
//!   flare features <cases.json>                      → JSON lines of feature maps
//!   flare predict  --model-dir DIR <cases.json>      → JSON lines of reports
//!   flare lgbm     <model.txt> <matrix.json>         → raw scores per row (parity)
//!
//! `cases.json`: [{"id": "...", "detections": [[mjd, fid, mag, err], ...],
//!                 "context": {"host_sep": 3.1, ...}}, ...]
//! `matrix.json`: {"feature_names": [...], "rows": [[...], ...]}  (null → NaN)

use std::collections::HashMap;

use serde::Deserialize;

use flare::lgbm::LgbmModel;
use flare::lightcurve::{detections_from_rows, LightCurve, HORIZON_DAYS};
use flare::model::Flare;
use flare::{features, FlareError};

#[derive(Deserialize)]
struct Case {
    id: String,
    detections: Vec<(f64, i64, f64, f64)>,
    #[serde(default)]
    context: HashMap<String, Option<f64>>,
}

#[derive(Deserialize)]
struct Matrix {
    feature_names: Vec<String>,
    rows: Vec<Vec<Option<f64>>>,
}

fn ctx(c: &Case) -> HashMap<String, f64> {
    c.context.iter().map(|(k, v)| (k.clone(), v.unwrap_or(f64::NAN))).collect()
}

fn main() -> Result<(), FlareError> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("features") => {
            let cases: Vec<Case> = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
            for c in cases {
                let dets = detections_from_rows(&c.detections);
                let out = match LightCurve::from_detections(&dets, HORIZON_DAYS) {
                    Some(lc) => {
                        let mut f = features::extract(&lc);
                        f.extend(ctx(&c));
                        serde_json::json!({"id": c.id, "quality": lc.passes_quality(), "features": f})
                    }
                    None => serde_json::json!({"id": c.id, "quality": false, "features": {}}),
                };
                println!("{}", serde_json::to_string(&out)?);
            }
        }
        Some("predict") => {
            let dir = args.iter().position(|a| a == "--model-dir").map(|i| args[i + 1].clone())
                .ok_or_else(|| FlareError::Artefact("--model-dir required".into()))?;
            let path = args.last().unwrap();
            let model = Flare::load(dir)?;
            let cases: Vec<Case> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            for c in cases {
                let dets = detections_from_rows(&c.detections);
                let out = match model.predict(&dets, &ctx(&c), None) {
                    Ok(r) => serde_json::json!({"id": c.id, "report": r}),
                    Err(e) => serde_json::json!({"id": c.id, "error": e.to_string()}),
                };
                println!("{}", serde_json::to_string(&out)?);
            }
        }
        Some("lgbm") => {
            let m = LgbmModel::from_file(&args[2])?;
            let mat: Matrix = serde_json::from_str(&std::fs::read_to_string(&args[3])?)?;
            let order: Vec<usize> = m
                .feature_names
                .iter()
                .map(|n| mat.feature_names.iter().position(|f| f == n)
                    .ok_or_else(|| FlareError::Artefact(format!("matrix lacks {n}"))))
                .collect::<Result<_, _>>()?;
            for row in mat.rows {
                let x: Vec<f64> = order.iter().map(|&i| row[i].unwrap_or(f64::NAN)).collect();
                println!("{}", serde_json::to_string(&m.predict_raw(&x))?);
            }
        }
        _ => {
            eprintln!("usage: flare features <cases.json> | flare predict --model-dir DIR <cases.json> | flare lgbm <model.txt> <matrix.json>");
            std::process::exit(2);
        }
    }
    Ok(())
}
