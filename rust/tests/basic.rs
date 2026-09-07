use std::collections::HashMap;

use flare::lgbm::LgbmModel;
use flare::lightcurve::{detections_from_rows, LightCurve, HORIZON_DAYS};
use flare::{features, model::Flare};

const TINY_MODEL: &str = "tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=1
objective=binary sigmoid:1
feature_names=a b
feature_infos=[0:1] [0:1]
tree_sizes=100

Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
split_gain=1 1
threshold=0.5 0.25
decision_type=10 2
left_child=1 -1
right_child=-2 -3
leaf_value=-1.0 1.0 2.0
leaf_weight=1 1 1
leaf_count=1 1 1
internal_value=0 0
internal_weight=0 0
internal_count=3 2
is_linear=0
shrinkage=1


end of trees
";

#[test]
fn lgbm_missing_value_semantics() {
    let m = LgbmModel::from_str(TINY_MODEL).unwrap();
    assert_eq!(m.feature_names, vec!["a", "b"]);
    // node0: a <= 0.5 → node1, else leaf1 (1.0); missing_type NaN, default left
    assert_eq!(m.predict_raw(&[0.9, 0.0])[0], 1.0);
    assert_eq!(m.predict_raw(&[f64::NAN, 0.0])[0], -1.0); // NaN → default left → node1 → b<=0.25 → leaf0
    // node1: decision_type 2 → missing_type None: NaN becomes 0.0 → 0.0 <= 0.25 → leaf0
    assert_eq!(m.predict_raw(&[0.1, f64::NAN])[0], -1.0);
    assert_eq!(m.predict_raw(&[0.1, 0.9])[0], 2.0);
    let p = m.predict_proba(&[0.1, 0.9])[0];
    assert!((p - 1.0 / (1.0 + (-2.0f64).exp())).abs() < 1e-12);
}

fn synthetic() -> Vec<(f64, i64, f64, f64)> {
    // a rising-then-fading transient in g and r over 40 days
    let mut rows = vec![];
    for k in 0..12 {
        let t = 60000.0 + 3.0 * k as f64;
        let phase = (k as f64 - 4.0) / 6.0;
        let mag_r = 18.0 + 0.8 * phase * phase;
        rows.push((t, 2, mag_r, 0.05));
        rows.push((t + 0.02, 1, mag_r + 0.2 + 0.03 * k as f64, 0.06));
    }
    rows
}

#[test]
fn lightcurve_and_features() {
    let dets = detections_from_rows(&synthetic());
    let lc = LightCurve::from_detections(&dets, HORIZON_DAYS).unwrap();
    assert!(lc.passes_quality());
    assert_eq!(lc.counts, [12, 12, 0]);
    assert_eq!(lc.g_r.value.len(), 24, "every g and r point has a partner within 1.5 d");
    let f = features::extract(&lc);
    assert_eq!(f.len(), 160, "160 light-curve features");
    assert_eq!(f["n_obs_total"], 24.0);
    assert_eq!(f["n_bands"], 2.0);
    assert!(f["i_amplitude"].is_nan(), "absent band → NaN block");
    assert!(f["g_r_mean"] > 0.0, "g fainter than r");
    assert!(f["r_bazin_rise_time"].is_finite());
    // determinism of the Bazin fit
    let f2 = features::extract(&lc);
    assert_eq!(f["r_bazin_amplitude"], f2["r_bazin_amplitude"]);
}

#[test]
fn quality_cut() {
    let short = detections_from_rows(&synthetic()[..6]);
    let lc = LightCurve::from_detections(&short, HORIZON_DAYS).unwrap();
    assert!(!lc.passes_quality());
}

#[test]
#[ignore = "needs the bts6 artefacts of the Python repository"]
fn end_to_end_bts6() {
    let model = Flare::load("/projects/bcrv/asasli/later/flare/models/bts6").unwrap();
    let dets = detections_from_rows(&synthetic());
    let r = model.predict(&dets, &HashMap::new(), None).unwrap();
    let s: f64 = r.proba.iter().sum();
    assert!((s - 1.0).abs() < 1e-9);
    assert!(!r.set.is_empty() || r.argmax_excluded);
}
