# flare (Rust)

The FLARE photometric classifier (Sasli et al., the light-curve module of
AppleCiDEr) as a Rust crate for alert brokers. No Python, no ONNX runtime.

```
detections (jd, fid, magpsf, sigmapsf)  +  context columns
        │
        ▼  lightcurve.rs   float32-faithful events, per-band flux curves, colours
        ▼  features.rs     160 physics features (light-curve-feature 0.20 + Bazin)
        ▼  lgbm.rs         exact LightGBM text-model evaluator (double thresholds)
        ▼  model.rs        5-way top level · SLSN branch · anomaly energy
        ▼  calib.rs        Mondrian conformal set · prevalence-aware P(anomaly)
        │
        ▼  Report: probabilities, label, set, credibility, confidence, energy,
                   likelihood ratio, novelty p-value, argmax-excluded flag
```

## Two ways to run it

* **Self-contained**: `Flare::load(dir)` reads the LightGBM text models and
  evaluates them with the crate's exact evaluator (`lgbm.rs`).
* **Raw scores from elsewhere**: brokers that keep every model as ONNX run the
  three files written by `scripts/export_onnx.py` (double thresholds, raw sum,
  calibration in `metadata_props`) in their own runtime and pass the raw
  scores to `Head::report`. `Head::from_metadata` builds the head from the top
  model's `model_card`, `conformal` and `anomaly_calibration` metadata.

## Use

```rust
use flare::{lightcurve::detections_from_rows, model::Flare};
use std::collections::HashMap;

let model = Flare::load("models/bts6")?;                 // the Python package's artefacts
let dets = detections_from_rows(&[(60000.0, 1, 18.7, 0.05), (60000.1, 2, 18.5, 0.04) /* … */]);
let mut ctx = HashMap::new();
ctx.insert("host_sep".to_string(), 3.1);                 // absent columns → NaN, as in training
let report = model.predict(&dets, &ctx, None)?;
println!("{} {:?} energy {:.2}", report.label, report.set, report.anomaly.energy);
```

`Flare::load` reads `top.txt`, `slsn_branch.txt`, `ad_space.txt`, `conformal.json`,
`anomaly_calibration.json`, `model_card.json` — exactly what `flare/models/bts6/`
in the Python repository contains.

## CLI (parity tests)

```
flare features <cases.json>                  # JSON lines: {id, quality, features}
flare predict  --model-dir DIR <cases.json>  # JSON lines: {id, report}
flare lgbm     <model.txt> <matrix.json>     # raw scores per row
```

`../parity/dump_cases.py` writes the cases from the benchmark and the Python
package's own outputs; `../parity/compare.py` runs the three comparisons.

## Parity with the Python package (300 benchmark objects)

| block | agreement |
|---|---|
| LightGBM raw scores, 3 boosters × 3000 rows | exact (0.0) |
| 139 non-Bazin light-curve features + context | ≤ 1e-6 relative on every object |
| 21 Bazin columns | ≤ 3e-6 relative with `--features ceres`; the default MCMC gives equally good, different solutions |
| end-to-end (`--features ceres`) | label and conformal set agree on 246/246 objects; max |ΔP| = 0.0 |

## Bazin solver

The Python package fits Bazin with Ceres. Build this crate with
`--features ceres` to match it (this compiles Ceres, Eigen and glog from source,
about 10 minutes). Two build-environment notes: Ceres 2.2 auto-enables CUDA when
`/usr/local/cuda` exists, and glog installs to `lib64` on RHEL-family systems,
so pass a toolchain file:

```
echo 'set(USE_CUDA OFF CACHE BOOL "" FORCE)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "" FORCE)' > nocuda.cmake
CMAKE_TOOLCHAIN_FILE=$PWD/nocuda.cmake cargo build --release --features ceres
```

Without that feature the crate uses the
deterministic MCMC of `light-curve-feature`; a model deployed that way must be
trained on features produced by this crate.

## Speed

300 objects, features only, one thread, including JSON: 0.21 s with Ceres
(≈0.7 ms per object; 0.72 s with MCMC). Inference on the three boosters adds ≈0.1 ms.
