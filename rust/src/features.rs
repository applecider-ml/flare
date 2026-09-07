//! The 160 light-curve features of FLARE (`flare.features.extract_from_array`).
//!
//! Per band (g, r, i): cadence (`n_obs`, `span`, `gap_median`, `gap_max`),
//! peak (`t_peak`, `logflux_peak`, `logflux_last_minus_peak`,
//! `frac_obs_before_peak`), `snr_mean`, the 26 `light-curve-feature`
//! statistics on log-flux with weights 1/σ², and a Bazin fit in flux space
//! (`bazin_amplitude … bazin_reduced_chi2`, `bazin_fall_over_rise`).
//! Cross-band: g−r and r−i colour statistics, `gr_peak_dt`,
//! `gr_peak_logflux_ratio`, `gr_n_ratio`, `n_obs_total`, `n_bands`, `span_total`.
//!
//! Bazin fits use Ceres when the crate is built with `--features ceres`, which
//! is what the Python package used and what the shipped model was trained on;
//! without it the crate falls back to the deterministic MCMC of
//! `light-curve-feature`, and a model deployed that way must be trained on
//! features produced by this crate.

use std::collections::HashMap;
use std::sync::OnceLock;

use light_curve_feature::prelude::*;
use light_curve_feature::CurveFitAlgorithm;
#[cfg(feature = "ceres")]
use light_curve_feature::CeresCurveFit;

use crate::lightcurve::{Band, BandLc, ColorSeries, LightCurve, LOG_CONST};

const MIN_STAT_POINTS: usize = 4;
const MIN_BAZIN_POINTS: usize = 6;

fn stat_features() -> &'static Vec<Feature<f64>> {
    static F: OnceLock<Vec<Feature<f64>>> = OnceLock::new();
    F.get_or_init(|| {
        vec![
            Amplitude::new().into(),
            AndersonDarlingNormal::new().into(),
            BeyondNStd::new(1.0).into(),
            BeyondNStd::new(2.0).into(),
            Cusum::new().into(),
            EtaE::new().into(),
            ExcessVariance::new().into(),
            InterPercentileRange::new(0.10).into(),
            InterPercentileRange::new(0.25).into(),
            Kurtosis::new().into(),
            LinearFit::new().into(),
            LinearTrend::new().into(),
            MagnitudePercentageRatio::new(0.4, 0.05).into(),
            MagnitudePercentageRatio::new(0.2, 0.10).into(),
            MaximumSlope::new().into(),
            Mean::new().into(),
            MeanVariance::new().into(),
            MedianAbsoluteDeviation::new().into(),
            MedianBufferRangePercentage::new(0.10).into(),
            PercentAmplitude::new().into(),
            PercentDifferenceMagnitudePercentile::new(0.05).into(),
            ReducedChi2::new().into(),
            Skew::new().into(),
            StandardDeviation::new().into(),
            StetsonK::new().into(),
            WeightedMean::new().into(),
        ]
    })
}

fn bazin() -> &'static BazinFit {
    static B: OnceLock<BazinFit> = OnceLock::new();
    B.get_or_init(|| {
        #[cfg(feature = "ceres")]
        let algorithm: CurveFitAlgorithm = CeresCurveFit::default().into(); // Python: BazinFit(algorithm="ceres")
        #[cfg(not(feature = "ceres"))]
        let algorithm: CurveFitAlgorithm = BazinFit::default_algorithm();
        BazinFit::new(algorithm, BazinFit::default_ln_prior(), BazinInitsBounds::Default)
    })
}

/// Names of the statistical block as the Python package emits them (without band prefix).
pub fn stat_names() -> Vec<String> {
    stat_features()
        .iter()
        .flat_map(|f| f.get_names().into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect()
}

fn bazin_names() -> Vec<String> {
    bazin()
        .get_names()
        .into_iter()
        .map(|s| s.trim_start_matches("bazin_fit_").to_string())
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn band_features(lc: &BandLc, prefix: &str, out: &mut HashMap<String, f64>) {
    let n = lc.t.len();
    let key = |s: &str| format!("{prefix}_{s}");
    out.insert(key("n_obs"), n as f64);
    out.insert(key("span"), if n > 1 { lc.t[n - 1] - lc.t[0] } else { 0.0 });
    if n > 1 {
        let mut gaps: Vec<f64> = lc.t.windows(2).map(|w| w[1] - w[0]).collect();
        let gmax = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        out.insert(key("gap_median"), median(&mut gaps));
        out.insert(key("gap_max"), gmax);
    } else {
        out.insert(key("gap_median"), f64::NAN);
        out.insert(key("gap_max"), f64::NAN);
    }
    // np.argmax: first maximum
    let mut ipk = 0;
    for i in 1..n {
        if lc.flux[i] > lc.flux[ipk] {
            ipk = i;
        }
    }
    out.insert(key("t_peak"), lc.t[ipk]);
    out.insert(key("logflux_peak"), lc.logflux[ipk]);
    out.insert(key("logflux_last_minus_peak"), lc.logflux[n - 1] - lc.logflux[ipk]);
    out.insert(
        key("frac_obs_before_peak"),
        lc.t.iter().filter(|&&t| t < lc.t[ipk]).count() as f64 / n as f64,
    );
    out.insert(
        key("snr_mean"),
        lc.logflux_err.iter().map(|&e| LOG_CONST / e.max(1e-6)).sum::<f64>() / n as f64,
    );

    // statistics on log-flux, weights = 1/σ²  (per-feature NaN fill, as Python)
    let names = stat_names();
    if n >= MIN_STAT_POINTS {
        let w: Vec<f64> = lc.logflux_err.iter().map(|&e| 1.0 / (e * e)).collect();
        let mut ts = TimeSeries::new(&lc.t[..], &lc.logflux[..], &w[..]);
        let mut k = 0;
        for f in stat_features() {
            let vals = f.eval_or_fill(&mut ts, f64::NAN);
            for v in vals {
                out.insert(key(&names[k]), v);
                k += 1;
            }
        }
    } else {
        for nm in &names {
            out.insert(key(nm), f64::NAN);
        }
    }

    // Bazin fit in flux space
    let bnames = bazin_names();
    if n >= MIN_BAZIN_POINTS {
        let w: Vec<f64> = lc.flux_err.iter().map(|&e| 1.0 / (e * e)).collect();
        let mut ts = TimeSeries::new(&lc.t[..], &lc.flux[..], &w[..]);
        let vals = bazin().eval_or_fill(&mut ts, f64::NAN);
        for (nm, v) in bnames.iter().zip(vals) {
            out.insert(format!("{prefix}_bazin_{nm}"), v);
        }
        let rise = out[&format!("{prefix}_bazin_rise_time")];
        let fall = out[&format!("{prefix}_bazin_fall_time")];
        out.insert(
            format!("{prefix}_bazin_fall_over_rise"),
            if rise.is_finite() && fall.is_finite() && rise > 0.0 { fall / rise } else { f64::NAN },
        );
    } else {
        for nm in &bnames {
            out.insert(format!("{prefix}_bazin_{nm}"), f64::NAN);
        }
        out.insert(format!("{prefix}_bazin_fall_over_rise"), f64::NAN);
    }
}

fn color_features(c: &ColorSeries, name: &str, out: &mut HashMap<String, f64>) {
    let n = c.value.len();
    let key = |s: &str| format!("{name}_{s}");
    out.insert(key("n"), n as f64);
    if n == 0 {
        for s in ["mean", "wmean", "std", "slope", "early", "late", "late_minus_early"] {
            out.insert(key(s), f64::NAN);
        }
        return;
    }
    let (t, v, e) = (&c.t, &c.value, &c.err);
    let mean = v.iter().sum::<f64>() / n as f64;
    let w: Vec<f64> = e.iter().map(|&x| 1.0 / (x.max(1e-6) * x.max(1e-6))).collect();
    let wsum: f64 = w.iter().sum();
    let wmean = v.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / wsum;
    out.insert(key("mean"), mean);
    out.insert(key("wmean"), wmean);
    out.insert(
        key("std"),
        if n > 1 { (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt() } else { f64::NAN },
    );
    let tmin = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let tmax = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ptp = tmax - tmin;
    out.insert(
        key("slope"),
        if n > 2 && ptp > 0.0 {
            // ordinary least squares, as np.polyfit(t, v, 1)[0]
            let tm = t.iter().sum::<f64>() / n as f64;
            let sxy: f64 = t.iter().zip(v).map(|(a, b)| (a - tm) * (b - mean)).sum();
            let sxx: f64 = t.iter().map(|a| (a - tm).powi(2)).sum();
            sxy / sxx
        } else {
            f64::NAN
        },
    );
    let half = if n > 1 { tmin + ptp / 2.0 } else { t[0] };
    let early: Vec<f64> = t.iter().zip(v).filter(|(a, _)| **a <= half).map(|(_, b)| *b).collect();
    let late: Vec<f64> = t.iter().zip(v).filter(|(a, _)| **a > half).map(|(_, b)| *b).collect();
    let em = if early.is_empty() { f64::NAN } else { early.iter().sum::<f64>() / early.len() as f64 };
    let lm = if late.is_empty() { f64::NAN } else { late.iter().sum::<f64>() / late.len() as f64 };
    out.insert(key("early"), em);
    out.insert(key("late"), lm);
    out.insert(key("late_minus_early"), if em.is_finite() && lm.is_finite() { lm - em } else { f64::NAN });
}

/// All 160 light-curve features, keyed by the Python column names.
pub fn extract(lc: &LightCurve) -> HashMap<String, f64> {
    let mut out = HashMap::with_capacity(200);
    out.insert("n_obs_total".into(), lc.n_events as f64);
    out.insert("n_bands".into(), lc.bands.iter().filter(|b| b.is_some()).count() as f64);
    out.insert("span_total".into(), if lc.n_events > 0 { lc.span_total } else { f64::NAN });
    for b in Band::ALL {
        match &lc.bands[b.index()] {
            Some(band_lc) => band_features(band_lc, b.name(), &mut out),
            None => {
                // full NaN block with identical columns
                let dummy = BandLc {
                    t: vec![0.0, 1.0, 2.0, 3.0],
                    flux: vec![1.0; 4],
                    flux_err: vec![0.1; 4],
                    logflux: vec![0.0; 4],
                    logflux_err: vec![0.04; 4],
                };
                let mut tmp = HashMap::new();
                band_features(&dummy, b.name(), &mut tmp);
                for k in tmp.keys() {
                    out.insert(k.clone(), f64::NAN);
                }
            }
        }
    }
    color_features(&lc.g_r, "g_r", &mut out);
    color_features(&lc.r_i, "r_i", &mut out);
    if lc.bands[0].is_some() && lc.bands[1].is_some() {
        out.insert("gr_peak_dt".into(), out["g_t_peak"] - out["r_t_peak"]);
        out.insert("gr_peak_logflux_ratio".into(), out["g_logflux_peak"] - out["r_logflux_peak"]);
        out.insert("gr_n_ratio".into(), out["g_n_obs"] / out["r_n_obs"].max(1.0));
    } else {
        out.insert("gr_peak_dt".into(), f64::NAN);
        out.insert("gr_peak_logflux_ratio".into(), f64::NAN);
        out.insert("gr_n_ratio".into(), f64::NAN);
    }
    out
}
