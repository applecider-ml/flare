//! Detections → the event representation FLARE was trained on.
//!
//! Mirrors `flare.fetch.to_events` and `flare.data.reconstruct_bands` /
//! `color_series` exactly, including the float32 storage of the training
//! arrays: every derived event quantity is rounded through `f32` before use,
//! so a broker computing from magnitudes and the training pipeline computing
//! from its `.npz` files agree to the last bit.

use serde::{Deserialize, Serialize};

/// AB zero point used to turn magnitudes into μJy.
pub const ZP: f64 = 23.9;
/// 1 / ln 10; logflux_err = flux_err / flux · LOG_CONST.
pub const LOG_CONST: f64 = 0.434_294_481_903_251_8;
/// A colour pairs detections in two bands closer than this many days.
pub const COLOR_WINDOW: f64 = 1.5;
/// Events later than this after the first detection are dropped.
pub const HORIZON_DAYS: f64 = 100.0;

pub const MIN_OBS_TOTAL: usize = 8;
pub const MIN_OBS_G: usize = 2;
pub const MIN_OBS_R: usize = 2;
pub const MIN_BANDS_OBSERVED: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Band {
    G = 0,
    R = 1,
    I = 2,
}

impl Band {
    /// ZTF `fid`: 1 = g, 2 = r, 3 = i.
    pub fn from_fid(fid: i64) -> Option<Band> {
        match fid {
            1 => Some(Band::G),
            2 => Some(Band::R),
            3 => Some(Band::I),
            _ => None,
        }
    }
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        match self {
            Band::G => "g",
            Band::R => "r",
            Band::I => "i",
        }
    }
    pub const ALL: [Band; 3] = [Band::G, Band::R, Band::I];
}

/// One PSF-photometry detection (alert `candidate` or `prv_candidates` entry).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Detection {
    pub mjd: f64,
    pub band: Band,
    pub mag: f64,
    pub mag_err: f64,
}

/// Single-band light curve in linear flux space (μJy), strictly increasing time.
#[derive(Clone, Debug, Default)]
pub struct BandLc {
    pub t: Vec<f64>,
    pub flux: Vec<f64>,
    pub flux_err: Vec<f64>,
    pub logflux: Vec<f64>,
    pub logflux_err: Vec<f64>,
}

/// A colour series (value = −2.5·(logf₁ − logf₂), i.e. m₁ − m₂) at event times.
#[derive(Clone, Debug, Default)]
pub struct ColorSeries {
    pub t: Vec<f64>,
    pub value: Vec<f64>,
    pub err: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct LightCurve {
    /// Number of events inside the horizon (all bands).
    pub n_events: usize,
    /// Largest `dt` inside the horizon.
    pub span_total: f64,
    /// Per-band curves, `None` when the band has no detection.
    pub bands: [Option<BandLc>; 3],
    pub g_r: ColorSeries,
    pub r_i: ColorSeries,
    /// Per-band detection counts (before time de-duplication), for the quality cut.
    pub counts: [usize; 3],
}

#[inline]
fn f32r(x: f64) -> f64 {
    // the training arrays are float32: round through f32 exactly as numpy did
    (x as f32) as f64
}

impl LightCurve {
    /// Build the light curve from detections, mirroring `to_events` +
    /// `reconstruct_bands` + `color_series`. Returns `None` when empty.
    pub fn from_detections(dets: &[Detection], horizon_days: f64) -> Option<LightCurve> {
        if dets.is_empty() {
            return None;
        }
        // sort by (mjd, band, mag, err) like Python's tuple sort
        let mut rows: Vec<Detection> = dets.to_vec();
        rows.sort_by(|a, b| {
            (a.mjd, a.band.index(), a.mag, a.mag_err)
                .partial_cmp(&(b.mjd, b.band.index(), b.mag, b.mag_err))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mjd0 = rows[0].mjd;
        let rows: Vec<Detection> = rows
            .into_iter()
            .filter(|d| d.mjd <= mjd0 + horizon_days)
            .collect();
        let n = rows.len();

        // event columns in f64, then rounded through f32 (numpy astype(float32))
        let mut dt = Vec::with_capacity(n);
        let mut logf = Vec::with_capacity(n);
        let mut logf_err = Vec::with_capacity(n);
        for d in &rows {
            let flux = 10f64.powf(-0.4 * (d.mag - ZP));
            let flux_err = d.mag_err * flux / (2.5 * LOG_CONST);
            let lf = flux.max(1e-6).log10();
            let lfe = flux_err * LOG_CONST / flux;
            dt.push(d.mjd - mjd0);
            logf.push(lf);
            logf_err.push(lfe);
        }
        let band: Vec<usize> = rows.iter().map(|d| d.band.index()).collect();
        let mjd: Vec<f64> = rows.iter().map(|d| d.mjd).collect();

        // colours: nearest other-band event within the window (f64, as Python)
        let colour = |i: usize, b1: usize, b2: usize| -> Option<(f64, f64)> {
            if band[i] != b1 && band[i] != b2 {
                return None;
            }
            let other = if band[i] == b1 { b2 } else { b1 };
            let mut best: Option<usize> = None;
            for j in 0..n {
                if band[j] == other {
                    let dj = (mjd[j] - mjd[i]).abs();
                    // np.argmin keeps the first minimum
                    match best {
                        None => best = Some(j),
                        Some(k) if dj < (mjd[k] - mjd[i]).abs() => best = Some(j),
                        _ => {}
                    }
                }
            }
            let j = best?;
            if (mjd[j] - mjd[i]).abs() > COLOR_WINDOW {
                return None;
            }
            let (lf1, lf2) = if band[i] == b1 { (logf[i], logf[j]) } else { (logf[j], logf[i]) };
            Some((-2.5 * (lf1 - lf2), 2.5 * logf_err[i].hypot(logf_err[j])))
        };

        let mut g_r = ColorSeries::default();
        let mut r_i = ColorSeries::default();
        for i in 0..n {
            if let Some((v, e)) = colour(i, 0, 1) {
                g_r.t.push(f32r(dt[i]));
                g_r.value.push(f32r(v));
                g_r.err.push(f32r(e));
            }
            if let Some((v, e)) = colour(i, 1, 2) {
                r_i.t.push(f32r(dt[i]));
                r_i.value.push(f32r(v));
                r_i.err.push(f32r(e));
            }
        }

        // per-band curves from the f32-rounded columns, de-duplicated in time
        let mut bands: [Option<BandLc>; 3] = [None, None, None];
        let mut counts = [0usize; 3];
        for b in 0..3 {
            let mut idx: Vec<usize> = (0..n).filter(|&i| band[i] == b).collect();
            counts[b] = idx.len();
            if idx.is_empty() {
                continue;
            }
            // stable sort by rounded dt
            idx.sort_by(|&i, &j| f32r(dt[i]).partial_cmp(&f32r(dt[j])).unwrap());
            let mut lc = BandLc::default();
            let mut last_t = f64::NEG_INFINITY;
            for &i in &idx {
                let t = f32r(dt[i]);
                if !(t > last_t) {
                    continue; // exact duplicate time: keep first
                }
                last_t = t;
                let lf = f32r(logf[i]);
                let lfe = f32r(logf_err[i]);
                let flux = 10f64.powf(lf);
                lc.t.push(t);
                lc.logflux.push(lf);
                lc.logflux_err.push(lfe);
                lc.flux.push(flux);
                lc.flux_err.push(lfe * flux / LOG_CONST);
            }
            bands[b] = Some(lc);
        }
        let span_total = dt.iter().map(|&x| f32r(x)).fold(f64::NEG_INFINITY, f64::max);
        Some(LightCurve { n_events: n, span_total, bands, g_r, r_i, counts })
    }

    /// The benchmark quality cut (`flare.data.passes_quality`).
    pub fn passes_quality(&self) -> bool {
        let observed = self.counts.iter().filter(|&&c| c > 0).count();
        self.n_events >= MIN_OBS_TOTAL
            && self.counts[0] >= MIN_OBS_G
            && self.counts[1] >= MIN_OBS_R
            && observed >= MIN_BANDS_OBSERVED
    }
}

/// Convenience: build from `(mjd, fid, magpsf, sigmapsf)` rows as brokers hold them.
pub fn detections_from_rows(rows: &[(f64, i64, f64, f64)]) -> Vec<Detection> {
    rows.iter()
        .filter_map(|&(mjd, fid, mag, err)| {
            Band::from_fid(fid).map(|band| Detection { mjd, band, mag, mag_err: err })
        })
        .collect()
}
