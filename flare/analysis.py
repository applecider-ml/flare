"""Offline BTS6 analysis using only detections through an explicit cutoff.

Forecasts and Ia PE use the exact saved model horizons. No research scripts or
network services are imported here. Uncertainty metadata travels with outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .config import BTS6, PKG_ROOT
from .features import extract_from_array


def json_safe(value):
    """Convert numerical results to strict JSON values (missing -> null)."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def observed_events(events, horizon_days):
    if not np.isfinite(horizon_days) or not 0 < horizon_days <= 100:
        raise ValueError("horizon_days must be in (0, 100]")
    a = np.asarray(events, dtype=float)
    if a.ndim != 2 or a.shape[1] != len(D.COL):
        raise ValueError("events must have the 15 FLARE columns")
    a = a[(a[:, 0] >= 0) & (a[:, 0] <= horizon_days)]
    required = a[:, [0, D.COL['band_id'], D.COL['logflux'], D.COL['logflux_err']]]
    if not np.isfinite(required).all() or np.any(a[:, D.COL['logflux_err']] < 0):
        raise ValueError("observed times, bands, logflux and errors must be finite; errors nonnegative")
    if not np.isin(a[:, D.COL['band_id']], [0, 1, 2]).all():
        raise ValueError("band_id must be 0, 1 or 2")
    if not len(a):
        raise ValueError("no detections within the observation cutoff")
    return a[np.argsort(a[:, 0], kind="stable")]


def recompute_observed_colors(events):
    """Compatibility wrapper for the shared prefix-only color reconstruction."""
    return D.recompute_colors(events)


def feature_frame(events, context=None):
    """Recompute brightness-dependent context from the truncated events.

    Supply z_phot to compute M_pseudo. A precomputed M_pseudo without z_phot
    is discarded because its observation cutoff cannot be established.
    """
    from .context import m_pseudo
    ctx = dict(context or {})
    ctx.pop("M_pseudo", None)
    peak = float(23.9 - 2.5 * events[:, D.COL['logflux']].max())
    z = ctx.get("z_phot")
    ctx["M_pseudo"] = m_pseudo(peak, z) if z is not None and np.isfinite(z) and z > 0 else np.nan
    row = extract_from_array(events)
    # Context is allowed to fill external columns only, never LC features.
    row.update({k: ctx.get(k, np.nan) for k in BTS6.external_cols})
    return pd.DataFrame([row]), peak


def reconstruct(events):
    """Matérn-3/2 interpolation inside each observed band span.

    Raw GP predictive standard deviations are model uncertainties. Pooled
    empirical multipliers measured in the original 412-object study are
    reported separately, without asserting conditional or early-time coverage.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    out = {}
    for name, lc in D.reconstruct_bands(events).items():
        if name == "i":
            continue
        if len(lc.t) < 3:
            out[name] = dict(status="insufficient_data", reason="needs three distinct epochs")
            continue
        kernel = ConstantKernel(1., (1e-2, 1e2)) * Matern(20., (1., 300.), nu=1.5) + WhiteKernel(1e-3, (1e-6, 1.))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=(2.5 * lc.logflux_err)**2,
                                      normalize_y=True, n_restarts_optimizer=2, random_state=0)
        gp.fit(lc.t[:, None], 23.9 - 2.5 * lc.logflux)
        days = np.linspace(lc.t[0], lc.t[-1], 101)
        mean, sd = gp.predict(days[:, None], return_std=True)
        out[name] = dict(status="ok", days=days, magnitude=mean, std_mag=sd,
                         lower68=mean - 1.211 * sd, upper68=mean + 1.211 * sd,
                         lower95=mean - 3.222 * sd, upper95=mean + 3.222 * sd)
    return json_safe(dict(status="ok" if any(v['status'] == 'ok' for v in out.values()) else "insufficient_data",
                          method="Matern-3/2 GP", bands=out,
                          calibration="Pooled empirical factors from 2707 held-out detections in the original 412-object study; early-time and per-class coverage unvalidated",
                          units="AB mag", time_origin="first detection"))


class FlareAnalysis:
    """Reusable offline classifier, anomaly scorer, forecaster and Ia estimator."""

    def __init__(self, classifier=None, model_dir=None, num_threads=2):
        self._automatic_classifier = classifier is None
        if classifier is None:
            from . import load_classifier
            classifier = load_classifier("bts6")
        self.classifier = classifier
        self.model_dir = Path(model_dir) if model_dir else PKG_ROOT / "models/bts6"
        self.num_threads = num_threads
        self._models = {}
        self._manifest = None
        self._early_classifiers = {}
        self._blend_forecaster = None

    def _classifier_at(self, horizon):
        if self._automatic_classifier and horizon in (5, 10, 20):
            from . import load_classifier
            if horizon not in self._early_classifiers:
                self._early_classifiers[horizon] = load_classifier('bts6', horizon_days=int(horizon))
            return self._early_classifiers[horizon]
        return self.classifier

    def _checked_path(self, relative):
        if self._manifest is None:
            self._manifest = json.loads((self.model_dir / "forecast_pe_manifest.json").read_text())
        path = (self.model_dir / relative).resolve()
        if not path.is_relative_to(self.model_dir.resolve()):
            raise ValueError("model path escapes model directory")
        expected = self._manifest.get(relative)
        if expected is None or hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"model manifest mismatch: {relative}")
        return path

    def verify_models(self):
        self._checked_path("forecast/h5.json")
        for name in self._manifest:
            self._checked_path(name)
        return dict(status="ok", files=len(self._manifest))

    def _metadata(self, task, horizon):
        supported = (5, 10, 15, 20, 30) if task == "forecast" else (5, 10, 20)
        if horizon not in supported:
            raise ValueError(f"{task} horizon must be one of {supported}; no automatic rounding")
        return json.loads(self._checked_path(f"{task}/h{int(horizon)}.json").read_text())

    def _predict(self, task, filename, frame):
        key = f"{task}/{filename}"
        if key not in self._models:
            import lightgbm as lgb
            self._models[key] = lgb.Booster(model_file=str(self._checked_path(key)))
        model = self._models[key]
        if list(frame.columns) != model.feature_name():
            raise ValueError(f"feature order mismatch for {key}")
        return model.predict(frame, num_threads=self.num_threads)

    def forecast(self, events, horizon_days, context=None, recipe="grid", days=None):
        """Forecast with the original grid recipe or the optional day10/20 blend.

        For the blend, ``days`` can request exact future timestamps through day90.
        Its residual intervals are empirical pointwise bands, not simultaneous
        or class-conditional coverage guarantees.
        """
        if recipe not in ("grid", "blend"):
            raise ValueError("forecast recipe must be 'grid' or 'blend'")
        if recipe == "blend":
            if horizon_days not in (10, 20):
                raise ValueError("blend forecast horizon must be 10 or 20")
            from .forecast_blend import BlendForecaster
            arr = observed_events(events, horizon_days)
            X, peak = feature_frame(arr, context)
            if self._blend_forecaster is None:
                self._blend_forecaster = BlendForecaster(self.model_dir / 'forecast_blend', self.num_threads)
            return json_safe(self._blend_forecaster.predict(arr, X, peak, horizon_days, days))
        if days is not None:
            raise ValueError("explicit forecast days are supported only with recipe='blend'")
        meta = self._metadata("forecast", horizon_days)
        arr = observed_events(events, horizon_days)
        X, peak = feature_frame(arr, context)
        X = X.reindex(columns=meta['feature_names'])
        # Routing uses probability argmax, as documented in the saved model.
        classifier = self._classifier_at(horizon_days)
        cx = X.reindex(columns=classifier.feature_names)
        predicted = BTS6.classes[int(classifier.predict_proba(cx)[0].argmax())]
        role = "pooled_w" if predicted in ("SLSN", "AGN", "TDE", "CV") else "pooled"
        days = np.array(meta['grid_days'], dtype=float)
        frame = pd.concat([X.assign(fc_day=float(d), fc_band=float(b))
                           for b in (0, 1) for d in days], ignore_index=True)
        pred = {k: self._predict("forecast", meta['models'][k], frame).reshape(2, len(days))
                for k in (role, 'q16', 'q84')}
        scale = np.maximum(np.abs(pred['q84'] - pred['q16']) / 2, .05)
        mult = meta['conformal']['multipliers'][role]
        anchor = meta['conformal']['anchored_halfwidth_at_horizon']
        # cross-conformal hard-Mondrian multipliers by predicted class, band and grid day, when shipped
        mond = meta['conformal'].get('mondrian')
        if mond and predicted in mond['classes'] and list(mond['grid_days']) == list(meta['grid_days']):
            m68 = np.asarray(mond['classes'][predicted]['m68'], float); m95 = np.asarray(mond['classes'][predicted]['m95'], float)
            band_calibration = f"cross-conformal Mondrian by predicted class ({predicted})"
        else:
            m68 = np.full((2, len(days)), mult['m68']); m95 = np.full((2, len(days)), mult['m95'])
            band_calibration = "marginal conformal multiplier"

        bands = {}
        for b, name in enumerate(('g', 'r')):
            in_band = arr[arr[:, D.COL['band_id']] == b]
            if not len(in_band):
                bands[name] = dict(status="insufficient_data", reason="no detection to anchor this band")
                continue
            last = float(23.9 - 2.5 * in_band[-1, D.COL['logflux']])
            mean = np.r_[last, pred[role][b] + peak]
            hw68 = np.r_[anchor['hw68'], m68[b] * scale[b]]
            hw95 = np.r_[anchor['hw95'], m95[b] * scale[b]]
            bands[name] = dict(status="ok", days=np.r_[horizon_days, days], magnitude=mean,
                               lower68=mean-hw68, upper68=mean+hw68,
                               lower95=mean-hw95, upper95=mean+hw95)
        return json_safe(dict(status="ok" if any(v['status'] == 'ok' for v in bands.values()) else "insufficient_data",
                              model_horizon_days=horizon_days, observation_cutoff_days=horizon_days,
                              reference_magnitude=peak, model_role=role, routing_class=predicted, band_calibration=band_calibration,
                              bands=bands, units="AB mag", time_origin="first detection",
                              calibration=meta['note'], extrapolation="none beyond last grid point"))

    def pe(self, events, horizon_days, context=None):
        """Ia-conditional SALT3 estimates; caller must explicitly accept Ia assumption."""
        meta = self._metadata("pe_ia", horizon_days)
        arr = observed_events(events, horizon_days)
        X, _ = feature_frame(arr, context)
        X = X.reindex(columns=meta['feature_names'])
        values = {}
        for target, cfg in meta['targets'].items():
            value = float(self._predict("pe_ia", cfg['model'], X)[0])
            widths = cfg['conformal_halfwidth']
            values[target] = dict(value=value, interval68=[value-widths['q68'], value+widths['q68']],
                                  interval95=[value-widths['q95'], value+widths['q95']])
        return json_safe(dict(status="conditional", assumption="SN_Ia", model_horizon_days=horizon_days,
                              observation_cutoff_days=horizon_days, parameters=values,
                              units=dict(t0="days since first detection", mB="rest-frame B apparent AB mag", x1="dimensionless", c="dimensionless"),
                              labels=meta['labels'], calibration="Evaluated on quality-selected true Ia; classifier gate and non-Ia contamination not calibrated",
                              stretch_note="Weak early constraint; not validated for stretch-based standardization"))

    def analyze(self, events, horizon_days=100, context=None, forecast_horizon=None,
                pe_horizon=None, include_reconstruction=True, base_rate=None,
                forecast_recipe="grid"):
        if forecast_recipe not in ("grid", "blend"):
            raise ValueError("forecast_recipe must be 'grid' or 'blend'")
        arr = observed_events(events, horizon_days)
        X, _ = feature_frame(arr, context)
        classifier = self._classifier_at(horizon_days)
        cx = X.reindex(columns=classifier.feature_names)
        conf = classifier.conformal_report(cx)[0]
        conf['probabilities'] = dict(zip(BTS6.classes, classifier.predict_proba(cx)[0]))
        model_horizon = getattr(classifier, 'horizon_days', 100)
        conf['model_horizon_days'] = model_horizon
        matched = model_horizon == horizon_days
        conf['calibration_scope'] = f"BTS6 calibrated at day {model_horizon:g}" if matched else f"BTS6 calibrated at day {model_horizon:g}; coverage at this observation cutoff unvalidated"
        if not matched:
            conf['full_curve_empirical_coverage'] = conf.get('empirical_coverage')
            conf['empirical_coverage'] = None
            conf['nominal_error'] = conf.get('guaranteed_error')
            conf['guaranteed_error'] = None
        # the anomaly block of the SkyPortal service for the same model: phase-matched blind consensus
        # with its alarms (energy fields kept as diagnostics), so offline and online users get one signal
        from .skyportal import classify as service_classify, DEFAULTS
        anomaly = service_classify(X, DEFAULTS["base_rate"] if base_rate is None else base_rate,
                                   horizon=None if model_horizon == 100 else int(model_horizon))["anomaly"]
        anomaly['calibration_scope'] = (f"BTS6 day-{model_horizon:g} model" if matched
                                        else f"BTS6 day-{model_horizon:g} model; alarm calibration at this observation cutoff unvalidated")
        result = dict(observation_cutoff_days=horizon_days, n_detections=len(arr),
                      classification=conf, anomaly=anomaly)
        result['reconstruction'] = reconstruct(arr) if include_reconstruction else dict(status="disabled")
        for task, h in [('forecast', forecast_horizon), ('pe', pe_horizon)]:
            if h is None:
                result[task] = dict(status="disabled", reason="choose an explicit saved model horizon")
                continue
            if h > horizon_days:
                raise ValueError(f"{task} model horizon exceeds observation cutoff")
            if task == 'pe':
                self._metadata('pe_ia', h)
                # Gate uses exactly the same early observations as the PE model.
                gx, _ = feature_frame(observed_events(arr, h), context)
                gate_clf = self._classifier_at(h)
                gate = gate_clf.conformal_report(gx.reindex(columns=gate_clf.feature_names))[0]
                if gate['set'] != ['SN_Ia']:
                    result[task] = dict(status="abstained", reason="Ia is not the sole conformal candidate at the PE horizon", gate_set=gate['set'])
                    continue
            if task == 'forecast':
                result[task] = self.forecast(arr, h, context, recipe=forecast_recipe)
            else:
                result[task] = self.pe(arr, h, context)
        return json_safe(result)
