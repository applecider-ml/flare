"""SkyPortal analysis-service adapter: the science half of a FLARE analysis job.

`analyze()` takes detections already parsed from a SkyPortal photometry payload
(mjd, fid, mag, magerr), the source redshift if SkyPortal has one, and the
service parameters, and returns the dictionary a SkyPortal analysis callback
carries: results, flat annotations, plot files. The thin bridge that lives in
skyportal/osg-skyportal-plugin only parses the payload and calls this, so the
logic is versioned here with the models it uses.

Parameters (all optional): horizon_days (100), context ("live" | "none"),
base_rate (0.0326), agent (False; LLM triage through applecider-agents, needs
ANTHROPIC_API_KEY), plot (True), redshift, ra, dec.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

DEFAULTS = {"horizon_days": 100.0, "context": "live", "base_rate": 0.0326, "agent": False, "plot": True, "min_snr": 3.0}

# FLARE's six classes -> the nearest label in SkyPortal's Sitewide Taxonomy (core
# collapse has no umbrella node there, Type II is the modal subtype; SLSN exists
# only as subtypes). Rows carry ml=True and a FLARE origin so they render as
# their own set, apart from other classifiers and from human labels.
SKYPORTAL_TAXONOMY = "Sitewide Taxonomy"
SKYPORTAL_ORIGIN = "FLARE"
FLARE_TO_TAXONOMY = {"SN_Ia": "Ia", "SN_CC": "Type II", "SLSN": "Ic-SLSN", "AGN": "AGN",
                     "TDE": "Tidal Disruption Event", "CV": "Cataclysmic", "SN": "Supernova"}


def _to_float(value):
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def resolve_coords(params: dict, resource_id: str):
    """ra/dec from analysis_parameters, else looked up by ZTF name (ALeRCE)."""
    ra, dec = _to_float(params.get("ra")), _to_float(params.get("dec"))
    if ra is not None and dec is not None:
        return ra, dec
    if str(resource_id).startswith("ZTF"):
        try:
            from .fetch import coords
            c = coords(resource_id)
            if c:
                return float(c[0]), float(c[1])
        except Exception:  # noqa: BLE001 — coordinates are optional; context degrades to NaN
            pass
    return None


def feature_frame(events, context: dict, z):
    """160 light-curve features from the truncated rows + the external columns
    the classifier was trained with; M_pseudo from the observed peak and z."""
    import numpy as np
    import pandas as pd
    from . import data as D
    from .config import BTS6
    from .context import m_pseudo
    from .features import extract_from_array

    peak = float(23.9 - 2.5 * events[:, D.COL["logflux"]].max())
    ctx = {k: v for k, v in (context or {}).items() if isinstance(v, (int, float)) and math.isfinite(v)}
    ctx.pop("M_pseudo", None)
    zz = z if z is not None else ctx.get("z_phot")
    ctx["M_pseudo"] = m_pseudo(peak, zz) if zz is not None and zz > 0 else np.nan
    row = extract_from_array(events)
    row.update({k: ctx.get(k, np.nan) for k in BTS6.external_cols})
    return pd.DataFrame([row]), peak, ctx


PHASE_BINS = ((7.5, 5), (15.0, 10), (35.0, 20))   # light-curve span upper bound (days) -> early model horizon
_CLASSIFIERS: dict = {}


def phase_horizon(span_days: float):
    """Which classifier sees this light curve: the 5, 10 or 20 day model (midpoint bins on the
    span of the detections) or None for the full-curve model above 35 days. The full-curve
    model applied to early inputs is what the Fritz audit of 2026-10-05 found wanting: on the
    paired benchmark at day 5 it has macro F1 0.42 and set coverage 0.49 against 0.63 and 0.90
    for the day-5 model, and it calls 5% of the supernovae CV against 0.2%."""
    for upper, h in PHASE_BINS:
        if span_days < upper:
            return h
    return None


def _classifier(horizon):
    """The full-curve classifier (horizon None) or an early one; an early classifier gets the
    full-curve AD booster plus its own phase anomaly calibration when early/h{h}/anomaly_calibration.json
    ships (energies of the validation split and of the rarities truncated at that horizon)."""
    if horizon not in _CLASSIFIERS:
        import json
        from . import load_classifier
        from .config import BTS6
        clf = load_classifier("bts6", horizon_days=horizon)
        if horizon is not None:
            import lightgbm as lgb
            full = _classifier(None)
            root = Path(BTS6.model_paths["top"]).parent / "early" / f"h{int(horizon)}"
            # the phase's own AD booster where it beat the full-curve one (day 5), else the full-curve booster
            clf.ad_space = lgb.Booster(model_file=str(root / "ad_space.txt")) if (root / "ad_space.txt").exists() else getattr(full, "ad_space", None)
            cal = root / "anomaly_calibration.json"
            clf.anomaly_cal = json.loads(cal.read_text()) if cal.exists() else None
        from .venn_abers import load_table
        from . import consensus
        root = Path(BTS6.model_paths["top"]).parent
        mdir = root if horizon is None else root / "early" / f"h{int(horizon)}"
        clf.va_table = load_table(mdir / "venn_abers.json")
        # blind consensus over the LUNA scorers, calibrated at this phase (None without luna-ood)
        clf.consensus = consensus.load(mdir)
        _CLASSIFIERS[horizon] = clf
    return _CLASSIFIERS[horizon]


def classify(X, base_rate: float, horizon=None) -> dict:
    """Probabilities, conformal set and p-values from the phase-matched classifier. The anomaly
    alarm comes from the phase-matched blind consensus when it ships (flare.consensus), at
    false-alarm rates of 1%, 5% and 10%; otherwise from the energy and its known population."""
    import numpy as np
    from .config import BTS6

    full = _classifier(None)
    clf = full if horizon is None else _classifier(horizon)
    Xc = X.reindex(columns=clf.feature_names)
    P = clf.predict_proba(Xc)[0]
    pv = clf.p_values(Xc)[0]
    sets = clf.prediction_sets(Xc, alpha=BTS6.alpha)[0]
    scorer = clf if getattr(clf, "anomaly_cal", None) else full      # phase calibration when shipped
    an = scorer.anomaly_probability(X.reindex(columns=full.feature_names), base_rate=base_rate)[0]
    known = np.asarray(scorer.anomaly_cal["known_energies"], dtype=float)
    calibrated = scorer is full and horizon is None or scorer is clf
    cons = None
    booster = getattr(clf, "ad_space", None) or getattr(full, "ad_space", None)
    if getattr(clf, "consensus", None) is not None and booster is not None:
        cons = clf.consensus.score(X.reindex(columns=full.feature_names), booster)[0]
    if cons is not None:
        alarm_p, alarms = cons["p_value"], cons["alarms"]
    else:       # energy: the conformal novelty p-value against its known population
        alarm_p = float(an.get("novelty_p", float("nan")))
        alarms = {a: bool(alarm_p <= a) for a in (0.01, 0.05, 0.10)}
    label = BTS6.classes[int(P.argmax())]
    order = np.argsort(pv)[::-1]
    n_cal_label = len(clf.cal_scores[label]) if getattr(clf, "cal_scores", None) else None
    va = None
    if getattr(clf, "va_table", None):
        # Venn-Abers interval for every class: [p0, p1] around the class probability, wider where
        # few calibration objects resemble this one (flare.venn_abers)
        from .venn_abers import intervals
        L, U, _ = intervals(P[None, :], clf.va_table, BTS6.classes)
        va = {c: [round(float(lo), 4), round(float(hi), 4)] for c, lo, hi in zip(BTS6.classes, L[0], U[0])}
    cov = (getattr(clf, "coverage", None) or {}).get(label)
    return {
        "predicted": label,
        # the four numbers a conformal output needs (paper, Sec. conformal): the nominal level, the coverage
        # measured for the predicted class with its binomial uncertainty, credibility and confidence
        "nominal_coverage": 1 - BTS6.alpha,
        "coverage_of_predicted": None if cov is None else round(float(cov), 3),
        "coverage_of_predicted_err": None if cov is None or not n_cal_label else round(float((cov * (1 - cov) / n_cal_label) ** .5), 3),
        "confidence": round(float(1 - pv[order[1]]), 4),
        "p_value_floor": None if not n_cal_label else round(1 / (n_cal_label + 1), 4),
        "model": "bts6" if horizon is None else f"bts6_h{horizon}",
        "model_horizon_days": 100 if horizon is None else horizon,
        "probabilities": {c: round(float(p), 4) for c, p in zip(BTS6.classes, P)},
        "probability_intervals": va,
        "prediction_set": list(sets),
        "alpha": BTS6.alpha,
        "p_values": {c: round(float(p), 4) for c, p in zip(BTS6.classes, pv)},
        "credibility": round(float(pv.max()), 4),
        "argmax_excluded_from_set": label not in sets,
        "anomaly": {
            "energy": round(float(an["energy"]), 4),
            "energy_percentile": round(float(an.get("energy_percentile", float("nan"))), 2),
            "novelty_p": round(float(an.get("novelty_p", float("nan"))), 4),
            "likelihood_ratio": round(float(an["likelihood_ratio"]), 3),
            "p_anomaly": round(float(an["p_anomaly"]), 4),
            "base_rate": base_rate,
            "threshold_1pct_far": round(float(np.percentile(known, 99)), 4),
            "above_1pct_far": bool(an["energy"] > np.percentile(known, 99)),
            # calibrated: the percentile, novelty p and 1% FAR threshold refer to a known population at
            # the same phase (full curves, or the validation split truncated at the model's horizon);
            # without a phase calibration the full-curve numbers are reported for the record only
            "calibrated": bool(calibrated or cons is not None),
            "calibration": ("full" if horizon is None else f"h{horizon}") if cons is not None
                           else ("full" if scorer is full else f"h{horizon}"),
            # the alarm: blind consensus when shipped, else energy; p is its conformal p-value and
            # alarm_far the smallest false-alarm rate (1%, 5%, 10%) at which the object is flagged
            "alarm_score": "consensus" if cons is not None else "energy",
            "consensus": None if cons is None else round(cons["consensus"], 4),
            "alarm_p": round(float(alarm_p), 4),
            "alarms": {f"{100 * a:g}%": v for a, v in alarms.items()},
            "alarm_far": min((a for a, v in alarms.items() if v), default=None),
        },
    }


SN_CLASSES = ("SN_Ia", "SN_CC", "SLSN")


def coarse_label(cls: dict):
    """'SN' when the 90% set has two or more members that are all supernova types: the
    headline then says supernova with the type unresolved, instead of nothing."""
    st = cls["prediction_set"]
    return "SN" if len(st) >= 2 and all(c in SN_CLASSES for c in st) else None


HOST_MISMATCH = {"host_sep_max": 5.0, "host_ext_min": 0.3, "point_sep_max": 1.5, "point_ext_max": 0.3}


def host_mismatch(cls: dict, ctx: dict) -> bool:
    """A CV call on an extended host (PS1 iPSF - iKron > 0.3 within 5 arcsec) with no point
    source within 1.5 arcsec underneath. On the BTS benchmark this configuration holds for
    1.3% of the CVs and 41% of the supernovae, so a CV label there is almost always a fast
    extragalactic transient that the six-class taxonomy has no name for (ZTF26abydsbe, 2026-10)."""
    if cls.get("predicted") != "CV":
        return False
    h = HOST_MISMATCH
    sep, ext = ctx.get("host_sep"), ctx.get("host_ext")
    if sep is None or ext is None or not (sep < h["host_sep_max"] and ext > h["host_ext_min"]):
        return False
    nsep, next_ = ctx.get("near_sep"), ctx.get("near_ext")
    point_under = (nsep is not None and nsep < h["point_sep_max"]
                   and next_ is not None and abs(next_) < h["point_ext_max"])
    return not point_under


MIN_DETECTIONS = 3            # below this nothing is said (insufficient_data)
ANOMALY_MIN_DETECTIONS = 8    # the full-curve energy threshold is trusted only from here


def rule_verdict(cls: dict, ctx: dict, n_det: int, n_g: int | None = None, n_r: int | None = None) -> dict:
    """The deterministic triage baseline. A headline class is reported only for a singleton
    conformal set (ordinary / needs_spectrum); a set of two or more is `ambiguous`; the
    CV label on an extended host is `host_mismatch`; insufficient_data below MIN_DETECTIONS.
    The 1% FAR alarm (consensus when shipped, else energy) counts only when it is calibrated
    at the model's phase and the object has ANOMALY_MIN_DETECTIONS."""
    a = cls["anomaly"]
    at_1pct = (a.get("alarms") or {}).get("1%", a["above_1pct_far"])
    alarm = bool(a.get("calibrated", True)) and n_det >= ANOMALY_MIN_DETECTIONS and bool(at_1pct)
    stellar = (ctx.get("parallax_over_error") or 0) > 3 or (ctx.get("pm_over_error") or 0) > 3
    nuclear = ctx.get("host_sep") is not None and ctx["host_sep"] < 0.5
    singleton = len(cls["prediction_set"]) == 1
    if n_det < MIN_DETECTIONS:
        v, p = "insufficient_data", 0
    elif stellar:
        v, p = "likely_stellar_or_agn", 1 if alarm else 0
    elif host_mismatch(cls, ctx):
        v, p = "host_mismatch", 2
    elif alarm or cls["argmax_excluded_from_set"] or cls["credibility"] < 0.1:
        v, p = "anomaly_review", 1
    elif (cls["predicted"] in ("SLSN", "TDE") and singleton) or ("TDE" in cls["prediction_set"] and nuclear and len(cls["prediction_set"]) <= 2):
        v, p = "needs_spectrum", 2
    elif not singleton:
        v, p = "ambiguous", 0
    else:
        v, p = "ordinary", 0
    return {"verdict": v, "priority": p, "source": "rules"}


def agent_verdict(resource_id: str, events, ctx: dict, X, peak, horizon_days: float) -> dict | None:
    """LLM triage through applecider-agents (the FLARE agent), seeded with this payload so nothing
    is re-fetched. Returns None when the package or the key is absent (the caller keeps the rule
    verdict). X and peak are recomputed by the agent's tools from the same events and context."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from applecider_agents.core import data as agent_data, loop
    except ImportError:
        return None
    agent_data.seed(resource_id, events, {**ctx, "_source": "skyportal"})
    rep = loop.triage("flare", resource_id, horizon_days=horizon_days)
    out = rep.model_dump()
    out["source"] = "agent"
    return out


def plot_lightcurve(events, cls: dict, resource_id: str, outdir: Path) -> list[str]:
    """One PNG: the light curve with the class probabilities in the corner."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from . import data as D
    except ImportError:
        return []
    bands = D.reconstruct_bands(events)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for b, c in (("g", "#2a9d8f"), ("r", "#e76f51"), ("i", "#8d5a97")):
        if b in bands:
            lc = bands[b]
            ax.errorbar(lc.t, 23.9 - 2.5 * lc.logflux, yerr=2.5 * lc.logflux_err, fmt="o", ms=3, color=c, label=b)
    ax.invert_yaxis(); ax.set_xlabel("days from first detection"); ax.set_ylabel("mag"); ax.legend(loc="upper right", fontsize=8)
    txt = "\n".join(f"{k} {v:.2f}" for k, v in sorted(cls["probabilities"].items(), key=lambda kv: -kv[1])[:3])
    ax.text(0.02, 0.05, f"FLARE: {cls['predicted'] or 'no class (' + str(cls.get('predicted_if_forced')) + ')'}  set {{{'|'.join(cls['prediction_set'])}}}\n{txt}\nanomaly p {cls['anomaly'].get('alarm_p', cls['anomaly']['novelty_p'])} ({cls['anomaly'].get('alarm_score', 'energy')})",
            transform=ax.transAxes, fontsize=8, va="bottom", family="monospace")
    ax.set_title(resource_id, fontsize=10)
    path = outdir / f"{resource_id}_flare.png"
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return [str(path)]


def annotations_for(cls: dict, verdict: dict) -> dict:
    """Flat annotation dict (human-readable summary numbers + the probability vector)."""
    a = cls["anomaly"]
    top = max(cls["probabilities"], key=cls["probabilities"].get)
    iv = (cls.get("probability_intervals") or {}).get(top)
    out = {**{f"flare_p_{k}": v for k, v in cls["probabilities"].items()},
        "flare_class": cls["predicted"],
        "flare_model": cls.get("model"),
        "flare_p_max": max(cls["probabilities"].values()),
        # Venn-Abers interval of the most probable class (flare_p_max): lower and upper probability
        "flare_p_low": iv[0] if iv else None,
        "flare_p_high": iv[1] if iv else None,
        "flare_set": "|".join(cls["prediction_set"]),
        "flare_credibility": cls["credibility"],
        "flare_confidence": cls.get("confidence"),
        "flare_coverage": cls.get("coverage_of_predicted"),
        "flare_energy_pct": a["energy_percentile"],
        "flare_novelty_p": a["novelty_p"],
        # the anomaly alarm: its p-value and the smallest false-alarm rate that flags the object
        "flare_anomaly_p": a.get("alarm_p"),
        "flare_alarm_far": a.get("alarm_far"),
        "flare_consensus": a.get("consensus"),
        "flare_verdict": verdict.get("verdict"),
        "flare_priority": verdict.get("priority"),
    }
    return {k: v for k, v in out.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}


def skyportal_annotations(cls: dict, verdict: dict) -> list:
    """The webhook form: a list of {origin, data}; a flat dict is silently dropped by SkyPortal."""
    return [{"origin": SKYPORTAL_ORIGIN, "data": annotations_for(cls, verdict)}]


def skyportal_classifications(cls: dict) -> list:
    """One ml classification of the predicted class on the Sitewide Taxonomy."""
    label = FLARE_TO_TAXONOMY.get(cls.get("predicted"))
    if not label:
        return []
    prob = cls.get("probabilities_coarse", {}).get(cls["predicted"], cls["probabilities"].get(cls["predicted"]))
    return [{"taxonomy": SKYPORTAL_TAXONOMY, "classification": label,
             "probability": prob, "ml": True, "origin": SKYPORTAL_ORIGIN}]




FORECAST_HORIZONS = (5, 10, 15, 20, 30)
SKYPORTAL_FILTER = {"g": "ztfg", "r": "ztfr"}


def forecast_overlay(rows, context: dict, z, params: dict) -> dict:
    """Forecast of the future g and r light curve from the FLARE analysis engine, shaped as
    SkyPortal's ``model_lightcurve`` overlay: {filter: [[mjd, mag, lower68, upper68], ...]}.

    The forecast day is the smallest shipped horizon (5, 10, 15, 20 or 30 d) that contains every
    detection so far, capped at 30 d (later detections are then not used); the blend recipe is used
    at 10 and 20 d. No forecast is returned once the detections span more than 90 d."""
    from .fetch import to_events
    from .analysis import FlareAnalysis
    rows = sorted(rows)
    first, last = float(rows[0][0]), float(rows[-1][0]); age = last - first
    if age > 90:
        return dict(status="skipped", reason="detections span more than 90 d")
    n_g = sum(1 for r in rows if int(r[1]) == 1); n_r = sum(1 for r in rows if int(r[1]) == 2)
    if len(rows) < 8 or n_g < 2 or n_r < 2:      # the same minimum as the classifier's insufficient_data rule
        return dict(status="skipped", reason=f"too few detections for a forecast ({len(rows)} total, {n_g} g, {n_r} r; need 8, with 2 in g and r)")
    h = int(params.get("forecast_horizon") or next((H for H in FORECAST_HORIZONS if H >= age), FORECAST_HORIZONS[-1]))
    recipe = "blend" if h in (10, 20) else "grid"
    events = to_events(list(rows), horizon_days=100.0)
    ctx = {k: v for k, v in (context or {}).items() if isinstance(v, (int, float)) and math.isfinite(v)}
    if z is not None:
        ctx["z_phot"] = float(z)
    out = FlareAnalysis().forecast(events, h, context=ctx, recipe=recipe)
    curves = {}
    for band, name in SKYPORTAL_FILTER.items():
        b = (out.get("bands") or {}).get(band) or {}
        if b.get("status", "ok") != "ok" or "magnitude" not in b:
            continue
        days, mag = b["days"], b["magnitude"]; lo, hi = b.get("lower68", mag), b.get("upper68", mag)
        curves[name] = [[round(first + float(d), 5), round(float(m), 4), round(float(l), 4), round(float(u), 4)]
                        for d, m, l, u in zip(days, mag, lo, hi) if m is not None and math.isfinite(float(m))]
    return dict(status="ok" if curves else "insufficient_data", forecast_horizon_days=h, recipe=recipe,
                routing_class=out.get("routing_class"), band_calibration=out.get("band_calibration"),
                model_lightcurve=curves, detections_used_until_day=min(age, h),
                note="68% band; the first point of each curve is the last detection on the forecast day")


def analyze(rows, redshift, params: dict | None, resource_id: str = "obj", work_dir: str = ".") -> dict:
    """rows: sorted (mjd, fid, mag, magerr) detections; redshift: SkyPortal value or None."""
    from .fetch import to_events

    params = {**DEFAULTS, **(params or {})}
    horizon = float(params["horizon_days"])
    from .fetch import detections
    n_input = len(list(rows))
    rows = sorted(detections(list(rows), params.get("min_snr")))
    events = to_events(list(rows), horizon_days=horizon)
    if events is None or len(events) < 2:
        raise ValueError("too few detections inside the horizon")
    z = _to_float(params.get("redshift"))
    if z is None:
        z = redshift
    context: dict = {}
    if str(params["context"]).lower() != "none":
        c = resolve_coords(params, resource_id)
        if c:
            try:
                from .context import context_features
                # Pre-staged cross-matches (e.g. from BOOM) -> run the context block
                # fully offline, no MAST/Data Lab calls; absent -> live (unchanged).
                provided = params.get("context_data")
                context = context_features(c[0], c[1], peak_mag=None, provided=provided)
            except Exception as e:  # noqa: BLE001 — context is optional by design; the model tolerates NaN
                context = {"_error": f"{type(e).__name__}: {e}"}
    X, peak, ctx = feature_frame(events, context, z)
    span = float(events[:, 0].max() - events[:, 0].min()) if len(events) else 0.0
    h_model = phase_horizon(span) if _as_bool(params.get("phase_models"), True) else None
    cls = classify(X, float(params["base_rate"]), horizon=h_model)
    bands = events[:, 2].astype(int) if len(events) else []
    n_g, n_r = int((bands == 0).sum()), int((bands == 1).sum())
    verdict = rule_verdict(cls, ctx, int(len(events)), n_g, n_r)
    # A specific class is posted only when the 90% set is exactly that class, whatever the verdict:
    # under anomaly_review the argmax is often excluded by its own set (the conformal echo of the anomaly
    # score), and under likely_stellar_or_agn the set may still be open. Too few points or a label that
    # contradicts the host post nothing; otherwise a set that is all supernova types posts the coarse
    # "SN" (taxonomy "Supernova") unless coarse_labels is off. The probabilities and the set stay in the
    # JSON and the annotation; predicted = None means the bridge posts no ML Classification.
    singleton_ok = len(cls["prediction_set"]) == 1 and cls["predicted"] in cls["prediction_set"]
    withheld = verdict["verdict"] in ("insufficient_data", "host_mismatch") or not singleton_ok
    if withheld:
        coarse = (coarse_label(cls) if verdict["verdict"] not in ("insufficient_data", "host_mismatch")
                  and _as_bool(params.get("coarse_labels"), True) else None)
        cls = {**cls, "predicted_if_forced": cls["predicted"], "predicted": coarse}
        if coarse:
            cls["probabilities_coarse"] = {"SN": round(sum(cls["probabilities"][c] for c in SN_CLASSES), 4)}
    if _as_bool(params.get("agent")):
        rep = agent_verdict(resource_id, events, ctx, X, peak, horizon)
        if rep:
            verdict = {k: rep[k] for k in ("verdict", "priority", "summary", "evidence", "suggested_action", "caveats")}
            verdict["source"] = "agent"
    work = Path(work_dir).resolve(); work.mkdir(parents=True, exist_ok=True)
    plots = plot_lightcurve(events, cls, resource_id, work) if _as_bool(params.get("plot"), True) else []
    a = cls["anomaly"]
    setstr = "{" + "|".join(cls["prediction_set"]) + "}"
    far = a.get("alarm_far")
    flagged = f"flagged at {100 * far:g}% FAR" if far else "not flagged at 10% FAR"
    anom = (f"anomaly {a.get('alarm_score', 'energy')} p={a.get('alarm_p', a['novelty_p']):.3f}, {flagged} "
            f"({a.get('calibration', 'full')} calibration)" if a.get("calibrated", True)
            else f"anomaly score not calibrated at this phase (energy percentile {a['energy_percentile']:.1f} for the record)")
    head = f"FLARE ({cls['model']}, {len(events)} detections over {span:.1f} d)"
    if verdict["verdict"] == "ambiguous" and cls["predicted"] == "SN":
        message = (f"{head}: supernova (p={cls['probabilities_coarse']['SN']:.2f}), type unresolved, 90% set {setstr}, "
                   f"most probable {cls['predicted_if_forced']} (p={max(cls['probabilities'].values()):.2f}); {anom}")
    elif verdict["verdict"] == "ambiguous":
        message = (f"{head}: 90% set {setstr}, most probable {cls['predicted_if_forced']} "
                   f"(p={max(cls['probabilities'].values()):.2f}); no single class yet; {anom}")
    elif verdict["verdict"] == "host_mismatch":
        message = (f"FLARE: fast transient, class unresolved; the light curve looks CV-like "
                   f"(p={max(cls['probabilities'].values()):.2f}) but sits {ctx['host_sep']:.1f}\" from an extended source "
                   f"(PSF-Kron {ctx['host_ext']:.2f}) with no point source underneath, so no class is reported; "
                   f"credibility {cls['credibility']:.2f}, {anom}; "
                   f"triage host_mismatch (priority {verdict['priority']}): spectrum recommended")
    elif cls["predicted"] is None and verdict["verdict"] not in ("insufficient_data",):
        message = (f"{head}: 90% set {setstr} does not single out the most probable class "
                   f"{cls['predicted_if_forced']} (p={max(cls['probabilities'].values()):.2f}); no class reported; "
                   f"{anom}; triage {verdict['verdict']} (priority {verdict['priority']})")
    elif cls["predicted"] is None:
        message = (f"{head}: insufficient data (need {MIN_DETECTIONS}); no class reported "
                   f"(the most probable would be {cls['predicted_if_forced']}, p={max(cls['probabilities'].values()):.2f}); {anom}")
    else:
        message = (f"{head}: {cls['predicted']} (p={max(cls['probabilities'].values()):.2f}), 90% set {setstr}, "
                   f"credibility {cls['credibility']:.2f}, {anom}; triage {verdict['verdict']} (priority {verdict['priority']})")
    forecast = None
    if verdict["verdict"] == "host_mismatch":
        forecast = dict(status="skipped", reason="no headline class: the forecast models are trained on the six classes")
    elif _as_bool(params.get("forecast"), True):
        try:
            forecast = forecast_overlay(rows, ctx, z, params)
        except Exception as e:  # noqa: BLE001 — the forecast is optional; classification and anomaly stand on their own
            forecast = dict(status="error", reason=f"{type(e).__name__}: {e}")
    result = {
        "status": "success", "message": message,
        "results": {"resource_id": resource_id, "n_detections": int(len(events)), "n_rows_input": int(n_input),
                    "detection_floor": f"SNR >= {params.get('min_snr')} (magerr <= {1.0857 / float(params['min_snr']):.3f})" if params.get("min_snr") else None,
                    "horizon_days": horizon, "span_days": round(span, 2), "model": cls["model"],
                    "peak_mag_observed": round(peak, 3), "redshift_used": z if z is not None else ctx.get("z_phot"),
                    "redshift_source": "skyportal" if z is not None else ("host photo-z" if ctx.get("z_phot") else None),
                    "classification": cls, "context": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ctx.items()},
                    "context_error": context.get("_error"), "triage": verdict},
        "annotations": skyportal_annotations(cls, verdict), "classifications": skyportal_classifications(cls),
        "annotations_flat": annotations_for(cls, verdict), "plot_files": plots,
    }
    if forecast is not None:
        result["results"]["forecast"] = {k: v for k, v in forecast.items() if k != "model_lightcurve"}
        if forecast.get("model_lightcurve"):
            result["model_lightcurve"] = forecast["model_lightcurve"]
    return result
