"""SkyPortal analysis-service adapter: the science half of a FLARE analysis job.

`analyze()` takes detections already parsed from a SkyPortal photometry payload
(mjd, fid, mag, magerr), the source redshift if SkyPortal has one, and the
service parameters, and returns the dictionary a SkyPortal analysis callback
carries: results, flat annotations, plot files. The thin bridge that lives in
skyportal/osg-skyportal-plugin only parses the payload and calls this, so the
logic is versioned here with the models it uses.

Parameters (all optional): horizon_days (100), context ("live" | "none"),
base_rate (0.0326), agent (False; LLM triage through agents-flare, needs
ANTHROPIC_API_KEY), plot (True), redshift, ra, dec.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

DEFAULTS = {"horizon_days": 100.0, "context": "live", "base_rate": 0.0326, "agent": False, "plot": True}

# FLARE's six classes -> the nearest label in SkyPortal's Sitewide Taxonomy (core
# collapse has no umbrella node there, Type II is the modal subtype; SLSN exists
# only as subtypes). Rows carry ml=True and a FLARE origin so they render as
# their own set, apart from other classifiers and from human labels.
SKYPORTAL_TAXONOMY = "Sitewide Taxonomy"
SKYPORTAL_ORIGIN = "FLARE"
FLARE_TO_TAXONOMY = {"SN_Ia": "Ia", "SN_CC": "Type II", "SLSN": "Ic-SLSN", "AGN": "AGN",
                     "TDE": "Tidal Disruption Event", "CV": "Cataclysmic"}


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


def classify(X, base_rate: float) -> dict:
    import numpy as np
    from .config import BTS6
    from .hierarchical import HierarchicalFlare

    clf = HierarchicalFlare.from_pretrained(BTS6)
    X = X.reindex(columns=clf.feature_names)
    P = clf.predict_proba(X)[0]
    pv = clf.p_values(X)[0]
    sets = clf.prediction_sets(X, alpha=BTS6.alpha)[0]
    an = clf.anomaly_probability(X, base_rate=base_rate)[0]
    known = np.asarray(clf.anomaly_cal["known_energies"], dtype=float)
    label = BTS6.classes[int(P.argmax())]
    return {
        "predicted": label,
        "probabilities": {c: round(float(p), 4) for c, p in zip(BTS6.classes, P)},
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
        },
    }


def rule_verdict(cls: dict, ctx: dict, n_det: int) -> dict:
    """The deterministic triage baseline (same rules as agents_flare.agent.mock_triage)."""
    a = cls["anomaly"]
    stellar = (ctx.get("parallax_over_error") or 0) > 3 or (ctx.get("pm_over_error") or 0) > 3
    nuclear = ctx.get("host_sep") is not None and ctx["host_sep"] < 0.5
    if n_det < 8:
        v, p = "insufficient_data", 0
    elif stellar:
        v, p = "likely_stellar_or_agn", 1 if a["above_1pct_far"] else 0
    elif a["above_1pct_far"] or cls["argmax_excluded_from_set"] or cls["credibility"] < 0.1:
        v, p = "anomaly_review", 1
    elif cls["predicted"] in ("SLSN", "TDE") or ("TDE" in cls["prediction_set"] and nuclear):
        v, p = "needs_spectrum", 2
    else:
        v, p = "ordinary", 0
    return {"verdict": v, "priority": p, "source": "rules"}


def agent_verdict(resource_id: str, events, ctx: dict, X, peak, horizon_days: float) -> dict | None:
    """LLM triage through agents-flare, seeded with this payload so nothing is re-fetched.
    Returns None when the package or the key is absent (the caller keeps the rule verdict)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from agents_flare import tools as T
        from agents_flare.agent import triage
    except ImportError:
        return None
    T._CACHE[(resource_id, float(horizon_days))] = (events, {**ctx, "_source": "skyportal"}, X, peak)
    rep = triage(resource_id, horizon_days=horizon_days)
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
    ax.text(0.02, 0.05, f"FLARE: {cls['predicted']}  set {{{'|'.join(cls['prediction_set'])}}}\n{txt}\nenergy pct {cls['anomaly']['energy_percentile']}",
            transform=ax.transAxes, fontsize=8, va="bottom", family="monospace")
    ax.set_title(resource_id, fontsize=10)
    path = outdir / f"{resource_id}_flare.png"
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return [str(path)]


def annotations_for(cls: dict, verdict: dict) -> dict:
    """Flat annotation dict (human-readable summary numbers + the probability vector)."""
    a = cls["anomaly"]
    out = {**{f"flare_p_{k}": v for k, v in cls["probabilities"].items()},
        "flare_class": cls["predicted"],
        "flare_p_max": max(cls["probabilities"].values()),
        "flare_set": "|".join(cls["prediction_set"]),
        "flare_credibility": cls["credibility"],
        "flare_energy_pct": a["energy_percentile"],
        "flare_novelty_p": a["novelty_p"],
        "flare_verdict": verdict.get("verdict"),
        "flare_priority": verdict.get("priority"),
    }
    return {k: v for k, v in out.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}


def skyportal_annotations(cls: dict, verdict: dict) -> list:
    """The webhook form: a list of {origin, data}; a flat dict is silently dropped by SkyPortal."""
    return [{"origin": SKYPORTAL_ORIGIN, "data": annotations_for(cls, verdict)}]


def skyportal_classifications(cls: dict) -> list:
    """One ml classification of the predicted class on the Sitewide Taxonomy."""
    label = FLARE_TO_TAXONOMY.get(cls["predicted"])
    if not label:
        return []
    return [{"taxonomy": SKYPORTAL_TAXONOMY, "classification": label,
             "probability": cls["probabilities"][cls["predicted"]], "ml": True, "origin": SKYPORTAL_ORIGIN}]



def analyze(rows, redshift, params: dict | None, resource_id: str = "obj", work_dir: str = ".") -> dict:
    """rows: sorted (mjd, fid, mag, magerr) detections; redshift: SkyPortal value or None."""
    from .fetch import to_events

    params = {**DEFAULTS, **(params or {})}
    horizon = float(params["horizon_days"])
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
                context = context_features(c[0], c[1], peak_mag=None)
            except Exception as e:  # noqa: BLE001 — context is optional by design; the model tolerates NaN
                context = {"_error": f"{type(e).__name__}: {e}"}
    X, peak, ctx = feature_frame(events, context, z)
    cls = classify(X, float(params["base_rate"]))
    verdict = rule_verdict(cls, ctx, int(len(events)))
    if _as_bool(params.get("agent")):
        rep = agent_verdict(resource_id, events, ctx, X, peak, horizon)
        if rep:
            verdict = {k: rep[k] for k in ("verdict", "priority", "summary", "evidence", "suggested_action", "caveats")}
            verdict["source"] = "agent"
    work = Path(work_dir).resolve(); work.mkdir(parents=True, exist_ok=True)
    plots = plot_lightcurve(events, cls, resource_id, work) if _as_bool(params.get("plot"), True) else []
    a = cls["anomaly"]
    message = (f"FLARE: {cls['predicted']} (p={max(cls['probabilities'].values()):.2f}), 90% set {{{'|'.join(cls['prediction_set'])}}}, "
               f"credibility {cls['credibility']:.2f}, anomaly energy percentile {a['energy_percentile']:.1f}; triage {verdict['verdict']} (priority {verdict['priority']})")
    return {
        "status": "success", "message": message,
        "results": {"resource_id": resource_id, "n_detections": int(len(events)), "horizon_days": horizon,
                    "peak_mag_observed": round(peak, 3), "redshift_used": z if z is not None else ctx.get("z_phot"),
                    "redshift_source": "skyportal" if z is not None else ("host photo-z" if ctx.get("z_phot") else None),
                    "classification": cls, "context": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ctx.items()},
                    "context_error": context.get("_error"), "triage": verdict},
        "annotations": skyportal_annotations(cls, verdict), "classifications": skyportal_classifications(cls),
        "annotations_flat": annotations_for(cls, verdict), "plot_files": plots,
    }
