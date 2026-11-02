"""Fixed-schedule trajectory class sets and abstaining decisions.

The class score is always 1-p(c). Calibration/query trajectories must use the
same frozen classifier at each prespecified look. Pooled calibration addresses
mixture coverage; Mondrian calibration addresses true-class conditional coverage
under within-class exchangeability. Neither implies low error conditional on
selection, emitted-label precision, or service false-alarm rates.

Envelope calibration applies the ordinary split-conformal finite-rank rule to
one maximum score per complete object trajectory (Vovk et al., 2005,
Algorithmic Learning in a Random World). With frozen scorers and schedule,
exchangeable complete trajectories give marginal whole-path inclusion; the
Mondrian version requires exchangeability within each true class. Arbitrary
within-trajectory dependence is allowed. The complete-path set is contained in
every prefix set, so wrong singleton decisions imply true-class exclusion.
These statements are marginal over calibration and query trajectories, not
conditional on the realized calibration sample. Extending the frozen schedule
requires a new calibration. Terminal/fixedlook methods calibrate one
prespecified look only; choosing a deadline after inspecting paths is uncovered.
"""
from __future__ import annotations

from math import ceil
from numbers import Integral

import numpy as np

def _class_names(class_names, classes):
    if isinstance(class_names, (str, bytes, dict, set, frozenset)):
        raise ValueError("class_names must be an ordered sequence of class names")
    try:
        names = tuple(class_names)
    except TypeError as exc:
        raise ValueError("class_names must be an ordered sequence of class names") from exc
    if (len(names) != classes or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != classes):
        raise ValueError("class_names must uniquely name all probability columns with nonempty strings")
    return names


def _schema(class_names, horizons, classes, looks=None):
    names = _class_names(class_names, classes)
    horizons = np.asarray(horizons, dtype=float)
    if (horizons.ndim != 1 or not len(horizons)
            or (looks is not None and horizons.shape != (looks,)) or not np.isfinite(horizons).all()
            or (horizons <= 0).any() or (np.diff(horizons) <= 0).any()):
        raise ValueError("horizons must be positive, increasing and match the frozen looks")
    return names, horizons


def _probabilities(probs):
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 3 or probs.shape[1] == 0 or probs.shape[2] < 2:
        raise ValueError("probabilities need shape (objects, looks>0, classes>=2)")
    if (not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
            or not np.allclose(probs.sum(axis=2), 1, rtol=0, atol=1e-7)):
        raise ValueError("probabilities must be finite, in [0,1], and sum to one")
    return probs


def _truth(y, n, classes):
    y = np.asarray(y)
    if n == 0 and y.shape == (0,):
        return np.empty(0, dtype=np.intp)
    if (y.shape != (n,) or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any() or (y >= classes).any()):
        raise ValueError("y must contain one integer true class index per object")
    return y.astype(np.intp, copy=False)


def _alpha(alpha):
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError("alpha must lie strictly between zero and one")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return alpha


def finite_sample_quantile(scores, alpha):
    """kth order statistic, k=ceil((n+1)*(1-alpha)); +inf if k>n.

    Ties are included in the prediction set. In particular, finite resolution
    is never bypassed by clipping k to n, or by interpolating quantiles.
    """
    scores = np.asarray(scores, dtype=float)
    alpha = _alpha(alpha)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional array")
    k = ceil((len(scores) + 1) * (1 - alpha))
    return float(np.partition(scores, k - 1)[k - 1]) if k <= len(scores) else float("inf")


def calibrate_sets(calibration_probs, calibration_y, alpha, *, class_names, horizons,
                   method="envelope", mondrian=True, look_index=None):
    """Fit fixed thresholds using calibration truth only.

    Threshold shape is (looks, classes). Envelope uses max_t(1-p_t(y));
    Bonferroni calibrates each look at alpha/number_of_fixed_looks; terminal
    calibrates only the last look and leaves earlier candidate sets unrestricted.
    Fixedlook calibrates full alpha at one prespecified deadline and leaves all
    other look thresholds unrestricted. It has a fixed-look guarantee only, not
    an adaptive full-alpha any-time guarantee. Empty classes have +inf thresholds
    and cannot be excluded by Mondrian sets.
    """
    probs = _probabilities(calibration_probs)
    y = _truth(calibration_y, len(probs), probs.shape[2])
    alpha = _alpha(alpha)
    class_names, horizons = _schema(class_names, horizons, probs.shape[2], probs.shape[1])
    if method not in {"envelope", "bonferroni", "terminal", "fixedlook"}:
        raise ValueError("unknown calibration method")
    if not isinstance(mondrian, (bool, np.bool_)):
        raise ValueError("mondrian must be boolean")
    looks, classes = probs.shape[1:]
    if method == "fixedlook":
        if (isinstance(look_index, (bool, np.bool_)) or not isinstance(look_index, Integral)
                or not 0 <= look_index < looks):
            raise ValueError("fixedlook needs a prespecified valid look_index")
        deadline_index = int(look_index)
    else:
        if look_index is not None:
            raise ValueError("look_index applies only to fixedlook")
        deadline_index = looks - 1 if method == "terminal" else None
    truth_scores = 1 - probs[np.arange(len(probs))[:, None], np.arange(looks)[None, :], y[:, None]]
    nclass = np.bincount(y, minlength=classes)
    counts = nclass.copy() if mondrian else np.full(classes, len(y), dtype=int)
    effective_alpha = alpha / looks if method == "bonferroni" else alpha
    thresholds = np.full((looks, classes), np.inf)
    for cls in range(classes):
        scores = truth_scores[y == cls] if mondrian else truth_scores
        if method == "envelope":
            thresholds[:, cls] = finite_sample_quantile(scores.max(axis=1), effective_alpha)
        elif method in {"terminal", "fixedlook"}:
            thresholds[deadline_index, cls] = finite_sample_quantile(scores[:, deadline_index], effective_alpha)
        else:
            for look in range(looks):
                thresholds[look, cls] = finite_sample_quantile(scores[:, look], effective_alpha)
    return dict(method=method, mondrian=bool(mondrian), alpha=alpha,
                class_names=class_names, horizons=horizons.tolist(),
                calibration_look_index=deadline_index,
                effective_alpha=effective_alpha, thresholds=thresholds,
                n_calibration_by_true_class=nclass, n_calibration_for_threshold=counts,
                pvalue_resolution=1 / (counts + 1),
                order_index=np.array([ceil((int(n) + 1) * (1 - effective_alpha)) for n in counts]))


def nested_candidate_sets(query_probs, calibration):
    """Intersect candidates through each look; no evaluation truth is accepted.

    A nonempty prefix may be shorter than the frozen calibration schedule.
    Thresholds are sliced without refitting or reallocating Bonferroni alpha.
    Membership compares 1-p directly to the stored score threshold, without
    subtracting the threshold back from one; exact score ties are retained.
    The prefix at time t depends only on query probabilities through t. For
    envelope thresholds this equals max_{s<=t}(1-p_s(c)) <= q(c).
    """
    probs = _probabilities(query_probs)
    thresholds = np.asarray(calibration["thresholds"], dtype=float)
    if (thresholds.ndim != 2 or thresholds.shape[1] != probs.shape[2]
            or probs.shape[1] > thresholds.shape[0]
            or np.isnan(thresholds).any() or (thresholds < 0).any()):
        raise ValueError("thresholds must cover the query prefix and classes, be nonnegative without NaNs")
    _schema(calibration["class_names"], calibration["horizons"], probs.shape[2], thresholds.shape[0])
    return np.logical_and.accumulate(1 - probs <= thresholds[None, :probs.shape[1], :], axis=1)


def calibration_to_dict(calibration):
    """Serialize a calibration to strict JSON data; +infinity has an explicit mask.

    A masked null is an unbounded threshold, never an unknown or missing value.
    Pass the result to json.dumps(..., allow_nan=False).
    """
    thresholds = np.asarray(calibration["thresholds"], dtype=float)
    if (thresholds.ndim != 2 or np.isnan(thresholds).any() or (thresholds < 0).any()):
        raise ValueError("thresholds must be a nonnegative matrix without NaNs")
    _schema(calibration["class_names"], calibration["horizons"], thresholds.shape[1], thresholds.shape[0])
    record = {key: value.tolist() if isinstance(value, np.ndarray) else
              list(value) if isinstance(value, tuple) else value
              for key, value in calibration.items() if key != "thresholds"}
    record["thresholds"] = [[float(x) if np.isfinite(x) else None for x in row] for row in thresholds]
    record["threshold_positive_infinity"] = np.isposinf(thresholds).tolist()
    return record


def calibration_from_dict(record):
    """Load strict JSON calibration data, rejecting unmarked unknown thresholds."""
    thresholds = np.array(record["thresholds"], dtype=float, copy=True)
    mask = np.asarray(record["threshold_positive_infinity"])
    if (mask.dtype != np.bool_ or mask.shape != thresholds.shape or thresholds.ndim != 2
            or np.any(np.isnan(thresholds) != mask) or np.isinf(thresholds).any()
            or np.any(thresholds[~mask] < 0)):
        raise ValueError("only explicitly masked null thresholds may represent positive infinity")
    thresholds[mask] = np.inf
    names, horizons = _schema(record["class_names"], record["horizons"],
                              thresholds.shape[1], thresholds.shape[0])
    calibration = {key: value for key, value in record.items()
                   if key not in {"thresholds", "threshold_positive_infinity"}}
    calibration.update(thresholds=thresholds, class_names=names, horizons=horizons.tolist())
    for key in ("n_calibration_by_true_class", "n_calibration_for_threshold", "pvalue_resolution", "order_index"):
        calibration[key] = np.asarray(calibration[key])
    return calibration


def _sets(candidate_sets):
    sets = np.asarray(candidate_sets)
    if sets.dtype != np.bool_ or sets.ndim != 3 or sets.shape[1] == 0 or sets.shape[2] < 2:
        raise ValueError("candidate_sets must be a boolean (objects,looks,classes) array")
    if np.any(sets[:, 1:] & ~sets[:, :-1]):
        raise ValueError("candidate sets must be nested")
    return sets


def _resources(nobs, horizons, shape):
    nobs, horizons = np.asarray(nobs, float), np.asarray(horizons, float)
    if (nobs.shape != shape or not np.isfinite(nobs).all() or (nobs < 0).any()
            or (np.diff(nobs, axis=1) < 0).any()):
        raise ValueError("nobs must be finite, nonnegative cumulative counts, nondecreasing across looks")
    if (horizons.shape != (shape[1],) or not np.isfinite(horizons).all()
            or (horizons <= 0).any() or (np.diff(horizons) <= 0).any()):
        raise ValueError("horizons must be positive, increasing and match looks")
    return nobs, horizons


def first_singleton(candidate_sets, nobs, *, horizons, budget_day=None):
    """Retain the first singleton by a fixed horizon; unresolved objects abstain.

    stop_index is the consumed look for EVERY object, including abstainers;
    decision_index is -1 for abstainers. Empty sets never force a label. All
    unresolved objects consume the budget, even if their sets are already empty.
    Future candidate sets cannot revise an earlier automatic decision.
    """
    sets = _sets(candidate_sets)
    nobs, horizons = _resources(nobs, horizons, sets.shape[:2])
    budget_day = horizons[-1] if budget_day is None else budget_day
    if not np.isscalar(budget_day) or not np.isfinite(budget_day):
        raise ValueError("budget_day must be an existing scheduled horizon")
    matching = np.flatnonzero(horizons == budget_day)
    if len(matching) != 1:
        raise ValueError("budget_day must be an existing scheduled horizon")
    cap = int(matching[0])
    count = len(sets)
    selected = np.zeros(count, dtype=bool)
    prediction = np.full(count, -1, dtype=int)
    decision_index = np.full(count, -1, dtype=int)
    stop_index = np.full(count, cap, dtype=int)
    # Only currently active objects' sets are read to make each decision.
    for look in range(cap + 1):
        active = np.flatnonzero(~selected)
        singleton = sets[active, look].sum(axis=1) == 1
        chosen = active[singleton]
        selected[chosen] = True
        prediction[chosen] = sets[chosen, look].argmax(axis=1)
        decision_index[chosen] = look
        stop_index[chosen] = look
    rows = np.arange(count)
    return dict(selected=selected, prediction=prediction, decision_index=decision_index,
                stop_index=stop_index, consumed_day=horizons[stop_index],
                decision_day=np.where(selected, horizons[stop_index], np.nan),
                nobs_consumed=nobs[rows, stop_index],
                set_size_at_stop=sets[rows, stop_index].sum(axis=1),
                budget_index=np.full(count, cap, dtype=int))


def summarize_sets(candidate_sets, decision, y, *, class_names):
    """Descriptive true-class metrics; all comparisons use dataset truth y.

    Lifetime exclusion uses the full fixed set path, including diagnostic looks
    after an operational decision. Consumed-prefix exclusion is also reported.
    Conditional selected error and wrong-decision probability over ALL objects
    have distinct denominators. No confidence/certification claim is computed.
    """
    sets = _sets(candidate_sets)
    n, looks, classes = sets.shape
    y = _truth(y, n, classes)
    class_names = _class_names(class_names, classes)
    selected = np.asarray(decision["selected"])
    prediction = np.asarray(decision["prediction"])
    stop_index = np.asarray(decision["stop_index"])
    if (selected.shape != (n,) or selected.dtype != np.bool_ or prediction.shape != (n,)
            or not np.issubdtype(prediction.dtype, np.integer)
            or (prediction[~selected] != -1).any()
            or (prediction[selected] < 0).any() or (prediction[selected] >= classes).any()
            or stop_index.shape != (n,) or not np.issubdtype(stop_index.dtype, np.integer)
            or (stop_index < 0).any() or (stop_index >= looks).any()):
        raise ValueError("invalid per-object decisions")
    consumed_day = np.asarray(decision["consumed_day"], float)
    nobs = np.asarray(decision["nobs_consumed"], float)
    if (consumed_day.shape != (n,) or nobs.shape != (n,)
            or not np.isfinite(consumed_day).all() or not np.isfinite(nobs).all()
            or (consumed_day <= 0).any() or (nobs < 0).any()):
        raise ValueError("invalid consumed resources")
    true_in_set = sets[np.arange(n)[:, None], np.arange(looks)[None, :], y[:, None]]
    lifetime_excluded = (~true_in_set).any(axis=1)
    consumed_excluded = ((~true_in_set) & (np.arange(looks)[None, :] <= stop_index[:, None])).any(axis=1)
    wrong = selected & (prediction != y)
    retained_unresolved = ~selected & true_in_set[np.arange(n), stop_index]
    set_size = sets[np.arange(n), stop_index].sum(axis=1)

    def group(mask):
        total = int(mask.sum())
        chosen = mask & selected
        ns, nw = int(chosen.sum()), int((wrong & mask).sum())
        nu = total - ns
        nr = int((retained_unresolved & mask).sum())
        unresolved = mask & ~selected
        excluded, consumed = int((lifetime_excluded & mask).sum()), int((consumed_excluded & mask).sum())
        return dict(n_objects=total, n_selected=ns, n_unresolved=nu, n_wrong_automatic=nw,
                    n_correct_automatic=ns - nw, correct_automatic_per_all=(ns - nw) / total if total else None,
                    n_true_class_retained_unresolved=nr,
                    true_class_retained_unresolved_per_all=nr / total if total else None,
                    true_class_retained_given_unresolved=nr / nu if nu else None,
                    mean_set_size_at_consumed=float(set_size[mask].mean()) if total else None,
                    mean_set_size_unresolved=float(set_size[unresolved].mean()) if nu else None,
                    n_true_class_lifetime_excluded=excluded, n_true_class_consumed_prefix_excluded=consumed,
                    true_class_lifetime_exclusion=excluded / total if total else None,
                    true_class_consumed_prefix_exclusion=consumed / total if total else None,
                    wrong_automatic_per_all=nw / total if total else None,
                    selected_error=nw / ns if ns else None,
                    selection_coverage=ns / total if total else None,
                    mean_day_selected=float(consumed_day[chosen].mean()) if ns else None,
                    mean_day_consumed_all=float(consumed_day[mask].mean()) if total else None,
                    total_nobs_consumed=float(nobs[mask].sum()),
                    mean_nobs_consumed_all=float(nobs[mask].mean()) if total else None)

    return dict(overall=group(np.ones(n, dtype=bool)),
                by_true_class={name: group(y == cls) for cls, name in enumerate(class_names)})
