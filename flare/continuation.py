"""Frozen exploratory continuation ranking, fitted on proper-training paths only.

Targets inspect future TRAINING paths; deployment scores receive only revealed
prefixes. Reward/cost regressions are ranking heuristics, not calibrated utility
probabilities. Argmax probability labels define all continuation features;
issued automatic decisions remain singleton set labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .trajectory_sets import _probabilities, _resources, _schema, _sets, _truth

VARIANTS = ("aggregate_output", "macro_output", "macro_quality")


def reliability_features(probs, nobs, labels, *, class_names, horizons, quality_temporal=False):
    """Return (objects, looks, features), ordered names; use only current/past looks."""
    probs = _probabilities(probs)
    class_names, horizons = _schema(class_names, horizons, probs.shape[2], probs.shape[1])
    nobs, _ = _resources(nobs, horizons, probs.shape[:2])
    labels = np.asarray(labels)
    classes = probs.shape[2]
    if (labels.shape != probs.shape[:2] or not np.issubdtype(labels.dtype, np.integer)
            or (labels < 0).any() or (labels >= classes).any()):
        raise ValueError("labels must contain valid integer class indices for each object and look")
    confidence = np.take_along_axis(probs, labels[..., None], axis=2)
    entropy = -(probs * np.log(np.maximum(probs, np.finfo(float).tiny))).sum(axis=2, keepdims=True) / np.log(classes)
    ordered = np.sort(probs, axis=2)
    margin = (ordered[..., -1] - ordered[..., -2])[..., None]
    onehot = np.eye(classes)[labels]
    blocks = [probs, confidence, entropy, margin, onehot]
    names = ([f"probability_{c}" for c in class_names]
             + ["predicted_label_probability", "normalized_entropy", "top_two_margin"]
             + [f"predicted_label_{c}" for c in class_names])
    if quality_temporal:
        change = np.zeros_like(nobs)
        switch = np.zeros_like(nobs)
        change[:, 1:] = np.abs(probs[:, 1:] - probs[:, :-1]).sum(axis=2)
        switch[:, 1:] = labels[:, 1:] != labels[:, :-1]
        log_horizon = np.broadcast_to(np.log1p(horizons)[None, :], nobs.shape)
        blocks += [np.log1p(nobs)[..., None], log_horizon[..., None], change[..., None], switch[..., None]]
        names += ["log1p_nobs", "log1p_horizon", "previous_look_probability_l1", "previous_look_label_switch"]
    return np.concatenate(blocks, axis=2), names


def training_targets(candidate_sets, y):
    """Targets for every active preterminal proper-training prefix.

    Reward is zero unless the first future closed state is a correct singleton.
    Macro reward is N/(C*n_y), with class frequencies from ALL training objects.
    Cost is the number of queries until first singleton/empty or final look.
    Every eligible object's total row weight equals one. Current singletons and
    empty sets have already closed and never create examples.
    """
    sets = _sets(candidate_sets)
    n, looks, classes = sets.shape
    y = _truth(y, n, classes)
    counts = np.bincount(y, minlength=classes)
    object_indices, look_indices, reward, macro, cost, weights = [], [], [], [], [], []
    for obj in range(n):
        eligible = []
        for look in range(looks - 1):
            if sets[obj, look].sum() <= 1:
                break
            eligible.append(look)
        for look in eligible:
            closure = looks - 1
            for later in range(look + 1, looks):
                if sets[obj, later].sum() <= 1:
                    closure = later
                    break
            correct = (sets[obj, closure].sum() == 1 and bool(sets[obj, closure, y[obj]]))
            object_indices.append(obj)
            look_indices.append(look)
            reward.append(float(correct))
            macro.append(float(correct) * n / (classes * counts[y[obj]]))
            cost.append(closure - look)
            weights.append(1 / len(eligible))
    return dict(object_index=np.array(object_indices, dtype=int),
                look_index=np.array(look_indices, dtype=int),
                reward_aggregate=np.array(reward, dtype=float),
                reward_macro=np.array(macro, dtype=float), cost=np.array(cost, dtype=float),
                sample_weight=np.array(weights, dtype=float))


def continuation_features(probs, nobs, *, class_names, horizons, quality_temporal=False):
    """Existing reliability features plus known current look index; prefix only."""
    probs = _probabilities(probs)
    class_names, horizons = _schema(class_names, horizons, probs.shape[2])
    if probs.shape[1] > len(horizons):
        raise ValueError("query prefix cannot exceed the frozen schedule")
    labels = probs.argmax(axis=2)
    features, names = reliability_features(
        probs, nobs, labels, class_names=class_names, quality_temporal=quality_temporal,
        horizons=horizons[:probs.shape[1]])
    if not quality_temporal:
        # Identical known-schedule basis in all variants: the quality ablation
        # differs only by nobs and previous-look probability/label changes.
        log_horizon = np.broadcast_to(np.log1p(horizons[:probs.shape[1]]), probs.shape[:2])[..., None]
        features = np.concatenate([features, log_horizon], axis=2)
        names = names + ["log1p_horizon"]
    index = np.broadcast_to(np.arange(probs.shape[1]), probs.shape[:2])[..., None]
    return np.concatenate([features, index], axis=2), names + ["current_look_index"]


def _regression(X, target, weights):
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1))])
    model.fit(X, target, scaler__sample_weight=weights, ridge__sample_weight=weights)
    return model


def fit_continuation_models(probs, nobs, y, candidate_sets, *, class_names, horizons):
    """Fit the three frozen reward/cost Ridge pairs without tuning.

    Pass only training objects here. The caller records classifier-fit membership
    and the separate source of candidate thresholds. No evaluation truth or
    saved hierarchical labels enter features, targets or scoring.
    """
    probs = _probabilities(probs)
    class_names, horizons = _schema(class_names, horizons, probs.shape[2], probs.shape[1])
    nobs, _ = _resources(nobs, horizons, probs.shape[:2])
    sets = _sets(candidate_sets)
    if sets.shape != probs.shape:
        raise ValueError("training candidate sets must match probability dimensions")
    y = _truth(y, len(probs), probs.shape[2])
    targets = training_targets(sets, y)
    rows, looks, weight = targets["object_index"], targets["look_index"], targets["sample_weight"]
    fitted = {}
    for name in VARIANTS:
        quality = name == "macro_quality"
        features, names = continuation_features(probs, nobs, class_names=class_names,
                                                horizons=horizons, quality_temporal=quality)
        X = features[rows, looks]
        reward = targets["reward_aggregate" if name == "aggregate_output" else "reward_macro"]
        no_positive = not np.any(reward > 0)
        reward_model = None if no_positive else _regression(X, reward, weight)
        cost_model = _regression(X, targets["cost"], weight) if len(rows) else None
        fitted[name] = dict(
            variant=name, quality_temporal=quality, feature_names=names,
            feature_label_source="probability_argmax",
            class_names=list(class_names), horizons=horizons.tolist(),
            training_role="caller-supplied training trajectories only; record classifier-fit membership separately",
            target_threshold_source="caller-supplied fixed candidate sets; record threshold calibration source separately",
            reward_model=reward_model, cost_model=cost_model,
            constant_reward=0.0 if no_positive else None,
            constant_cost=1.0 if not len(rows) else None,
            no_positive_reward=no_positive, no_eligible_prefixes=not len(rows),
            n_training_objects=len(probs), training_class_counts=np.bincount(y, minlength=probs.shape[2]).tolist(),
            n_eligible_objects=len(np.unique(rows)), n_training_rows=len(rows),
            n_positive_reward_rows=int(np.sum(reward > 0)), sum_sample_weights=float(weight.sum()),
            ridge_alpha=1.0, cost_clip=[1, max(1, probs.shape[1] - 1)], reward_clip_lower=0)
    return fitted


def continuation_scorer(record):
    """Pure deterministic ranking callback using only PrefixObservation fields."""
    cache = {}

    def score(observation):
        probs = np.asarray(observation.probabilities)[None]
        nobs = np.asarray(observation.nobs)[None]
        if observation.look != probs.shape[1] - 1:
            raise ValueError("observation must expose exactly the prefix through current look")
        # An exactly zero reward model produces deterministic zero ties without
        # fitting a fictitious positive utility or consulting future information.
        if record["constant_reward"] == 0:
            return 0.0
        key = (str(observation.obj_id), int(observation.look), probs.tobytes(), nobs.tobytes())
        if key in cache:
            return cache[key]
        features, names = continuation_features(
            probs, nobs, class_names=record["class_names"], horizons=record["horizons"],
            quality_temporal=record["quality_temporal"])
        if names != record["feature_names"]:
            raise ValueError("continuation feature schema mismatch")
        X = features[:, -1]
        reward = float(record["reward_model"].predict(X)[0])
        cost = (float(record["cost_model"].predict(X)[0]) if record["cost_model"] is not None
                else record["constant_cost"])
        value = float(max(reward, 0) / np.clip(cost, *record["cost_clip"]))
        # Memoize only prefixes already exposed by a callback. This never
        # constructs scores for unrevealed future looks and does not affect ties.
        cache[key] = value
        return value
    return score


def model_record(record):
    """JSON-readable metadata, scaler parameters and Ridge coefficients."""
    out = {k: v for k, v in record.items() if k not in {"reward_model", "cost_model"}}
    for name in ("reward", "cost"):
        model = record[f"{name}_model"]
        if model is None:
            out[f"{name}_coefficients"] = None
        else:
            scaler, ridge = model.named_steps["scaler"], model.named_steps["ridge"]
            out[f"{name}_coefficients"] = dict(
                scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                coefficient=ridge.coef_.tolist(), intercept=float(ridge.intercept_))
    return out
