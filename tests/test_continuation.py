"""Continuation training targets, fixed feature schema and causal scoring."""
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flare.checkpoint_budget import PrefixObservation, allocate_query_budget
from flare.continuation import (
    VARIANTS, continuation_features, continuation_scorer, fit_continuation_models,
    model_record, reliability_features, training_targets,
)


def schema(classes=6, looks=4):
    return dict(class_names=tuple(f"class_{i}" for i in range(classes)),
                horizons=(5, 10, 20, 100)[:looks])


def paths():
    sets = np.ones((4, 4, 6), dtype=bool)
    sets[0, 2:] = False
    sets[0, 2:, 0] = True
    sets[0, 3] = False
    sets[1, 1:] = False
    sets[1, 1:, 5] = True
    sets[2] = False
    return sets, np.array([0, 1, 2, 0])


def arrays(classes=6, looks=4):
    rng = np.random.default_rng(42)
    probs = rng.dirichlet(np.ones(classes), size=(4, looks))
    nobs = np.broadcast_to([1, 3, 7, 11][:looks], (4, looks)).copy()
    return probs, nobs


def test_targets_first_closure_object_weights_and_class_reward_denominator():
    sets, y = paths()
    target = training_targets(sets, y.astype(np.uint64))
    np.testing.assert_array_equal(target["object_index"], [0, 0, 1, 3, 3, 3])
    np.testing.assert_array_equal(target["look_index"], [0, 1, 0, 0, 1, 2])
    np.testing.assert_array_equal(target["reward_aggregate"], [1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(target["reward_macro"], [1 / 3, 1 / 3, 0, 0, 0, 0])
    np.testing.assert_array_equal(target["cost"], [2, 1, 1, 3, 2, 1])
    for obj in np.unique(target["object_index"]):
        assert target["sample_weight"][target["object_index"] == obj].sum() == pytest.approx(1)


@pytest.mark.parametrize(("classes", "looks"), [(5, 2), (6, 4)])
@pytest.mark.parametrize("quality", [False, True])
def test_feature_prefixes_are_causal_for_generic_shape(classes, looks, quality):
    probs, nobs = arrays(classes, looks)
    full, names = continuation_features(probs, nobs, quality_temporal=quality, **schema(classes, looks))
    for length in range(1, looks + 1):
        prefix, prefix_names = continuation_features(probs[:, :length], nobs[:, :length],
                                                     quality_temporal=quality, **schema(classes, looks))
        assert prefix_names == names
        np.testing.assert_array_equal(prefix, full[:, :length])
    assert names[-1] == "current_look_index"
    np.testing.assert_array_equal(full[..., -1], np.broadcast_to(np.arange(looks), (4, looks)))


def test_four_by_six_feature_order_and_common_known_time_basis():
    probs, nobs = arrays()
    output, names = continuation_features(probs, nobs, **schema())
    quality, quality_names = continuation_features(probs, nobs, quality_temporal=True, **schema())
    expected = ([f"probability_class_{i}" for i in range(6)] +
                ["predicted_label_probability", "normalized_entropy", "top_two_margin"] +
                [f"predicted_label_class_{i}" for i in range(6)])
    assert names == expected + ["log1p_horizon", "current_look_index"]
    assert quality_names == expected + ["log1p_nobs", "log1p_horizon",
                                       "previous_look_probability_l1", "previous_look_label_switch",
                                       "current_look_index"]
    for name in names:
        np.testing.assert_array_equal(output[..., names.index(name)], quality[..., quality_names.index(name)])
    np.testing.assert_array_equal(output[..., 6], probs.max(axis=2))
    np.testing.assert_array_equal(output[..., 9:15].argmax(axis=2), probs.argmax(axis=2))
    np.testing.assert_array_equal(output[..., -2], np.broadcast_to(np.log1p([5, 10, 20, 100]), (4, 4)))
    # The base reliability API can describe an explicit label. Continuation
    # always chooses probability argmax, independently of any issued label.
    explicit, _ = reliability_features(probs, nobs, np.zeros((4, 4), dtype=int), **schema())
    np.testing.assert_array_equal(explicit[..., 6], probs[..., 0])


@pytest.mark.parametrize(("classes", "looks"), [(5, 2), (6, 4)])
def test_all_variants_fit_weighted_scaler_and_ridge_without_tuning(classes, looks):
    probs, nobs = arrays(classes, looks)
    if looks == 4:
        sets, y = paths()
    else:
        sets = np.ones((4, looks, classes), dtype=bool)
        sets[0, 1] = False
        sets[0, 1, 0] = True
        y = np.array([0, 1, 2, 0])
    fitted = fit_continuation_models(probs, nobs, y, sets, **schema(classes, looks))
    target = training_targets(sets, y)
    for variant in VARIANTS:
        record = fitted[variant]
        features, names = continuation_features(probs, nobs, quality_temporal=variant == "macro_quality",
                                                **schema(classes, looks))
        X = features[target["object_index"], target["look_index"]]
        weights = target["sample_weight"]
        assert record["ridge_alpha"] == 1
        assert record["cost_clip"] == [1, looks - 1]
        assert record["feature_names"] == names
        for kind in ("reward", "cost"):
            y_target = target["cost"] if kind == "cost" else target[
                "reward_aggregate" if variant == "aggregate_output" else "reward_macro"]
            reference = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1))])
            reference.fit(X, y_target, scaler__sample_weight=weights, ridge__sample_weight=weights)
            np.testing.assert_array_equal(record[f"{kind}_model"].predict(X), reference.predict(X))
            np.testing.assert_allclose(record[f"{kind}_model"].named_steps["scaler"].mean_,
                                       np.average(X, axis=0, weights=weights))
        json.dumps(model_record(record), allow_nan=False)


@pytest.mark.parametrize("looks", [1, 2, 4])
def test_no_reward_or_eligible_prefixes_is_explicit_zero_fallback(looks):
    probs, nobs = arrays(5, looks)
    for closed in (False, True):
        sets = np.full(probs.shape, not closed, dtype=bool)
        fitted = fit_continuation_models(probs, nobs, [0, 1, 2, 3], sets, **schema(5, looks))
        for record in fitted.values():
            assert record["reward_model"] is None and record["constant_reward"] == 0
            assert record["no_eligible_prefixes"] == (closed or looks == 1)
            assert record["cost_clip"] == [1, max(1, looks - 1)]
            observation = PrefixObservation(0, "toy", 0, sets[0, :1], probs[0, :1], nobs[0, :1])
            assert continuation_scorer(record)(observation) == 0


def test_empty_training_is_supported_without_fictitious_examples():
    fitted = fit_continuation_models(np.empty((0, 2, 5)), np.empty((0, 2)), [],
                                    np.empty((0, 2, 5), dtype=bool), **schema(5, 2))
    for record in fitted.values():
        assert record["n_training_rows"] == record["n_training_objects"] == 0
        assert record["cost_model"] is None and record["constant_cost"] == 1


class ConstantPrediction:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


@pytest.mark.parametrize(("reward", "cost", "expected"), [(-2, 2, 0), (2, .2, 2), (2, 10, 2 / 3)])
def test_scorer_clips_only_reward_lower_bound_and_cost(reward, cost, expected):
    probs, nobs = arrays()
    sets, y = paths()
    record = fit_continuation_models(probs, nobs, y, sets, **schema())["aggregate_output"]
    record.update(constant_reward=None, reward_model=ConstantPrediction(reward), cost_model=ConstantPrediction(cost))
    observation = PrefixObservation(0, "toy", 1, sets[0, :2], probs[0, :2], nobs[0, :2])
    scorer = continuation_scorer(record)
    assert scorer(observation) == pytest.approx(expected)
    assert scorer(observation) == pytest.approx(expected)


def test_trained_callback_uses_only_revealed_queries_and_common_endpoints():
    probs, nobs = arrays()
    sets, y = paths()
    fitted = fit_continuation_models(probs, nobs, y, sets, **schema())
    ids = ["a", "b", "c", "d"]
    endpoints = {}
    for name, record in fitted.items():
        scorer = continuation_scorer(record)
        before = allocate_query_budget(sets, probs, nobs, ids, 1, policy="custom", scorer=scorer)
        changed_sets = sets.copy()
        changed_sets[:, 1:] = False
        changed_probs = probs.copy()
        changed_probs[:, 1:] = np.eye(6)[5]
        changed_counts = nobs.copy()
        changed_counts[:, 1:] = 1000
        after = allocate_query_budget(changed_sets, changed_probs, changed_counts, ids, 1,
                                      policy="custom", scorer=continuation_scorer(record))
        assert before.actions[0].index == after.actions[0].index
        endpoints[name] = [allocate_query_budget(sets, probs, nobs, ids, budget,
                           policy="custom", scorer=continuation_scorer(record)) for budget in (0, 12)]
    for i in (0, 1):
        reference = endpoints[VARIANTS[0]][i]
        for name in VARIANTS[1:]:
            np.testing.assert_array_equal(endpoints[name][i].final_look, reference.final_look)
            np.testing.assert_array_equal(endpoints[name][i].predictions, reference.predictions)
            np.testing.assert_array_equal(endpoints[name][i].status, reference.status)
