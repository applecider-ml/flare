"""Finite-rank and prefix contracts for fixed-schedule trajectory sets."""
import json

import numpy as np
import pytest

from flare.trajectory_sets import (
    calibrate_sets, calibration_from_dict, calibration_to_dict,
    finite_sample_quantile, first_singleton, nested_candidate_sets, summarize_sets,
)


def schema(classes, looks):
    return dict(class_names=tuple(f"class_{i}" for i in range(classes)),
                horizons=tuple(2 ** i for i in range(looks)))


def test_finite_rank_ties_and_missing_classes_survive_strict_json():
    assert finite_sample_quantile([], .2) == np.inf
    assert finite_sample_quantile([.1, .3, .3, .9], .4) == .3
    assert finite_sample_quantile([.1, .3, .3, .9], .19) == np.inf
    assert finite_sample_quantile(np.arange(19), .05) == 18
    probs = np.tile([[[.9, .1]]], (3, 2, 1))
    calibration = calibrate_sets(probs, np.zeros(3, dtype=np.uint64), .4, **schema(2, 2))
    assert np.isposinf(calibration["thresholds"][:, 1]).all()
    np.testing.assert_array_equal(calibration["n_calibration_for_threshold"], [3, 0])
    record = calibration_to_dict(calibration)
    loaded = calibration_from_dict(json.loads(json.dumps(record, allow_nan=False)))
    assert loaded["horizons"] == [1., 2.]
    assert record["thresholds"][0][1] is None
    assert record["threshold_positive_infinity"][0][1]
    np.testing.assert_array_equal(loaded["thresholds"], calibration["thresholds"])
    np.testing.assert_array_equal(nested_candidate_sets(probs, loaded),
                                  nested_candidate_sets(probs, calibration))
    record["threshold_positive_infinity"][0][1] = False
    with pytest.raises(ValueError, match="masked null"):
        calibration_from_dict(record)


@pytest.mark.parametrize("method", ["envelope", "bonferroni", "terminal", "fixedlook"])
@pytest.mark.parametrize("mondrian", [False, True])
@pytest.mark.parametrize(("classes", "looks"), [(5, 2), (6, 4)])
def test_generic_prefixes_match_frozen_full_schedule(method, mondrian, classes, looks):
    rng = np.random.default_rng(13)
    probs = rng.dirichlet(np.ones(classes), size=(120, looks))
    y = np.arange(120) % classes
    extra = {"look_index": 0} if method == "fixedlook" else {}
    calibration = calibrate_sets(probs, y, .2, method=method, mondrian=mondrian,
                                 **schema(classes, looks), **extra)
    full = nested_candidate_sets(probs, calibration)
    assert not np.any(full[:, 1:] & ~full[:, :-1])
    alpha = .2 / looks if method == "bonferroni" else .2
    assert calibration["effective_alpha"] == alpha
    original = calibration["thresholds"].copy()
    for length in range(1, looks + 1):
        np.testing.assert_array_equal(nested_candidate_sets(probs[:, :length], calibration), full[:, :length])
    np.testing.assert_array_equal(calibration["thresholds"], original)
    changed = probs.copy()
    changed[:, 1:] = np.eye(classes)[0]
    np.testing.assert_array_equal(nested_candidate_sets(changed, calibration)[:, :1], full[:, :1])
    with pytest.raises(ValueError, match="prefix"):
        nested_candidate_sets(np.concatenate([probs, probs[:, :1]], axis=1), calibration)


@pytest.mark.parametrize("method", ["envelope", "bonferroni", "terminal", "fixedlook"])
@pytest.mark.parametrize("mondrian", [False, True])
def test_exhaustive_held_out_assignments_obey_marginal_rank_budget(method, mondrian):
    # Enumerate the query role in a fixed exchangeable finite population. This
    # checks marginal ranks, not a guarantee conditional on a fitted threshold.
    rng = np.random.default_rng(71)
    probs = rng.dirichlet([.5, .5], size=(32, 4))
    y = np.repeat([0, 1], 16)
    for alpha in (.25, .5):
        excluded = [[], []]
        for query in range(len(probs)):
            keep = np.arange(len(probs)) != query
            calibration = calibrate_sets(
                probs[keep], y[keep], alpha, method=method, mondrian=mondrian,
                **schema(2, 4), **({"look_index": 1} if method == "fixedlook" else {}),
            )
            path = nested_candidate_sets(probs[[query]], calibration)
            excluded[y[query]].append(not path[0, :, y[query]].all())
        if mondrian:
            assert all(np.mean(values) <= alpha for values in excluded)
        else:
            assert np.mean(excluded) <= alpha


def test_one_look_matches_existing_bounded_lac_memberships():
    from flare.conformal import calibrate, prediction_sets

    rng = np.random.default_rng(15)
    # Existing conformal API has its own taxonomy; this comparison only uses
    # classes present there and tests membership rather than inf serialization.
    from flare.taxonomy import NUM_CLASSES
    probs = rng.dirichlet(np.ones(NUM_CLASSES), size=100)
    query = rng.dirichlet(np.ones(NUM_CLASSES), size=30)
    y = np.arange(100) % NUM_CLASSES
    for alpha in (.01, .2):
        original = calibrate(probs, y, alpha)
        calibration = calibrate_sets(probs[:, None], y, alpha, **schema(NUM_CLASSES, 1))
        finite = np.isfinite(calibration["thresholds"][0])
        np.testing.assert_array_equal(calibration["thresholds"][0, finite], original[finite])
        # Independent queries avoid the different floating point roundoff of
        # score<=q versus p>=1-q at a calibration score's exact boundary.
        np.testing.assert_array_equal(nested_candidate_sets(query[:, None], calibration)[:, 0],
                                      prediction_sets(query, original))
    # Exactly representable scores also verify inclusive boundary membership.
    probs = np.tile([.5, .25, .125, .0625, .0625], (100, 1))
    calibration = calibrate_sets(probs[:, None], y, .2, **schema(NUM_CLASSES, 1))
    np.testing.assert_array_equal(nested_candidate_sets(probs[:, None], calibration)[:, 0],
                                  prediction_sets(probs, calibrate(probs, y, .2)))


def test_exact_calibration_score_ties_are_retained_before_complement_roundoff():
    rng = np.random.default_rng(15)
    probs = rng.dirichlet(np.ones(5), size=100)
    y = np.arange(100) % 5
    calibration = calibrate_sets(probs[:, None], y, .2, **schema(5, 1))
    path = nested_candidate_sets(probs[:, None], calibration)
    for cls in range(5):
        tied = (y == cls) & (1 - probs[:, cls] == calibration["thresholds"][0, cls])
        assert tied.any()
        assert path[tied, 0, cls].all()


def test_first_singleton_preserves_early_decision_and_distinct_error_denominators():
    sets = np.array([[[1, 0], [0, 0], [0, 0], [0, 0]],
                     [[0, 0], [0, 0], [0, 0], [0, 0]],
                     [[1, 1], [1, 1], [0, 1], [0, 1]]], dtype=bool)
    nobs = np.broadcast_to([2, 5, 9, 25], (3, 4))
    horizons = [5, 10, 20, 100]
    decision = first_singleton(sets, nobs, horizons=horizons)
    np.testing.assert_array_equal(decision["prediction"], [0, -1, 1])
    np.testing.assert_array_equal(decision["stop_index"], [0, 3, 2])
    np.testing.assert_array_equal(decision["nobs_consumed"], [2, 25, 9])
    early = first_singleton(sets, nobs, horizons=horizons, budget_day=10)
    np.testing.assert_array_equal(early["stop_index"], [0, 1, 1])
    np.testing.assert_array_equal(early["prediction"], [0, -1, -1])
    report = summarize_sets(sets, decision, [1, 0, 1], class_names=("a", "b"))
    assert report["overall"]["wrong_automatic_per_all"] == pytest.approx(1 / 3)
    assert report["overall"]["selected_error"] == .5
    assert report["overall"]["total_nobs_consumed"] == 36
    assert report["by_true_class"]["a"]["selected_error"] is None
    report = summarize_sets(sets, decision, [0, 0, 1], class_names=("a", "b"))
    assert report["overall"]["n_wrong_automatic"] == 0
    assert report["overall"]["n_true_class_lifetime_excluded"] == 2
    assert report["overall"]["n_true_class_consumed_prefix_excluded"] == 1


@pytest.mark.parametrize("method", ["terminal", "fixedlook"])
def test_single_look_controls_leave_other_looks_unrestricted(method):
    probs = np.tile([[[.9, .1]]], (20, 4, 1))
    deadline = 3 if method == "terminal" else 1
    calibration = calibrate_sets(probs, np.zeros(20, dtype=int), .2, method=method,
                                 mondrian=False, **schema(2, 4),
                                 **({"look_index": deadline} if method == "fixedlook" else {}))
    path = nested_candidate_sets(probs, calibration)
    assert path[:, :deadline].all()
    assert np.isposinf(np.delete(calibration["thresholds"], deadline, axis=0)).all()
    decision = first_singleton(path, np.ones((20, 4)), horizons=schema(2, 4)["horizons"])
    assert (decision["stop_index"] == deadline).all()
    assert decision["selected"].all()


@pytest.mark.parametrize("names", ["ab", {"a", "b"}, {"a": 1, "b": 2}, ["a", "a"], ["a", ""], 2])
def test_class_order_is_explicit_and_validated(names):
    with pytest.raises(ValueError, match="class_names"):
        calibrate_sets(np.full((3, 2, 2), .5), [0, 1, 0], .2, class_names=names, horizons=[1, 2])


def test_empty_calibration_queries_and_invalid_thresholds_are_explicit():
    calibration = calibrate_sets(np.empty((0, 2, 5)), [], .1, **schema(5, 2))
    assert np.isposinf(calibration["thresholds"]).all()
    path = nested_candidate_sets(np.empty((0, 2, 5)), calibration)
    decision = first_singleton(path, np.empty((0, 2)), horizons=[1, 2])
    report = summarize_sets(path, decision, np.array([], dtype=int), class_names=schema(5, 2)["class_names"])
    assert report["overall"]["selected_error"] is None
    for bad in (0.5, np.ones((2, 4)), np.full((2, 5), np.nan), np.full((2, 5), -1)):
        with pytest.raises(ValueError, match="thresholds"):
            nested_candidate_sets(np.full((3, 1, 5), .2), dict(calibration, thresholds=bad))


@pytest.mark.parametrize("alpha", [0, 1, np.nan, True, [.1]])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        finite_sample_quantile([.1, .2], alpha)
