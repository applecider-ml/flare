"""Scheduler contract checks using synthetic trajectories, not performance claims."""
import numpy as np
import pytest

from flare.checkpoint_budget import PrefixObservation, allocate_query_budget


POLICIES = ("round_robin", "random", "confidence", "entropy", "candidate_count", "rare_mass", "custom")


def trajectories(n=3, looks=4, classes=6):
    sets = np.ones((n, looks, classes), dtype=bool)
    probs = np.full(sets.shape, 1 / classes)
    counts = np.broadcast_to(np.arange(looks), (n, looks)).copy()
    return sets, probs, counts, [f"id-{i}" for i in range(n)]


def options(policy):
    return dict(policy=policy, scorer=lambda obs: float(obs.candidate_sets[-1].sum())) if policy == "custom" else dict(policy=policy, rare_indices=(2, 4)) if policy == "rare_mass" else dict(policy=policy)


def test_round_robin_spine_and_token_cost():
    sets = np.ones((3, 3, 2), dtype=bool)
    probs = np.full(sets.shape, .5)
    counts = np.array([[1, 100, 1000], [1, 2, 3], [2, 4, 6]])
    result = allocate_query_budget(sets, probs, counts, ["c", "a", "b"], 4)
    assert [a.obj_id for a in result.actions] == ["a", "b", "c", "a"]
    assert result.final_look.tolist() == [1, 2, 1]
    assert result.predictions.tolist() == [-1, -1, -1]
    assert result.status.tolist() == ["budget", "exhausted", "budget"]
    assert result.consumed_tokens == len(result.actions) == result.final_look.sum() == 4
    assert [a.observed_nobs for a in result.actions] == [2, 4, 100, 3]


@pytest.mark.parametrize("policy", POLICIES)
def test_future_counterfactual_cannot_change_first_action(policy):
    sets, probs, counts, ids = trajectories()
    before = allocate_query_budget(sets, probs, counts, ids, 1, **options(policy))
    # Valid but radically different unrevealed futures: counts cannot be used
    # to reject a query as unaffordable, nor can later set sizes rank it.
    sets[:, 1:] = False
    sets[1, 1:, 4] = True
    probs[:, 1:] = 0
    probs[:, 1:, 4] = 1
    counts[:, 1:] = 10**12
    after = allocate_query_budget(sets, probs, counts, ids, 1, **options(policy))
    assert before.actions[0].index == after.actions[0].index
    assert before.consumed_tokens == after.consumed_tokens == 1
    assert after.actions[0].observed_nobs == 10**12


def test_callback_has_only_independent_observed_prefixes_and_no_truth():
    sets, probs, counts, ids = trajectories(n=2)
    original = [a.copy() for a in (sets, probs, counts)]
    seen = []

    def scorer(observation):
        assert isinstance(observation, PrefixObservation)
        assert set(vars(observation)) == {"index", "obj_id", "look", "candidate_sets", "probabilities", "nobs"}
        for prefix, source in zip((observation.candidate_sets, observation.probabilities, observation.nobs), original):
            assert prefix.shape[0] == observation.look + 1
            assert not prefix.flags.writeable
            assert prefix.base is None
            assert not np.shares_memory(prefix, source)
        seen.append((observation.obj_id, observation.look))
        # Even a callback that re-enables writes sees its own copy, with no
        # backing view through which to reveal future data or alter closure.
        observation.candidate_sets.setflags(write=True)
        observation.candidate_sets[:] = False
        return -observation.look

    result = allocate_query_budget(sets, probs, counts, ids, 3, policy="custom", scorer=scorer)
    assert [a.obj_id for a in result.actions] == ["id-0", "id-1", "id-0"]
    assert result.predictions.tolist() == [-1, -1]
    assert seen == [("id-0", 0), ("id-1", 0), ("id-0", 1), ("id-1", 1)]
    for actual, expected in zip((sets, probs, counts), original):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(TypeError):
        allocate_query_budget(sets, probs, counts, ids, 1, y=np.array([0, 1]))


@pytest.mark.parametrize("policy", POLICIES)
def test_budget_increases_extend_same_actions_and_charge_one_per_advance(policy):
    data = list(trajectories())
    data[0][0, 1:, 1:] = False  # singleton closes after one token
    data[0][1, 2:] = False     # empty closes after two
    previous = None
    for budget in range(12):
        result = allocate_query_budget(*data, budget, **options(policy))
        assert result.consumed_tokens == len(result.actions) == result.final_look.sum()
        assert result.consumed_tokens == min(budget, 6)
        assert [a.token for a in result.actions] == list(range(1, result.consumed_tokens + 1))
        assert all(a.to_look == a.from_look + 1 for a in result.actions)
        if previous is not None:
            assert result.actions[:previous.consumed_tokens] == previous.actions
            assert (result.final_look >= previous.final_look).all()
            selected = previous.predictions >= 0
            np.testing.assert_array_equal(result.predictions[selected], previous.predictions[selected])
        previous = result
    assert result.final_look.tolist() == [1, 2, 3]
    assert result.status.tolist() == ["automatic", "empty", "exhausted"]


@pytest.mark.parametrize("policy", POLICIES)
def test_singleton_and_empty_are_absorbing_for_every_arm(policy):
    sets, probs, counts, ids = trajectories(n=5, looks=3)
    sets[0, :, 1:] = False  # free initial singleton
    sets[0, 1:] = False    # never revise an earlier singleton to abstention
    sets[1] = False        # free initial empty
    sets[2, 1:, :] = False
    sets[3, 1:, 1:] = False
    sets[4, 2:, 1:] = False
    result = allocate_query_budget(sets, probs, counts, ids, 100, **options(policy))
    assert result.final_look.tolist() == [0, 0, 1, 1, 2]
    assert result.predictions.tolist() == [0, -1, -1, 0, 0]
    assert result.status.tolist() == ["automatic", "empty", "empty", "automatic", "automatic"]
    assert result.consumed_tokens == 4
    assert {a.obj_id for a in result.actions} == set(ids[2:])


def test_round_robin_reaches_every_eligible_object_before_next_wave():
    sets, probs, counts, ids = trajectories(n=4)
    sets[0, :, 1:] = False
    for budget in (3, 6, 9):
        result = allocate_query_budget(sets, probs, counts, ids, budget)
        assert result.final_look.tolist() == [0] + [budget // 3] * 3


def test_heuristic_ranking_and_id_ties():
    sets, probs, counts, _ = trajectories(classes=3)
    ids = ["z", "a", "b"]
    sets[0, :, 2] = False
    result = allocate_query_budget(sets, probs, counts, ids, 1, policy="candidate_count")
    assert result.actions[0].obj_id == "a"
    probs[0, 0] = [.98, .01, .01]
    probs[1, 0] = [.8, .1, .1]
    result = allocate_query_budget(sets, probs, counts, ids, 1, policy="entropy")
    assert result.actions[0].obj_id == "b"
    result = allocate_query_budget(sets, probs, counts, ids, 1, policy="rare_mass", rare_indices=(0,))
    assert result.actions[0].obj_id == "z"
    result = allocate_query_budget(sets, probs, counts, ids, 2, policy="custom", scorer=lambda obs: obs.nobs[-1])
    assert [a.obj_id for a in result.actions] == ["a", "a"]


def test_seeded_random_repeatability_and_seed_effect():
    data = trajectories(n=20)
    first = allocate_query_budget(*data, 30, policy="random", seed=817)
    repeat = allocate_query_budget(*data, 30, policy="random", seed=817)
    other = allocate_query_budget(*data, 30, policy="random", seed=818)
    assert first.actions == repeat.actions
    assert first.actions != other.actions


def test_confidence_orders_by_current_maximum_then_object_id():
    sets, probs, counts, _ = trajectories(classes=3)
    probs[0, :, :] = [.2, .6, .2]
    probs[1, :, :] = [.9, .05, .05]
    probs[2, :, :] = [.05, .9, .05]
    result = allocate_query_budget(sets, probs, counts, ["a", "z", "b"], 4, policy="confidence")
    assert [action.obj_id for action in result.actions] == ["b", "b", "b", "z"]
    assert [action.score for action in result.actions] == pytest.approx([.9] * 4)


@pytest.mark.skipif(np.finfo(np.longdouble).max <= np.finfo(float).max,
                    reason="requires a floating dtype with wider range than Python float")
@pytest.mark.parametrize("sign", [-1, 1])
def test_custom_scorer_rejects_finite_value_that_overflows_python_float(sign):
    value = sign * np.finfo(np.longdouble).max
    assert np.isfinite(value)
    with pytest.raises(ValueError, match="representable as a finite float"):
        allocate_query_budget(*trajectories(), 1, policy="custom", scorer=lambda obs: value)


def test_zero_budget_last_look_and_empty_batch():
    data = trajectories()
    result = allocate_query_budget(*data, 0)
    assert result.final_look.tolist() == [0, 0, 0]
    assert result.status.tolist() == ["budget"] * 3
    assert result.consumed_tokens == 0 and result.actions == ()
    sets, probs, counts, ids = trajectories(looks=1)
    sets[0, 0, 1:] = False
    sets[1] = False
    result = allocate_query_budget(sets, probs, counts, ids, 100)
    assert result.status.tolist() == ["automatic", "empty", "exhausted"]
    assert result.consumed_tokens == 0
    result = allocate_query_budget(*trajectories(n=0), 100)
    assert result.final_look.shape == result.predictions.shape == result.status.shape == (0,)
    assert result.consumed_tokens == 0


@pytest.mark.parametrize("bad", [-1, 1.5, True, np.nan, "2"])
def test_invalid_budget(bad):
    with pytest.raises(ValueError, match="budget"):
        allocate_query_budget(*trajectories(), bad)


@pytest.mark.parametrize("which", ["non_nested", "set_values", "set_shape", "prob_shape", "prob_nan", "prob_negative", "prob_sum", "count_shape", "count_nan", "count_negative", "count_decreasing", "id_count", "id_duplicate", "id_empty", "id_type", "id_mixed"])
def test_invalid_trajectories(which):
    sets, probs, counts, ids = trajectories()
    if which == "non_nested":
        sets[0, 0, 0] = False
    elif which == "set_values":
        sets = sets.astype(float)
        sets[0, 0, 0] = .5
    elif which == "set_shape":
        sets = sets[:, 0]
    elif which == "prob_shape":
        probs = probs[:, 0]
    elif which == "prob_nan":
        probs[0, 1, 0] = np.nan
    elif which == "prob_negative":
        probs[0, 0, :2] = [-.1, 1.1]
    elif which == "prob_sum":
        probs[0, 0] *= 2
    elif which == "count_shape":
        counts = counts[:, 0]
    elif which == "count_nan":
        counts = counts.astype(float)
        counts[0, 0] = np.nan
    elif which == "count_negative":
        counts[0, 0] = -1
    elif which == "count_decreasing":
        counts[0, -1] = 0
    elif which == "id_count":
        ids = ids[:-1]
    elif which == "id_duplicate":
        ids[0] = ids[1]
    elif which == "id_empty":
        ids[0] = ""
    elif which == "id_type":
        ids = [1, 2, 3]
    elif which == "id_mixed":
        ids[0] = 3
    with pytest.raises(ValueError):
        allocate_query_budget(sets, probs, counts, ids, 1)


@pytest.mark.parametrize("kwargs", [
    dict(policy="unknown"), dict(policy="custom"), dict(scorer=lambda obs: 0),
    dict(policy="random", seed=-1), dict(policy="random", seed=True),
    dict(policy="rare_mass"), dict(policy="rare_mass", rare_indices=(2, 2)), dict(policy="rare_mass", rare_indices=(6,)),
    dict(policy="rare_mass", rare_indices=(1.5,)), dict(policy="rare_mass", rare_indices=()),
    dict(policy="custom", scorer=lambda obs: np.nan), dict(policy="custom", scorer=lambda obs: [1]),
])
def test_invalid_policy_configuration(kwargs):
    with pytest.raises(ValueError):
        allocate_query_budget(*trajectories(), 1, **kwargs)
