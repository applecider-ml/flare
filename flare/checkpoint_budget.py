"""Causal allocation of a batch budget of archived checkpoint queries.

Look zero is available for every object. Each token reveals exactly the next
checkpoint of one unresolved object, regardless of its future observation count.
These tokens are archive queries, not counts or costs of telescope observations.
Singletons are labelled immediately; empty sets and exhausted ambiguous paths
are referred. The scheduler never accepts evaluation truth.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import heapq
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class PrefixObservation:
    """One object's observed history; arrays own copies and hide later looks."""

    index: int
    obj_id: str
    look: int
    candidate_sets: np.ndarray
    probabilities: np.ndarray
    nobs: np.ndarray


@dataclass(frozen=True)
class BudgetAction:
    token: int
    index: int
    obj_id: str
    from_look: int
    to_look: int
    observed_nobs: float
    score: float | None


@dataclass(frozen=True)
class BudgetResult:
    final_look: np.ndarray
    predictions: np.ndarray
    status: np.ndarray
    consumed_tokens: int
    actions: tuple[BudgetAction, ...]


def _validate_inputs(candidate_sets, probabilities, nobs, obj_ids, budget):
    sets = np.asarray(candidate_sets)
    if (sets.ndim != 3 or sets.shape[1] == 0 or sets.shape[2] < 2
            or sets.dtype.kind not in "biuf" or not np.isin(sets, [0, 1]).all()):
        raise ValueError("candidate_sets must be a boolean/0-1 (objects, looks>0, classes>=2) array")
    sets = sets.astype(bool, copy=True)
    if np.any(sets[:, 1:] & ~sets[:, :-1]):
        raise ValueError("candidate sets must be nested across looks")
    try:
        probs = np.array(probabilities, dtype=float, copy=True)
        counts = np.array(nobs, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities and nobs must be numeric arrays") from exc
    if (probs.shape != sets.shape or not np.isfinite(probs).all()
            or (probs < 0).any() or (probs > 1).any()
            or not np.allclose(probs.sum(axis=2), 1, rtol=0, atol=1e-7)):
        raise ValueError("probabilities must match candidate_sets, be finite in [0,1], and sum to one")
    if (counts.shape != sets.shape[:2] or not np.isfinite(counts).all()
            or (counts < 0).any() or (np.diff(counts, axis=1) < 0).any()):
        raise ValueError("nobs must be finite, nonnegative, nondecreasing cumulative counts matching (objects, looks)")
    # Preserve element types until validation; numpy's default string promotion
    # would silently turn a mixed ["id", 3] batch into apparently valid IDs.
    ids = np.asarray(obj_ids, dtype=object)
    if (ids.shape != (len(sets),)
            or any(not isinstance(value, (str, np.str_)) or not value for value in ids)):
        raise ValueError("obj_ids must contain one nonempty string per object")
    ids = ids.astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("obj_ids must be unique")
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, Integral) or budget < 0:
        raise ValueError("budget must be a nonnegative integer number of query tokens")
    return sets, probs, counts, ids, int(budget)


def allocate_query_budget(
    candidate_sets, probabilities, nobs, obj_ids, budget, *,
    policy: str = "round_robin", seed: int = 0,
    scorer: Callable[[PrefixObservation], float] | None = None,
    rare_indices: Sequence[int] | None = None,
) -> BudgetResult:
    """Allocate a fixed number of unit-cost next-checkpoint queries causally.

    ``round_robin`` advances in waves, choosing smallest current look first.
    ``random`` draws uniformly from active objects at each token (seeded).
    ``confidence``, ``entropy``, ``candidate_count`` and ``rare_mass`` prioritize
    larger current maximum probability, entropy, set size, and probability mass
    on rare_indices, respectively.
    rare_mass requires explicit rare_indices in the supplied probability column order.
    All deterministic ties use ascending object ID then input index.

    ``custom`` requires a pure, deterministic per-object scorer returning a
    finite scalar; larger scores are queried first. It receives only read-only
    copies of that object's revealed prefix. Scores for unchanged objects are
    cached, so callbacks must not depend on external state or other objects.
    No callback sees total budget, future probabilities/counts/sets, or truth.
    Validation checks complete supplied trajectories, but ranking and closure
    use only revealed data. Inputs must contain valid nested trajectories.

    Output status is ``automatic``, ``empty``, ``exhausted`` (ambiguous at the
    last look), or ``budget`` (still ambiguous when tokens run out). Predictions
    are class indices for automatic labels and -1 for every referral/abstention.
    All final_look entries include the free initial look, including abstainers.
    """
    sets, probs, counts, ids, budget = _validate_inputs(
        candidate_sets, probabilities, nobs, obj_ids, budget)
    if policy not in {"round_robin", "random", "confidence", "entropy", "candidate_count", "rare_mass", "custom"}:
        raise ValueError("unknown budget policy")
    if (policy == "custom" and not callable(scorer)) or (policy != "custom" and scorer is not None):
        raise ValueError("a callable scorer is required exactly for policy='custom'")
    if policy == "random" and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0
    ):
        raise ValueError("random seed must be a nonnegative integer")
    if policy == "rare_mass":
        if rare_indices is None:
            raise ValueError("rare_mass requires explicit rare_indices")
        rare_indices = tuple(rare_indices)
        if (not rare_indices or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                                    or i < 0 or i >= sets.shape[2] for i in rare_indices)
                or len(set(rare_indices)) != len(rare_indices)):
            raise ValueError("rare_indices must contain distinct valid integer class indices")
    looks = np.zeros(len(sets), dtype=int)
    predictions = np.full(len(sets), -1, dtype=int)
    status = np.full(len(sets), "budget", dtype="U9")

    def finalize(index):
        current = sets[index, looks[index]]
        size = int(current.sum())
        if size == 1:
            predictions[index] = int(np.flatnonzero(current)[0])
            status[index] = "automatic"
        elif size == 0:
            status[index] = "empty"
        elif looks[index] == sets.shape[1] - 1:
            status[index] = "exhausted"
        return status[index] != "budget"

    def priority(index):
        look = int(looks[index])
        if policy == "round_robin":
            return float(-look)
        if policy == "candidate_count":
            return float(sets[index, look].sum())
        if policy == "confidence":
            return float(probs[index, look].max())
        if policy == "entropy":
            p = probs[index, look]
            return float(-(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum())
        if policy == "rare_mass":
            return float(probs[index, look, list(rare_indices)].sum())
        prefixes = []
        for values in (sets, probs, counts):
            prefix = values[index, :look + 1].copy()
            prefix.setflags(write=False)
            prefixes.append(prefix)
        observation = PrefixObservation(index, str(ids[index]), look, *prefixes)
        value = np.asarray(scorer(observation))
        if value.ndim != 0 or value.dtype.kind not in "biuf" or not np.isfinite(value):
            raise ValueError("custom scorer must return one finite scalar")
        # Extended precision scalars can be finite in their source dtype but
        # overflow the Python float stored in the heap and action history.
        try:
            score = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError("custom scorer must be representable as a finite float") from exc
        if not np.isfinite(score):
            raise ValueError("custom scorer must be representable as a finite float")
        return score

    # Heap updates touch only the advanced object. Random's swap-delete active
    # list avoids scanning the whole remaining batch for each uniform draw.
    active = sorted((i for i in range(len(sets)) if not finalize(i)), key=lambda i: (ids[i], i))
    if policy == "random":
        rng = np.random.default_rng(int(seed))
        heap = []
    else:
        heap = [(-priority(i), str(ids[i]), i) for i in active] if budget else []
        heapq.heapify(heap)
    actions = []
    while (active if policy == "random" else heap) and len(actions) < budget:
        if policy == "random":
            position = int(rng.integers(len(active)))
            index = active[position]
            obj_id, score = str(ids[index]), None
        else:
            negative_score, obj_id, index = heapq.heappop(heap)
            score = -negative_score
        previous = int(looks[index])
        looks[index] += 1
        actions.append(BudgetAction(len(actions) + 1, index, obj_id, previous,
                                    int(looks[index]), float(counts[index, looks[index]]),
                                    score))
        closed = finalize(index)
        if policy == "random":
            if closed:
                active[position] = active[-1]
                active.pop()
        elif not closed and len(actions) < budget:
            heapq.heappush(heap, (-priority(index), obj_id, index))
    return BudgetResult(looks, predictions, status, len(actions), tuple(actions))
