"""Selective diagnostics must keep truth and emitted-label denominators distinct."""
import json

import numpy as np
import pytest

from flare.selective_metrics import selective_classification_report as report


def test_counts_denominators_and_confusion_orientation():
    result = report([0, 0, 0, 1, 1, 2, 2, 2], [0, 1, -1, 1, 2, -1, 0, 2],
                    class_names=["a", "b", "c"])
    assert result["overall"] == dict(
        n_total=8, n_selected=6, n_abstained=2, n_correct=3, n_wrong=3,
        selection_rate=6 / 8, unconditional_error=3 / 8, selective_risk=3 / 6)
    assert result["selected_confusion_true_rows_predicted_columns"] == [
        [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    assert result["by_true_class"]["a"] == dict(
        n_total=3, n_selected=2, n_abstained=1, n_correct=1, n_wrong=1,
        selection_rate=2 / 3, unconditional_error=1 / 3, selective_risk=1 / 2)
    assert result["by_predicted_class"]["a"] == dict(
        n_emitted=2, n_correct=1, n_wrong=1, precision=1 / 2)
    for key in ("n_total", "n_selected", "n_abstained", "n_correct", "n_wrong"):
        assert sum(group[key] for group in result["by_true_class"].values()) == result["overall"][key]
    assert sum(group["n_emitted"] for group in result["by_predicted_class"].values()) == 6


@pytest.mark.parametrize("classes", [
    ["SNI", "SNII", "CV", "AGN", "TDE"],
    ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"],
])
def test_all_selected_and_explicit_five_or_six_class_order(classes):
    truth = np.arange(len(classes), dtype=np.uint64)
    prediction = np.roll(truth, 1)
    result = report(truth, prediction, class_names=tuple(classes))
    assert result["class_names"] == classes
    assert list(result["by_true_class"]) == classes
    assert list(result["by_predicted_class"]) == classes
    assert result["overall"]["n_abstained"] == 0
    assert result["overall"]["selection_rate"] == 1
    assert result["overall"]["selective_risk"] == result["overall"]["unconditional_error"] == 1
    for true_index, predicted_index in enumerate(prediction):
        assert result["selected_confusion_true_rows_predicted_columns"][true_index][predicted_index] == 1


def test_no_selections_absent_class_and_never_emitted_labels():
    result = report([0, 0], [-1, -1], class_names=["present", "absent"])
    assert result["overall"] == dict(
        n_total=2, n_selected=0, n_abstained=2, n_correct=0, n_wrong=0,
        selection_rate=0, unconditional_error=0, selective_risk=None)
    assert result["by_true_class"]["absent"] == dict(
        n_total=0, n_selected=0, n_abstained=0, n_correct=0, n_wrong=0,
        selection_rate=None, unconditional_error=None, selective_risk=None)
    for group in result["by_predicted_class"].values():
        assert group == dict(n_emitted=0, n_correct=0, n_wrong=0, precision=None)
    assert result["selected_confusion_true_rows_predicted_columns"] == [[0, 0], [0, 0]]


def test_absent_true_class_can_still_be_emitted():
    result = report([0, 0], [1, -1], class_names=["present", "absent"])
    assert result["by_true_class"]["absent"]["unconditional_error"] is None
    assert result["by_predicted_class"]["absent"]["precision"] == 0
    assert result["by_predicted_class"]["present"]["precision"] is None
    assert result["overall"]["unconditional_error"] == 1 / 2
    assert result["overall"]["selective_risk"] == 1


@pytest.mark.parametrize("empty", [[], np.array([], dtype=np.int64)])
def test_empty_dataset_and_strict_json(empty):
    result = report(empty, empty, class_names=["a", "b"])
    assert result["overall"] == dict(
        n_total=0, n_selected=0, n_abstained=0, n_correct=0, n_wrong=0,
        selection_rate=None, unconditional_error=None, selective_risk=None)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_true_minority_protection_does_not_imply_emitted_precision():
    result = report([1] * 5 + [0] * 62, [1] * 67, class_names=["other", "TDE"])
    true_tde = result["by_true_class"]["TDE"]
    assert true_tde["n_correct"] == true_tde["n_total"] == 5
    assert true_tde["unconditional_error"] == true_tde["selective_risk"] == 0
    assert result["by_predicted_class"]["TDE"] == dict(
        n_emitted=67, n_correct=5, n_wrong=62, precision=5 / 67)


def test_observation_and_class_permutations():
    truth = np.array([0, 0, 1, 2, 2])
    prediction = np.array([1, -1, 1, 0, 2])
    names = ["a", "b", "c"]
    original = report(truth, prediction, class_names=names)
    order = np.array([4, 1, 3, 0, 2])
    assert report(truth[order], prediction[order], class_names=names) == original
    # New class indices reorder c,a,b while retaining the same named outcomes.
    remap = np.array([1, 2, 0])
    permuted_pred = np.where(prediction == -1, -1, remap[np.maximum(prediction, 0)])
    permuted = report(remap[truth], permuted_pred, class_names=["c", "a", "b"])
    assert permuted["overall"] == original["overall"]
    assert permuted["by_true_class"] == original["by_true_class"]
    assert permuted["by_predicted_class"] == original["by_predicted_class"]
    index = [2, 0, 1]
    expected = np.array(original["selected_confusion_true_rows_predicted_columns"])[np.ix_(index, index)]
    assert permuted["selected_confusion_true_rows_predicted_columns"] == expected.tolist()


def test_pooled_metrics_use_concatenated_counts_not_fold_percentages():
    folds = [([0], [1]), ([0] * 9, [0] * 8 + [-1])]
    reports = [report(y, p, class_names=["a", "b"]) for y, p in folds]
    pooled = report(np.concatenate([y for y, _ in folds]),
                    np.concatenate([p for _, p in folds]), class_names=["a", "b"])
    for key in ("n_total", "n_selected", "n_abstained", "n_correct", "n_wrong"):
        assert pooled["overall"][key] == sum(r["overall"][key] for r in reports)
    assert pooled["overall"]["unconditional_error"] == 1 / 10
    assert pooled["overall"]["selective_risk"] == 1 / 9
    assert pooled["overall"]["unconditional_error"] != np.mean(
        [r["overall"]["unconditional_error"] for r in reports])


@pytest.mark.parametrize("bad", [
    [True], [0, True], [0.0], [0, 1.0], ["0"], [None], [[0]], 0,
    np.array([0.0]), np.array([], dtype=float), np.array([True]),
    np.array([0], dtype=object), np.array([0 + 0j]), [2], [2 ** 100],
    np.array([2 ** 64 - 1], dtype=np.uint64),
])
@pytest.mark.parametrize("argument", ["y_true", "y_pred"])
def test_reject_invalid_index_types_shapes_and_upper_bounds(bad, argument):
    kwargs = dict(y_true=[0], y_pred=[0], class_names=["a", "b"])
    kwargs[argument] = bad
    with pytest.raises(ValueError):
        report(**kwargs)


@pytest.mark.parametrize("truth,prediction", [([-1], [0]), ([0], [-2]), ([0, 1], [0])])
def test_reject_negative_truth_invalid_abstention_and_length_mismatch(truth, prediction):
    with pytest.raises(ValueError):
        report(truth, prediction, class_names=["a", "b"])


@pytest.mark.parametrize("names", [[], ["a", "a"], [""], [1], [None], "ab", None,
                                    {"a", "b"}, {"a": 0, "b": 1}, [["a"]]])
def test_reject_invalid_class_names(names):
    with pytest.raises(ValueError):
        report([0], [0], class_names=names)
