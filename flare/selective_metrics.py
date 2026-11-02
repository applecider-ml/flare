"""Descriptive counts and rates for classification with abstention."""
from __future__ import annotations

import numpy as np


def _indices(values, name, n_classes, *, abstention=False):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not isinstance(values, np.ndarray):
        # Inspect sequences before NumPy can coerce mixed bool/int or float/int.
        objects = np.asarray(values, dtype=object)
        if any(isinstance(x, (bool, np.bool_)) or
               not isinstance(x, (int, np.integer)) for x in objects):
            raise ValueError(f"{name} must contain integer indices, not bools or floats")
        if not len(array):
            return np.empty(0, dtype=np.intp)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must have an integer dtype, not bool or float")
    valid = (array >= 0) & (array < n_classes)
    if abstention:
        valid |= array == -1
    if not valid.all():
        domain = "-1 or [0, n_classes)" if abstention else "[0, n_classes)"
        raise ValueError(f"{name} indices must be in {domain}")
    return array.astype(np.intp, copy=False)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _counts(n_total, n_selected, n_correct):
    n_wrong = n_selected - n_correct
    return dict(n_total=n_total, n_selected=n_selected,
                n_abstained=n_total - n_selected, n_correct=n_correct, n_wrong=n_wrong,
                selection_rate=_ratio(n_selected, n_total),
                unconditional_error=_ratio(n_wrong, n_total),
                selective_risk=_ratio(n_wrong, n_selected))


def selective_classification_report(y_true, y_pred, *, class_names):
    """Report pooled, true-class, and emitted-label selective diagnostics.

    ``class_names`` is a nonempty ordered sequence of unique nonempty strings;
    index ``i`` denotes its ith name. No taxonomy is assumed. ``y_true`` and
    ``y_pred`` must be equally sized 1D integer arrays or integer sequences.
    Truth indices are in ``[0, len(class_names))``; predictions additionally
    accept ``-1`` for abstention. Bools and floats are rejected, not coerced.

    ``overall`` and each ``by_true_class`` group contain total, selected,
    abstained, correct, and wrong counts. Abstentions are neither correct nor
    wrong. ``selection_rate`` is selected/total, ``unconditional_error`` is
    wrong/total, and ``selective_risk`` is wrong/selected. Each
    ``by_predicted_class`` group instead counts emitted labels and reports
    ``precision`` = correct/emitted, a different conditioning event.

    ``selected_confusion_true_rows_predicted_columns`` is a KxK count matrix:
    true classes are rows, predicted classes are columns, both in class_names
    order. It excludes abstentions, counted separately in each true-class
    group's ``n_abstained``. All classes appear even if absent or never emitted.

    Empty inputs are supported. Zero-denominator rates are None; all returned
    values are JSON serializable. Overall rates use pooled counts: concatenate
    observations before reporting across folds, rather than averaging fold
    percentages. These are descriptive metrics, with no confidence intervals
    or coverage guarantee; this function does not change any decisions.
    """
    if isinstance(class_names, (str, bytes, dict, set, frozenset)):
        raise ValueError("class_names must be an ordered sequence of unique names")
    try:
        names = list(class_names)
    except TypeError as exc:
        raise ValueError("class_names must be an ordered sequence of unique names") from exc
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("class_names must contain unique nonempty strings")
    names = [str(name) for name in names]
    k = len(names)
    truth = _indices(y_true, "y_true", k)
    prediction = _indices(y_pred, "y_pred", k, abstention=True)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have equal lengths")
    selected = prediction != -1
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (truth[selected], prediction[selected]), 1)
    totals = np.bincount(truth, minlength=k)
    by_true = {}
    by_predicted = {}
    for i, name in enumerate(names):
        n_correct = int(confusion[i, i])
        n_selected = int(confusion[i].sum())
        n_emitted = int(confusion[:, i].sum())
        by_true[name] = _counts(int(totals[i]), n_selected, n_correct)
        by_predicted[name] = dict(n_emitted=n_emitted, n_correct=n_correct,
                                  n_wrong=n_emitted - n_correct,
                                  precision=_ratio(n_correct, n_emitted))
    return dict(class_names=names,
                overall=_counts(len(truth), int(selected.sum()), int(confusion.trace())),
                by_true_class=by_true, by_predicted_class=by_predicted,
                selected_confusion_true_rows_predicted_columns=confusion.tolist())
