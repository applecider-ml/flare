"""Portable exploratory trajectory-set and checkpoint-budget benchmark.

Input schema and public-data reproduction: docs/trajectory_benchmark.md.
The complete grids are descriptive; no policy is selected for deployment.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import numpy as np

from .trajectory_sets import (calibrate_sets, nested_candidate_sets, first_singleton,
                              summarize_sets, calibration_to_dict as calibration_record)
from .checkpoint_budget import allocate_query_budget
from .continuation import (VARIANTS, continuation_scorer, fit_continuation_models,
                           model_record, training_targets)

RARE_ALPHAS = (.01, .05, .10, .20)
ARMS = (("envelope_pooled", "envelope", False),
        ("envelope_mondrian", "envelope", True),
        ("bonferroni_pooled", "bonferroni", False),
        ("bonferroni_mondrian", "bonferroni", True),
        ("terminal_mondrian", "terminal", True),
        ("fixedlook_mondrian", "fixedlook", True))
ALPHAS = (.1, .2)
BUDGET_MULTIPLIERS = (0, .25, .5, 1, 2, 3)
HEURISTICS = ("round_robin", "entropy", "candidate_count", "rare_mass", "confidence")
RANDOM_SEEDS = (0, 1, 2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_fold(path):
    """Load the non-pickle fold schema; fail before any fitting or output writes."""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    version = data.get("schema_version", np.array(None))
    if version.shape != () or version.dtype.kind not in "iu" or version.item() != 1:
        raise ValueError("expected scalar schema_version=1")
    names = data.get("class_names", np.array([]))
    horizons = data.get("horizons", np.array([]))
    if (names.ndim != 1 or len(names) < 2 or names.dtype.kind != "U"
            or any(not name for name in names) or len(set(names)) != len(names)):
        raise ValueError("class_names must be unique nonempty strings in index order")
    if (horizons.ndim != 1 or not len(horizons) or horizons.dtype.kind not in "iuf"
            or not np.isfinite(horizons).all() or (horizons <= 0).any()
            or (horizons[1:] <= horizons[:-1]).any()):
        raise ValueError("horizons must be a finite positive increasing schedule")
    role = data.get("training_score_role", np.array(None))
    if role.shape != () or role.item() not in {"classifier_early_stop", "held_out_training", "synthetic"}:
        raise ValueError("training_score_role must identify classifier_early_stop, held_out_training, or synthetic")
    seen = set()
    for role in ("training", "calibration", "evaluation"):
        try:
            ids, probs, counts, y = (data[f"{role}_{field}"] for field in ("ids", "probabilities", "nobs", "y"))
        except KeyError as exc:
            raise ValueError(f"missing {role} array") from exc
        if (ids.ndim != 1 or ids.dtype.kind != "U" or any(not value for value in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError("role IDs must be nonempty unique strings, disjoint within each fold")
        seen.update(ids)
        if (probs.shape != (len(ids), len(horizons), len(names)) or probs.dtype.kind not in "iuf"
                or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
                or not np.allclose(probs.sum(axis=2), 1, atol=1e-7, rtol=0)):
            raise ValueError(f"invalid {role} probability trajectories")
        if (counts.shape != probs.shape[:2] or counts.dtype.kind not in "iuf"
                or not np.isfinite(counts).all() or (counts < 0).any()
                or (counts[:, 1:] < counts[:, :-1]).any()):
            raise ValueError(f"invalid {role} cumulative nobs")
        if y.shape != (len(ids),) or y.dtype.kind not in "iu" or (y < 0).any() or (y >= len(names)).any():
            raise ValueError(f"invalid {role} truth indices")
    if "classifier_fit_ids" in data:
        ids = data["classifier_fit_ids"]
        if (ids.ndim != 1 or ids.dtype.kind != "U" or any(not value for value in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError("classifier_fit_ids must be unique and disjoint from the three prediction roles")
    # Internal combined arrays keep the rare/budget runners on one common schema.
    data["probabilities"] = np.concatenate([data["calibration_probabilities"], data["evaluation_probabilities"]])
    data["nobs"] = np.concatenate([data["calibration_nobs"], data["evaluation_nobs"]])
    data["class_names"] = names.tolist()
    data["horizons"] = horizons.tolist()
    return data


def training_arrays(inputs):
    return {target: inputs[f"training_{source}"]
            for target, source in (("obj_id", "ids"), ("probabilities", "probabilities"),
                                   ("nobs", "nobs"), ("y", "y"))}
def _aggregate_groups(groups):
    """Pool counts/resource sums, never average per-fold fractions."""
    counts = ("n_objects", "n_selected", "n_unresolved", "n_wrong_automatic", "n_correct_automatic",
              "n_true_class_retained_unresolved", "n_true_class_lifetime_excluded",
              "n_true_class_consumed_prefix_excluded")
    out = {name: sum(g[name] for g in groups) for name in counts}
    n, selected = out["n_objects"], out["n_selected"]
    for target, numerator in (
        ("true_class_lifetime_exclusion", "n_true_class_lifetime_excluded"),
        ("true_class_consumed_prefix_exclusion", "n_true_class_consumed_prefix_excluded"),
        ("wrong_automatic_per_all", "n_wrong_automatic"),
        ("correct_automatic_per_all", "n_correct_automatic"),
        ("true_class_retained_unresolved_per_all", "n_true_class_retained_unresolved"),
        ("selection_coverage", "n_selected")):
        out[target] = out[numerator] / n if n else None
    out["selected_error"] = out["n_wrong_automatic"] / selected if selected else None
    unresolved = out["n_unresolved"]
    out["true_class_retained_given_unresolved"] = (out["n_true_class_retained_unresolved"] / unresolved
                                                 if unresolved else None)
    for metric, weight in (("mean_set_size_at_consumed", "n_objects"),
                           ("mean_set_size_unresolved", "n_unresolved")):
        denominator = out[weight]
        out[metric] = (sum(g[metric] * g[weight] for g in groups if g[weight]) / denominator
                       if denominator else None)
    out["mean_day_selected"] = (sum(g["mean_day_selected"] * g["n_selected"] for g in groups if g["n_selected"])
                                / selected if selected else None)
    out["mean_day_consumed_all"] = (sum(g["mean_day_consumed_all"] * g["n_objects"] for g in groups if g["n_objects"])
                                    / n if n else None)
    out["total_nobs_consumed"] = sum(g["total_nobs_consumed"] for g in groups)
    out["mean_nobs_consumed_all"] = out["total_nobs_consumed"] / n if n else None
    return out


def aggregate_folds(folds, class_names):
    combined = []
    keys = [(r["arm"], r["alpha"], r["budget_day"]) for r in folds[0]["records"]]
    indexed = [{(r["arm"], r["alpha"], r["budget_day"]): r for r in fold["records"]} for fold in folds]
    if any(set(index) != set(keys) or len(index) != len(keys) for index in indexed):
        raise ValueError("fold policy grids differ")
    for arm, alpha, budget_day in keys:
        metrics = [index[(arm, alpha, budget_day)]["metrics"] for index in indexed]
        aggregate = dict(overall=_aggregate_groups([m["overall"] for m in metrics]),
                         by_true_class={c: _aggregate_groups([m["by_true_class"][c] for m in metrics])
                                        for c in class_names})
        _pool_emissions(aggregate, metrics, class_names)
        combined.append(dict(arm=arm, alpha=alpha, budget_day=budget_day, metrics=aggregate))
    return combined


def run_fold(inputs, out):
    """Compute all frozen arms and budgets; selection cannot access evaluation y."""
    class_names, horizons = inputs["class_names"], inputs["horizons"]
    out = Path(out)
    out.mkdir()
    nc = len(inputs["calibration_ids"])
    probs, nobs = inputs["probabilities"], inputs["nobs"]
    cal_y, ev_y = inputs["calibration_y"], inputs["evaluation_y"]
    paths, records, calibrations = {}, [], {}
    for name, method, mondrian in ARMS:
        for alpha_index, alpha in enumerate(RARE_ALPHAS):
            base_key = f"{name}_a{alpha_index}"
            if method != "fixedlook":
                cal = calibrate_sets(probs[:nc], cal_y, alpha, method=method, mondrian=mondrian, class_names=class_names, horizons=horizons)
                sets = nested_candidate_sets(probs[nc:], cal)
                paths[f"{base_key}_candidate_sets"] = sets
                paths[f"{base_key}_thresholds"] = cal["thresholds"]
                calibrations[base_key] = calibration_record(cal)
            for budget_index, budget in enumerate(horizons):
                key = base_key
                if method == "fixedlook":
                    key = f"{base_key}_deadline{budget}"
                    cal = calibrate_sets(probs[:nc], cal_y, alpha, method=method, mondrian=mondrian,
                                         look_index=budget_index, class_names=class_names, horizons=horizons)
                    sets = nested_candidate_sets(probs[nc:], cal)
                    paths[f"{key}_candidate_sets"] = sets
                    paths[f"{key}_thresholds"] = cal["thresholds"]
                    calibrations[key] = calibration_record(cal)
                decision = first_singleton(sets, nobs[nc:], budget_day=budget, horizons=horizons)
                for field, values in decision.items():
                    paths[f"{key}_b{budget}_{field}"] = values
                metrics = summarize_sets(sets, decision, ev_y, class_names=class_names)
                metrics.update(prediction_diagnostics(decision["prediction"], ev_y, decision["selected"], class_names))
                records.append(dict(arm=name, alpha=alpha, alpha_index=alpha_index, budget_day=budget,
                                    calibration_key=key, metrics=metrics))
    np.savez_compressed(out / "sets_decisions.npz", evaluation_ids=inputs["evaluation_ids"], **paths)
    write_json(out / "calibration.json", calibrations)
    return dict(n_calibration=nc, n_evaluation=len(ev_y), records=records)


def prediction_diagnostics(predictions, y, selected, class_names):
    """Emitted-label precision is distinct from metrics grouped by TRUE class."""
    predictions, selected = np.asarray(predictions), np.asarray(selected)
    if predictions.shape != y.shape or selected.shape != y.shape or selected.dtype != np.bool_:
        raise ValueError("invalid prediction diagnostics dimensions")
    if ((predictions[~selected] != -1).any() or (predictions[selected] < 0).any()
            or (predictions[selected] >= len(class_names)).any()):
        raise ValueError("invalid automatic predictions")
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    np.add.at(confusion, (y[selected], predictions[selected]), 1)
    groups = {}
    for cls, name in enumerate(class_names):
        emitted = int(confusion[:, cls].sum())
        correct = int(confusion[cls, cls])
        groups[name] = dict(n_emitted=emitted, n_correct=correct, n_wrong=emitted-correct,
                            precision=correct/emitted if emitted else None,
                            error_given_emitted=(emitted-correct)/emitted if emitted else None)
    return dict(confusion_true_rows_predicted_columns=confusion.tolist(), by_predicted_class=groups)


def _add_derived_metrics(metrics):
    groups = list(metrics["by_true_class"].values())
    represented = [g for g in groups if g["n_objects"]]
    metrics["macro_true_class_correct_automatic_rate"] = (
        float(np.mean([g["correct_automatic_per_all"] for g in represented])) if represented else None)
    metrics["macro_true_class_wrong_automatic_rate"] = (
        float(np.mean([g["wrong_automatic_per_all"] for g in represented])) if represented else None)
    metrics["n_classes_represented_for_macro"] = len(represented)
    metrics["classes_represented_for_macro"] = [name for name, g in metrics["by_true_class"].items() if g["n_objects"]]
    return metrics


def summarize_allocation(sets, nobs, allocation, y, allocated_tokens, class_names, horizons):
    selected = allocation.status == "automatic"
    rows = np.arange(len(selected))
    final = allocation.final_look
    decision = dict(selected=selected, prediction=allocation.predictions, stop_index=final,
                    consumed_day=np.asarray(horizons)[final], nobs_consumed=nobs[rows, final])
    metrics = _add_derived_metrics(summarize_sets(sets, decision, y, class_names=class_names))
    metrics.update(prediction_diagnostics(allocation.predictions, y, selected, class_names))
    metrics["resources"] = dict(initial_checkpoint_queries=len(selected), allocated_extra_tokens=int(allocated_tokens),
                                consumed_extra_tokens=allocation.consumed_tokens,
                                unused_extra_tokens=int(allocated_tokens)-allocation.consumed_tokens,
                                total_checkpoint_queries=len(selected)+allocation.consumed_tokens)
    metrics["status_counts"] = {status: int(np.sum(allocation.status == status))
                                for status in ("automatic", "empty", "exhausted", "budget")}
    return metrics


def save_allocation(path, allocation, sets, nobs, evaluation_ids, horizons):
    final, rows = allocation.final_look, np.arange(len(evaluation_ids))
    actions = allocation.actions
    arrays = dict(evaluation_ids=evaluation_ids, final_look=final, prediction=allocation.predictions,
                  status=allocation.status, consumed_day=np.asarray(horizons)[final],
                  nobs_consumed=nobs[rows, final], set_size_at_stop=sets[rows, final].sum(axis=1),
                  final_candidate_sets=sets[rows, final])
    for field in ("token", "index", "from_look", "to_look"):
        arrays[f"action_{field}"] = np.array([getattr(a, field) for a in actions], dtype=int)
    arrays["action_obj_id"] = np.array([a.obj_id for a in actions], dtype=str)
    arrays["action_observed_nobs"] = np.array([a.observed_nobs for a in actions], dtype=float)
    arrays["action_score"] = np.array([np.nan if a.score is None else a.score for a in actions], dtype=float)
    np.savez_compressed(path, **arrays)


def run_budget_fold(inputs, gate, out, *, rare_indices):
    """All frozen cells; only the final summary call receives evaluation truth."""
    class_names, horizons = inputs["class_names"], inputs["horizons"]
    out = Path(out)
    out.mkdir()
    decision_dir = out / "decisions"
    decision_dir.mkdir()
    nc, ne = len(inputs["calibration_ids"]), len(inputs["evaluation_ids"])
    cal_probs, evaluation_probs = inputs["probabilities"][:nc], inputs["probabilities"][nc:]
    nobs = inputs["nobs"][nc:]
    all_models, metadata, records, path_arrays = {}, {}, [], {}
    for regime, mondrian in (("pooled", False), ("mondrian", True)):
        for alpha_index, alpha in enumerate(ALPHAS):
            key = f"envelope_{regime}_a{alpha_index}"
            cal = calibrate_sets(cal_probs, inputs["calibration_y"], alpha, mondrian=mondrian, class_names=class_names, horizons=horizons)
            sets = nested_candidate_sets(evaluation_probs, cal)
            gate_sets = nested_candidate_sets(gate["probabilities"], cal)
            targets = training_targets(gate_sets, gate["y"])
            np.savez_compressed(out / f"{key}_training_targets.npz", **targets)
            models = fit_continuation_models(gate["probabilities"], gate["nobs"], gate["y"], gate_sets, class_names=class_names, horizons=horizons)
            all_models[key] = models
            metadata[key] = dict(calibration=calibration_record(cal),
                                 models={name: model_record(model) for name, model in models.items()})
            path_arrays[f"{key}_evaluation_candidate_sets"] = sets
            path_arrays[f"{key}_training_candidate_sets"] = gate_sets
            path_arrays[f"{key}_thresholds"] = cal["thresholds"]
            # Reuse deterministic callbacks across budgets, memoizing ONLY
            # prefixes actually exposed in previous callback invocations.
            policies = [(name, name, None, None) for name in HEURISTICS]
            policies += [("random", f"random_seed{seed}", seed, None) for seed in RANDOM_SEEDS]
            policies += [("custom", name, None, continuation_scorer(models[name])) for name in VARIANTS]
            for multiplier in BUDGET_MULTIPLIERS:
                budget = int(np.floor(multiplier * ne))
                for policy, policy_name, seed, scorer in policies:
                    allocation = allocate_query_budget(
                        sets, evaluation_probs, nobs, inputs["evaluation_ids"], budget,
                        policy=policy, seed=0 if seed is None else seed, scorer=scorer, rare_indices=rare_indices)
                    filename = f"{key}_b{multiplier:g}_{policy_name}.npz"
                    path = decision_dir / filename
                    save_allocation(path, allocation, sets, nobs, inputs["evaluation_ids"], horizons)
                    metrics = summarize_allocation(sets, nobs, allocation, inputs["evaluation_y"], budget, class_names, horizons)
                    records.append(dict(regime=regime, alpha=alpha, policy=policy_name, random_seed=seed,
                                        budget_multiplier=multiplier, allocated_extra_tokens=budget,
                                        decision_artifact=f"decisions/{filename}", decision_sha256=sha(path), metrics=metrics))
    with (out / "models.pkl").open("wb") as stream:
        pickle.dump(all_models, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(out / "model_metadata.json", metadata)
    np.savez_compressed(out / "candidate_paths.npz", **path_arrays)
    return dict(n_gate_training=len(gate["obj_id"]), n_calibration=nc, n_evaluation=ne, records=records)


def aggregate_budget_folds(folds, class_names):
    keys = [(r["regime"], r["alpha"], r["policy"], r["budget_multiplier"]) for r in folds[0]["records"]]
    indexed = [{(r["regime"], r["alpha"], r["policy"], r["budget_multiplier"]): r for r in fold["records"]}
               for fold in folds]
    if any(set(index) != set(keys) or len(index) != len(keys) for index in indexed):
        raise ValueError("fold allocation grids differ")
    pooled = []
    for regime, alpha, policy, multiplier in keys:
        metrics = [index[(regime, alpha, policy, multiplier)]["metrics"] for index in indexed]
        aggregate = dict(overall=_aggregate_groups([m["overall"] for m in metrics]),
                         by_true_class={c: _aggregate_groups([m["by_true_class"][c] for m in metrics]) for c in class_names})
        _add_derived_metrics(aggregate)
        confusion = np.sum([m["confusion_true_rows_predicted_columns"] for m in metrics], axis=0)
        aggregate["confusion_true_rows_predicted_columns"] = confusion.tolist()
        aggregate["by_predicted_class"] = {}
        for c, cls in enumerate(class_names):
            emitted, correct = int(confusion[:, c].sum()), int(confusion[c, c])
            aggregate["by_predicted_class"][cls] = dict(n_emitted=emitted, n_correct=correct, n_wrong=emitted-correct,
                                                       precision=correct/emitted if emitted else None,
                                                       error_given_emitted=(emitted-correct)/emitted if emitted else None)
        aggregate["resources"] = {k: sum(m["resources"][k] for m in metrics) for k in metrics[0]["resources"]}
        aggregate["status_counts"] = {k: sum(m["status_counts"][k] for m in metrics) for k in metrics[0]["status_counts"]}
        pooled.append(dict(regime=regime, alpha=alpha, policy=policy, budget_multiplier=multiplier, metrics=aggregate))
    return pooled



def _pool_emissions(aggregate, metrics, class_names):
    confusion = np.sum([m['confusion_true_rows_predicted_columns'] for m in metrics], axis=0)
    aggregate['confusion_true_rows_predicted_columns'] = confusion.tolist()
    aggregate['by_predicted_class'] = {}
    for i, name in enumerate(class_names):
        emitted, correct = int(confusion[:, i].sum()), int(confusion[i, i])
        aggregate['by_predicted_class'][name] = dict(
            n_emitted=emitted, n_correct=correct, n_wrong=emitted - correct,
            precision=correct / emitted if emitted else None,
            error_given_emitted=(emitted - correct) / emitted if emitted else None)


def _csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_grid(path, folds, pooled, class_names, *, budget):
    """All fold and pooled cells, with counts sufficient to recover class rates."""
    rows = []
    for fold, records in [(f['fold'], f['records']) for f in folds] + [('pooled', pooled)]:
        for record in records:
            context = ('regime', 'alpha', 'policy', 'budget_multiplier') if budget else ('arm', 'alpha', 'budget_day')
            row = dict(fold=fold, **{key: record[key] for key in context})
            metrics = record['metrics']
            keys = ('n_objects', 'n_selected', 'n_unresolved', 'n_correct_automatic', 'n_wrong_automatic',
                    'n_true_class_lifetime_excluded', 'n_true_class_consumed_prefix_excluded',
                    'n_true_class_retained_unresolved')
            row.update({f'overall.{key}': metrics['overall'][key] for key in keys})
            row.update({f'overall.{key}': metrics['overall'][key] for key in (
                'wrong_automatic_per_all', 'selected_error', 'selection_coverage',
                'mean_day_consumed_all', 'total_nobs_consumed', 'mean_nobs_consumed_all')})
            for name in class_names:
                row.update({f'true.{name}.{key}': metrics['by_true_class'][name][key] for key in keys})
                row.update({f'emitted.{name}.{key}': metrics['by_predicted_class'][name][key]
                            for key in ('n_emitted', 'n_correct', 'n_wrong')})
            if budget:
                row.update(metrics['resources'])
                row.update({key: metrics[key] for key in (
                    'macro_true_class_correct_automatic_rate', 'macro_true_class_wrong_automatic_rate',
                    'n_classes_represented_for_macro')})
                row['classes_represented_for_macro'] = '|'.join(metrics['classes_represented_for_macro'])
            rows.append(row)
    _csv(path, rows)


def run_benchmark(input_dir, out, *, rare_classes, mode='all'):
    """Validate all folds first, then execute the complete chosen grid(s)."""
    input_dir, out = Path(input_dir), Path(out)
    if mode not in {'all', 'rare', 'budget'}:
        raise ValueError('mode must be all, rare, or budget')
    if out.exists():
        raise ValueError('output directory already exists; choose a new directory')
    paths = sorted(input_dir.glob('fold*.npz'))
    if not paths:
        raise ValueError('input directory must contain fold*.npz archives')
    data = [load_fold(path) for path in paths]
    class_names, horizons = data[0]['class_names'], data[0]['horizons']
    seen, truths = set(), {}
    for fold in data:
        if fold['class_names'] != class_names or fold['horizons'] != horizons:
            raise ValueError('fold class order and schedules must agree')
        if seen.intersection(fold['evaluation_ids']):
            raise ValueError('evaluation IDs must be unique across folds')
        seen.update(fold['evaluation_ids'])
        for role in ('training', 'calibration', 'evaluation'):
            for obj_id, y in zip(fold[f'{role}_ids'], fold[f'{role}_y']):
                if obj_id in truths and truths[obj_id] != int(y):
                    raise ValueError('truth for a repeated object ID differs across folds')
                truths[obj_id] = int(y)
    rare_classes = list(rare_classes)
    if (not rare_classes or len(set(rare_classes)) != len(rare_classes)
            or any(name not in class_names for name in rare_classes)):
        raise ValueError('rare_classes must explicitly name unique classes from class_names')
    rare_indices = tuple(class_names.index(name) for name in rare_classes)
    out.mkdir(parents=True)
    provenance = dict(schema_version=1, class_names=class_names, horizons=horizons,
                      rare_classes=rare_classes, input_sha256={p.name: sha(p) for p in paths},
                      training_score_roles=[str(f['training_score_role'].item()) for f in data],
                      package_versions={name: importlib.metadata.version(name)
                                        for name in ('numpy', 'scipy', 'scikit-learn', 'lightgbm')},
                      interpretation='Exploratory archived-checkpoint replay; no confirmatory or deployment claim',
                      cost='One extra token reveals one next archived checkpoint; initial checkpoint reported separately')
    provenance['code_sha256'] = {f'flare/{name}': sha(Path(__file__).with_name(name)) for name in (
        'trajectory_benchmark.py', 'trajectory_sets.py', 'checkpoint_budget.py', 'continuation.py')}
    provenance['grid'] = dict(rare_alphas=RARE_ALPHAS, rare_arms=[arm[0] for arm in ARMS],
                              budget_alphas=ALPHAS, budget_multipliers=BUDGET_MULTIPLIERS,
                              heuristics=HEURISTICS, random_seeds=RANDOM_SEEDS, continuation_variants=VARIANTS)
    write_json(out / 'provenance.json', provenance)
    count_rows = []
    for i, fold in enumerate(data):
        for c, name in enumerate(class_names):
            count_rows.append(dict(fold=i, class_name=name,
                                   **{f'n_{role}': int(np.sum(fold[f'{role}_y'] == c))
                                      for role in ('training', 'calibration', 'evaluation')}))
    _csv(out / 'class_counts.csv', count_rows)
    results = dict(provenance=provenance)
    for grid in ('rare', 'budget'):
        if mode != 'all' and mode != grid:
            continue
        directory = out / grid
        directory.mkdir()
        folds = []
        for i, fold in enumerate(data):
            if grid == 'rare':
                result = run_fold(fold, directory / f'fold{i}')
            else:
                result = run_budget_fold(fold, training_arrays(fold), directory / f'fold{i}', rare_indices=rare_indices)
            result['fold'] = i
            write_json(directory / f'fold{i}' / 'result.json', result)
            folds.append(result)
            print(f'{grid} fold={i}: {len(result["records"])} cells', flush=True)
        pooled = (aggregate_folds(folds, class_names) if grid == 'rare'
                  else aggregate_budget_folds(folds, class_names))
        results[grid] = dict(folds=folds, pooled=pooled)
        write_grid(out / f'{grid}_grid.csv', folds, pooled, class_names, budget=grid == 'budget')
    write_json(out / 'results.json', results)
    (out / 'results.json.gz').write_bytes(gzip.compress((out / 'results.json').read_bytes(), mtime=0))
    (out / 'COMPLETE').write_text(sha(out / 'results.json') + '\n')
    return results


def synthetic_archives(out, *, n_objects=24, seed=0):
    """Small reproducible archives for examples and integration tests, not evidence."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    for fold in range(2):
        arrays = dict(schema_version=np.array(1), class_names=np.array(['common', 'rare']),
                      horizons=np.array([5, 10, 20]), training_score_role=np.array('synthetic'))
        for role in ('training', 'calibration', 'evaluation'):
            y = rng.integers(0, 2, n_objects)
            probs = rng.dirichlet([2, 2], size=(n_objects, 3))
            # Increasing training/evaluation discrimination gives nontrivial closures.
            for look in range(3):
                probs[:, look] *= 1 - .25 * look
                probs[np.arange(n_objects), look, y] += .25 * look
            arrays.update({f'{role}_ids': np.array([f'{fold}_{role}_{i}' for i in range(n_objects)]),
                           f'{role}_probabilities': probs, f'{role}_y': y,
                           f'{role}_nobs': np.cumsum(rng.integers(1, 5, (n_objects, 3)), axis=1)})
        np.savez_compressed(out / f'fold{fold}.npz', **arrays)
