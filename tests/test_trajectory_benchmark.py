"""Portable archive validation, full-grid execution, and evaluation-truth isolation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from flare.trajectory_benchmark import load_fold, run_benchmark, run_budget_fold, synthetic_archives, training_arrays


@pytest.fixture
def archives(tmp_path):
    path = tmp_path / 'input'
    synthetic_archives(path, n_objects=12)
    return path


def rewrite(path, **changes):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


def test_cli_complete_grids_and_pooled_counts(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = repo / 'scripts' / 'run_trajectory_benchmark.py'
    source, output = tmp_path / 'input', tmp_path / 'output'
    subprocess.run([sys.executable, str(script), 'synthetic', '--out', str(source), '--objects', '8'],
                   check=True, cwd=tmp_path, capture_output=True, text=True)
    subprocess.run([sys.executable, str(script), 'run', '--input', str(source), '--out', str(output),
                    '--rare-classes', 'rare'], check=True, cwd=tmp_path, capture_output=True, text=True)
    payload = (output / 'results.json').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (output / 'COMPLETE').read_text().strip()
    results = json.loads(payload)
    assert results['provenance']['class_names'] == ['common', 'rare']
    assert results['provenance']['training_score_roles'] == ['synthetic', 'synthetic']
    for grid, cells in [('rare', 72), ('budget', 264)]:
        assert [len(f['records']) for f in results[grid]['folds']] == [cells, cells]
        assert len(results[grid]['pooled']) == cells
        assert all(cell['metrics']['overall']['n_objects'] == 16 for cell in results[grid]['pooled'])
        with (output / f'{grid}_grid.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3 * cells
        assert {row['fold'] for row in rows} == {'0', '1', 'pooled'}
    assert results['budget']['pooled'][0]['metrics']['classes_represented_for_macro'] == ['common', 'rare']
    with pytest.raises(ValueError, match='already exists'):
        run_benchmark(source, output, rare_classes=['rare'])


@pytest.mark.parametrize('changes', [
    {'training_ids': np.array(['duplicate'] * 12)},
    {'training_ids': np.array([f'0_calibration_{i}' for i in range(12)])},
    {'training_ids': np.arange(12)},
    {'class_names': np.array(['same', 'same'])},
    {'horizons': np.array([5, 5, 20])},
    {'horizons': np.array([5, 3, 9], dtype=np.uint64)},
    {'training_score_role': np.array('untouched_oof_without_role')},
    {'evaluation_y': np.zeros(12, dtype=bool)},
    {'evaluation_y': np.full(12, 2)},
    {'calibration_probabilities': np.full((12, 3, 2), np.nan)},
    {'calibration_probabilities': np.full((12, 3, 2), .7)},
    {'evaluation_nobs': np.tile([5, 3, 9], (12, 1))},
    {'evaluation_nobs': np.tile(np.array([5, 3, 9], dtype=np.uint64), (12, 1))},
    {'classifier_fit_ids': np.array(['0_evaluation_0'])},
    {'schema_version': np.array([1])},
    {'schema_version': np.array(True)},
])
def test_reject_invalid_archives_before_outputs(archives, tmp_path, changes):
    rewrite(archives / 'fold0.npz', **changes)
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        run_benchmark(archives, output, rare_classes=['rare'])
    assert not output.exists()


def test_reject_cross_fold_evaluation_overlap(archives, tmp_path):
    first = load_fold(archives / 'fold0.npz')
    rewrite(archives / 'fold1.npz', evaluation_ids=first['evaluation_ids'])
    with pytest.raises(ValueError, match='across folds'):
        run_benchmark(archives, tmp_path / 'rejected', rare_classes=['rare'])


def test_reject_inconsistent_truth_for_reused_cross_fold_objects(archives, tmp_path):
    first = load_fold(archives / 'fold0.npz')
    second = load_fold(archives / 'fold1.npz')
    ids, truth = second['training_ids'].astype(object), second['training_y'].copy()
    ids[0], truth[0] = first['evaluation_ids'][0], 1 - first['evaluation_y'][0]
    rewrite(archives / 'fold1.npz', training_ids=ids.astype(str), training_y=truth)
    with pytest.raises(ValueError, match='truth.*across folds'):
        run_benchmark(archives, tmp_path / 'rejected', rare_classes=['rare'])


def test_evaluation_truth_changes_only_diagnostics(archives, tmp_path):
    inputs = load_fold(archives / 'fold0.npz')
    first = tmp_path / 'first'
    second = tmp_path / 'changed_truth'
    original = run_budget_fold(inputs, training_arrays(inputs), first, rare_indices=(1,))
    inputs['evaluation_y'] = 1 - inputs['evaluation_y']
    changed = run_budget_fold(inputs, training_arrays(inputs), second, rare_indices=(1,))
    assert (first / 'model_metadata.json').read_bytes() == (second / 'model_metadata.json').read_bytes()
    assert len(original['records']) == len(changed['records']) == 264
    for path in first.rglob('*.npz'):
        relative = path.relative_to(first)
        with np.load(path, allow_pickle=False) as before, np.load(second / relative, allow_pickle=False) as after:
            assert before.files == after.files
            for key in before.files:
                np.testing.assert_array_equal(before[key], after[key])
