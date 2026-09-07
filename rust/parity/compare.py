"""Compare flare-rs against the Python package on the dumped cases."""
import json, subprocess, sys, numpy as np
P = '/projects/bcrv/asasli/later/flare/rust/parity/'
BIN = __import__('os').environ.get('FLARE_BIN', '/projects/bcrv/asasli/later/flare/rust/target/release/flare')
MODEL = '/projects/bcrv/asasli/later/flare/models/bts6'
def run(*a): return subprocess.run([BIN, *a], capture_output=True, text=True, check=True).stdout.strip().splitlines()

# ---- 1. LightGBM evaluator: raw scores on 3000 rows
mat = json.load(open(P + 'matrix.json'))
for name in ['top', 'slsn_branch', 'ad_space']:
    rs = np.array([json.loads(l) for l in run('lgbm', f'{MODEL}/{name}.txt', P + 'matrix.json')])
    ref = np.array(mat['raw'][name]); ref = ref.reshape(rs.shape)
    d = np.abs(rs - ref); print(f'lgbm {name:12s} rows {len(rs)}  max|d raw| {d.max():.2e}  rows>1e-9: {(d.max(1) > 1e-9).sum()}')

# ---- 2. features
rs = {json.loads(l)['id']: json.loads(l) for l in run('features', P + 'cases.json')}
py = {c['id']: c for c in json.load(open(P + 'py_features.json'))}
worst = {}; nmiss = 0; qual_mismatch = 0
for oid, pc in py.items():
    rc = rs[oid]
    if rc['quality'] != pc['quality']: qual_mismatch += 1
    for k, pv in pc['features'].items():
        rv = rc['features'].get(k)
        if rv is None and pv is None: continue
        if (rv is None) != (pv is None): nmiss += 1; worst.setdefault(k, []).append(('nan-mismatch', oid)); continue
        rel = abs(rv - pv) / max(abs(pv), 1e-9)
        worst.setdefault(k, []).append((rel, oid))
print(f'\nfeatures: {len(py)} objects, quality mismatches {qual_mismatch}, NaN-pattern mismatches {nmiss}')
rows = []
for k, lst in worst.items():
    rels = [r for r, _ in lst if r != 'nan-mismatch']
    rows.append((max(rels) if rels else 0.0, k, np.median(rels) if rels else 0.0, sum(1 for r in rels if r > 1e-6)))
rows.sort(reverse=True)
print(f'{"feature":45s}{"max rel":>10s}{"median rel":>12s}{"n>1e-6":>8s}')
for mx, k, med, n in rows[:25]: print(f'{k:45s}{mx:10.2e}{med:12.2e}{n:8d}')
exact = sum(1 for mx, k, med, n in rows if mx <= 1e-6); print(f'\n{exact} of {len(rows)} feature columns agree to 1e-6 relative on every object')
bazin = [r for r in rows if 'bazin' in r[1]]; print('bazin columns (MCMC vs Ceres) median rel:', np.median([r[2] for r in bazin]) if bazin else None)

# ---- 3. end-to-end reports (Python features -> Python model vs Rust features -> Rust model)
rr = {json.loads(l)['id']: json.loads(l) for l in run('predict', '--model-dir', MODEL, P + 'cases.json')}
pr = {c['id']: c for c in json.load(open(P + 'py_reports.json'))}
agree_label = agree_set = n = 0; dp = []; de = []
for oid, p in pr.items():
    r = rr[oid]
    if 'error' in p or 'error' in r:
        if ('error' in p) != ('error' in r): print('quality disagreement', oid, p.get('error'), r.get('error'))
        continue
    n += 1; rep = r['report']
    agree_label += rep['label'] == p['label']; agree_set += sorted(rep['set']) == sorted(p['set'])
    dp.append(np.abs(np.array(rep['proba']) - np.array(p['proba'])).max()); de.append(abs(rep['anomaly']['energy'] - p['energy']))
dp, de = np.array(dp), np.array(de)
print(f'\nend-to-end on {n} objects: label agreement {agree_label/n:.3f}, set agreement {agree_set/n:.3f}, max|dP| median {np.median(dp):.2e} / 95% {np.percentile(dp,95):.2e} / max {dp.max():.2e}, |dE| median {np.median(de):.2e} max {de.max():.2e}')
print('(build the Rust crate with --features ceres for exact agreement; the default MCMC Bazin gives different fits)')
