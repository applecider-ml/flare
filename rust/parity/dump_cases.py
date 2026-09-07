"""Dump parity cases for flare-rs from the benchmark.

  cases.json      detections (mjd, fid, mag, err) reconstructed from the training
                  event arrays, plus the context columns of each object
  py_features.json  the Python package's features for the same objects
  matrix.json     the 190-column feature matrix of 3000 benchmark rows, with the
                  Python boosters' raw scores, for the LightGBM evaluator test
"""
import glob, json, sys, numpy as np, pandas as pd
sys.path.insert(0, '/projects/bcrv/asasli/later/flare')
from flare import data as D, features as FE
from flare.hierarchical import HierarchicalFlare
OUT = '/projects/bcrv/asasli/later/flare/rust/parity/'
ZP = 23.9
rng = np.random.default_rng(0)
files = sorted(glob.glob('/projects/bcrv/asasli/later/bts_dataset/data/events/*.npz'))
pick = [files[i] for i in rng.choice(len(files), 300, replace=False)]
clf = HierarchicalFlare.from_pretrained(); FN = list(clf.feature_names)
X = pd.read_parquet('/projects/bcrv/asasli/later/flare/benchmark/features_all.parquet')
for f in ['hosts.csv', 'hosts_photoz.csv', 'context_gaia_wise.csv', 'ps1_position.csv']:
    d = pd.read_csv('/projects/bcrv/asasli/later/flare/benchmark/' + f).drop_duplicates('obj_id')
    X = X.merge(d[['obj_id'] + [c for c in d.columns if c in FN and c not in X.columns]], on='obj_id', how='left')
for c in FN:
    if c not in X.columns: X[c] = np.nan
X = X.set_index('obj_id')
ctx_cols = [c for c in FN if c not in FE.extract_from_array(D.load_events(pick[0])).keys()]
cases, pyfeat = [], []
for fp in pick:
    oid = fp.split('/')[-1][:-4]
    arr = D.load_events(fp)
    dt = arr[:, D.COL['dt']].astype(np.float64); band = arr[:, D.COL['band_id']].astype(int)
    lf = arr[:, D.COL['logflux']].astype(np.float64); lfe = arr[:, D.COL['logflux_err']].astype(np.float64)
    dets = [[59000.0 + float(t), int(b) + 1, float(ZP - 2.5 * l), float(2.5 * e)] for t, b, l, e in zip(dt, band, lf, lfe)]
    ctx = {c: (None if not np.isfinite(X.loc[oid, c]) else float(X.loc[oid, c])) for c in ctx_cols} if oid in X.index else {}
    cases.append(dict(id=oid, detections=dets, context=ctx))
    f = FE.extract_from_array(arr); f.update({k: (np.nan if v is None else v) for k, v in ctx.items()})
    pyfeat.append(dict(id=oid, quality=bool(D.passes_quality(arr)), features={k: (None if not np.isfinite(v) else float(v)) for k, v in f.items()}))
json.dump(cases, open(OUT + 'cases.json', 'w')); json.dump(pyfeat, open(OUT + 'py_features.json', 'w'))
# LightGBM matrix
Xs = X.sample(3000, random_state=0)[FN]
rows = [[None if not np.isfinite(v) else float(v) for v in r] for r in Xs.values]
raw = {name: b.predict(Xs[b.feature_name()], raw_score=True).tolist() for name, b in [('top', clf.top), ('slsn_branch', clf.branch), ('ad_space', clf.ad_space)]}
json.dump(dict(feature_names=FN, rows=rows, raw=raw), open(OUT + 'matrix.json', 'w'))
# python reports for the 300 cases (via the same maps)
reps = []
for c in pyfeat:
    if not c['quality']: reps.append(dict(id=c['id'], error='quality')); continue
    df = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in c['features'].items()}])
    for col in FN:
        if col not in df.columns: df[col] = np.nan
    P6 = clf.predict_proba(df)[0]; lab = clf.predict(df)[0]; sets = clf.prediction_sets(df)[0]; E = float(clf.anomaly_energy(df)[0]); pv = clf.p_values(df)[0]
    reps.append(dict(id=c['id'], proba=P6.tolist(), label=lab, set=sets, energy=E, p_values=pv.tolist()))
json.dump(reps, open(OUT + 'py_reports.json', 'w'))
print('cases', len(cases), 'quality pass', sum(p['quality'] for p in pyfeat), '| matrix rows', len(rows), '| context cols', len(ctx_cols))
