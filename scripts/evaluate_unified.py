"""Development benchmark for classification, AD, forecast intervals and Ia PE.

Design fixed before execution: paired StratifiedKFold(5, shuffle=True, seed=42),
same object partitions for every task and horizon. Outer train -> 20% calibration,
remaining -> 15% early stopping. Nothing is tuned on outer test or novel objects.
Classification compares a full-curve-trained hierarchy at early inference with
a horizon-trained hierarchy; each has its own untouched calibration sample.
PE is refitted on quality-selected true Ia in the SAME partitions, then evaluated
both on all eligible Ia and on a singleton-Ia gate from the early classifier.
Forecast uses adopted pooled L1 parameters and quantile scales; compare global
calibration with fixed groups (lead <=15d and n_obs <8), minimum 100 calibration
points per group. Score real future detections only, through day 90.
AD compares narrow-context five-way top-level energy with forecast residuals
available through day 50 (90th percentile standardized absolute residual per
object). Residuals require >=3 future detections. Known-object calibration ranks,
and fixed Bonferroni min-p combination; no tuning on the 52 novel objects.

Predictions: U1 early hierarchy macro F1 exceeds full-trained early inference by
>=0.02 at h=5. U2 singleton gate Ia purity >=0.95 at h=10. U3 grouping improves
SLSN/TDE 68% coverage by >=0.05 each at h=5 without >20% mean-width inflation.
U4 combined AD detects >=0.05 more of the novel sample than energy at h=10,
with measured known-object FAR <=0.02. These are hypotheses, not promises.

This is further development on repeatedly inspected BTS data, not a fresh test.
Forecast bands have empirical pointwise coverage, not simultaneous validity.
The same 52 novel objects appear in each fold: report per-object average flag
rates; never count them as 260 independent objects. No deployment replacement.
"""
from __future__ import annotations

import argparse, os
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from flare.analysis import feature_frame, observed_events, json_safe
from flare.config import BTS6, HOST_COLS, GW_COLS, POS_COLS
from flare.data import load_events
from flare.hierarchical import HierarchicalFlare

SIX = BTS6.classes
Y5 = np.array([0, 1, 1, 2, 3, 4])
TOP = dict(objective='multiclass', num_class=5, metric='multi_logloss', learning_rate=.06,
           num_leaves=40, min_child_samples=20, feature_fraction=.55, bagging_fraction=.85, lambda_l2=1.)
BR = dict(objective='binary', metric='binary_logloss', learning_rate=.05,
          num_leaves=24, min_child_samples=25, feature_fraction=.6, bagging_fraction=.85, lambda_l2=1.)


def quantile(values, level):
    """Finite-sample order statistic; unattainable levels give infinity."""
    v = np.sort(np.asarray(values)[np.isfinite(values)])
    k = int(np.ceil((len(v)+1)*level))
    return float(v[k-1]) if 0 < k <= len(v) else np.inf


def pvalue(cal, score):
    cal = np.sort(np.asarray(cal)[np.isfinite(cal)])
    result = (1 + len(cal) - np.searchsorted(cal, score, side='left')) / (len(cal)+1)
    return np.where(np.isfinite(score), result, np.nan)


def fit(X, y, E, ey, params, args, weight=None):
    ok, oke = np.isfinite(y), np.isfinite(ey)
    d = lgb.Dataset(X.loc[ok], label=y[ok], weight=None if weight is None else weight[ok])
    e = lgb.Dataset(E.loc[oke], label=ey[oke], reference=d)
    return lgb.train(dict(params, verbosity=-1, seed=42, num_threads=args.threads, bagging_freq=1),
                     d, args.rounds, valid_sets=[e], callbacks=[lgb.early_stopping(80, verbose=False)])


def hierarchy(X, y, tr, es, cal, args):
    xt = X.iloc[tr].copy()
    rng = np.random.default_rng(42)
    for block in (HOST_COLS, GW_COLS, POS_COLS, ['M_pseudo']):
        xt.loc[rng.random(len(xt)) < .3, block] = np.nan
    y5 = Y5[y]
    counts = np.bincount(y5[tr], minlength=5)
    weights = (len(tr)/(5*counts))**.8
    top = fit(xt, y5[tr], X.iloc[es], y5[es], TOP, args, weights[y5[tr]])
    ct, ce = np.isin(y[tr], [1, 2]), np.isin(y[es], [1, 2])
    yt, ye = (y[tr][ct] == 2).astype(int), (y[es][ce] == 2).astype(int)
    wb = np.where(yt == 1, (len(yt)/max(yt.sum(), 1))**.5, 1.)
    branch = fit(xt.loc[ct], yt, X.iloc[es].loc[ce], ye, BR, args, wb)
    pb = branch.predict(X.iloc[es].loc[ce], num_threads=args.threads)
    threshold = max(np.linspace(.05, .9, 35), key=lambda t: f1_score(ye, pb > t, zero_division=0))
    model = HierarchicalFlare(top, branch, threshold)
    pc = model.predict_proba(X.iloc[cal])
    model.qhat = np.array([quantile(1-pc[y[cal] == c, c], .9) for c in range(6)])
    return model


def long(X, days, truth=None):
    rows = [X.assign(fc_day=float(d), fc_band=float(b)) for b in (0, 1) for d in days]
    ys = np.concatenate([truth[:, b, int(d/5)] for b in (0, 1) for d in days]) if truth is not None else None
    return pd.concat(rows, ignore_index=True), ys


def context_table(root, ids):
    out = pd.DataFrame({'obj_id': ids})
    for filename, cols in [('hosts.csv', HOST_COLS), ('context_gaia_wise.csv', GW_COLS),
                           ('ps1_position.csv', POS_COLS), ('hosts_photoz.csv', ['z_phot'])]:
        src = pd.read_csv(root/'data'/filename).drop_duplicates('obj_id')
        out = out.merge(src[['obj_id']+cols], how='left', on='obj_id', validate='one_to_one')
    return out.set_index('obj_id')


def point_residuals(X, models, days, last, points, h, args):
    xl, _ = long(X, days)
    pred = [m.predict(xl, num_threads=args.threads).reshape(2, len(days), len(X)).transpose(2, 0, 1) for m in models]
    scale = np.maximum(np.abs(pred[2]-pred[1])/2, .05)
    rows, scores = [], np.full(len(X), np.nan)
    for i, fp in enumerate(points):
        fp = np.asarray(fp, dtype=float).reshape(-1, 3)
        residuals = []
        for t, b, truth in fp:
            b = int(b)
            if not h < t <= days[-1] or not np.isfinite(last[i, b]):
                continue
            mean = np.interp(t, [h]+days, np.r_[last[i,b], pred[0][i,b]])
            # Anchor scale: earliest predicted quantile width; calibration
            # is recomputed on real points, not borrowed from deployed bands.
            sd = np.interp(t, [h]+days, np.r_[scale[i,b,0], scale[i,b]])
            z = abs(mean-truth)/sd
            group = int(t-h > 15)*2 + int(X.iloc[i]['n_obs_total'] < 8)
            rows.append((i, abs(mean-truth), sd, z, group))
            if t <= 50:
                residuals.append(z)
        if len(residuals) >= 3:
            scores[i] = np.quantile(residuals, .9)
    return pd.DataFrame(rows, columns=['i', 'error', 'scale', 'z', 'group']), scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path(os.environ.get('FLARE_PAPER_DATA', PKG/'paper_data')))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--horizons', type=int, nargs='+', default=[5, 10, 20])
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--rounds', type=int, default=2000)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    root, cache = a.data, a.data/'results/reconstruction/cache'
    provenance = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                      interpretation='development benchmark; previously inspected population')
    (a.out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    df = pd.concat([pd.read_csv(root/f'data/splits/{s}.csv') for s in ('train', 'val', 'test')], ignore_index=True)
    y = np.array([SIX.index(c) for c in df.cls]); n = len(df)
    novel = pd.read_csv(root/'data/splits/anomaly.csv')
    ctx = context_table(root, pd.concat([df.obj_id, novel.obj_id]).values)
    lab = pd.read_csv(root/'results/salt3_labels_ia.csv').set_index('obj_id')
    labels = lab.reindex(df.obj_id)[['t0','x1','c','mB']].to_numpy(dtype=float)
    full = pd.read_parquet(root/'data/features_all.parquet').set_index('obj_id').reindex(df.obj_id)
    Xh, arrays, novel_X, novel_last, novel_points = {}, {}, {}, {}, {}
    for h in a.horizons:
        Xh[h] = pd.read_parquet(cache/f'X_pool_h{h}.parquet')
        z = np.load(root/f'results/reconstruction/forecast_v2_h{h}.npz', allow_pickle=True)
        assert np.array_equal(z['obj_id'], df.obj_id) and np.array_equal(z['cls'], y)
        assert len(Xh[h]) == n
        arrays[h] = np.load(cache/f'TL_pool_h{h}.npz', allow_pickle=True)
        rows, ls, fps = [], [], []
        for r in novel.itertuples():
            arr = load_events(root/'data'/r.filepath)
            early = observed_events(arr, h)
            f, peak = feature_frame(early, ctx.loc[r.obj_id].to_dict()); rows.append(f)
            last = np.full(2, np.nan)
            for b in (0, 1):
                band = early[early[:,2] == b]
                if len(band): last[b] = 23.9-2.5*band[-1,3]-peak
            ls.append(last)
            fut = arr[(arr[:,0] > h) & (arr[:,2] < 2)]
            fps.append(np.c_[fut[:,0], fut[:,2], 23.9-2.5*fut[:,3]-peak])
        novel_X[h] = pd.concat(rows, ignore_index=True).reindex(columns=Xh[h].columns)
        novel_last[h], novel_points[h] = np.array(ls), fps
    # Full-curve reference uses measured within-window peak, not catalog peak.
    X100 = Xh[a.horizons[0]].copy()
    for col in full.columns: X100[col] = full[col].to_numpy()
    from flare.context import lumdist_mpc
    peak = 23.9-2.5*full[['g_logflux_peak','r_logflux_peak','i_logflux_peak']].max(axis=1).to_numpy()
    X100['M_pseudo'] = peak-5*np.log10(lumdist_mpc(ctx.loc[df.obj_id].z_phot.to_numpy())*1e5)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    results, cls_rows, pe_rows, ad_rows = [], [], [], []
    chosen = json.loads((root/'results/reconstruction/final_choice.json').read_text())
    for fi in a.folds:
        tr0, te = folds[fi]
        tr1, cal = train_test_split(tr0, test_size=.2, stratify=y[tr0], random_state=fi)
        tr, es = train_test_split(tr1, test_size=.15, stratify=y[tr1], random_state=fi)
        np.savez(a.out/f'fold{fi}.npz', obj_id=df.obj_id.to_numpy(dtype=str), fit=tr, early_stop=es, calibration=cal, test=te)
        full_model = hierarchy(X100, y, tr, es, cal, a)
        for h in a.horizons:
            X, z = Xh[h], arrays[h]
            early = hierarchy(X, y, tr, es, cal, a)
            gates = {}
            for name, model in [('full_trained', full_model), ('horizon_trained', early)]:
                pred = model.predict(X.iloc[te]); sets = model.prediction_sets(X.iloc[te])
                gate = np.array([s == ['SN_Ia'] for s in sets]); gates[name] = gate
                res = dict(task='classification', h=h, fold=fi, variant=name,
                           macro_f1=f1_score(df.cls.iloc[te], pred, labels=SIX, average='macro', zero_division=0),
                           coverage=np.mean([SIX[y[i]] in s for i,s in zip(te,sets)]),
                           mean_set_size=np.mean([len(s) for s in sets]), gate_n=int(gate.sum()),
                           gate_purity=float(np.mean(y[te][gate] == 0)) if gate.any() else None,
                           gate_ia_completeness=float(gate[y[te] == 0].mean()))
                results.append(res)
                for i,p,s,g in zip(te,pred,sets,gate):
                    cls_rows.append(dict(obj_id=df.obj_id[i], h=h, fold=fi, variant=name, truth=SIX[y[i]], predicted=p, candidates='|'.join(s), gate=bool(g)))
            # PE shares object folds with the gate. Non-Ia have no comparable labels.
            for j,target in enumerate(('t0','x1','c','mB')):
                valid = (y == 0) & np.isfinite(labels[:,j])
                it, ie, ic = [idx[valid[idx]] for idx in (tr, es, cal)]
                params = dict(chosen['pe'][target], bagging_fraction=.8)
                m = fit(X.iloc[it], labels[it,j], X.iloc[ie], labels[ie,j], params, a)
                q = quantile(np.abs(m.predict(X.iloc[ic], num_threads=a.threads)-labels[ic,j]), .68)
                pt = m.predict(X.iloc[te], num_threads=a.threads)
                for pos,i in enumerate(te):
                    if valid[i]: pe_rows.append(dict(obj_id=df.obj_id[i], h=h, fold=fi, target=target,
                        error=abs(pt[pos]-labels[i,j]), covered=abs(pt[pos]-labels[i,j]) <= q,
                        halfwidth=q, gate=bool(gates['horizon_trained'][pos])))
            # Forecasts and quantile scales trained without calibration/test objects.
            days = [d for d in (10,15,20,25,30,40,50,60,75,90) if d > h]
            xt, yt = long(X.iloc[tr], days, z['T'][tr]); xe, ye = long(X.iloc[es], days, z['T'][es])
            models = []
            for alpha in (None, .16, .84):
                params = dict(chosen['forecaster'])
                if alpha is not None: params.update(objective='quantile', alpha=alpha)
                models.append(fit(xt, yt, xe, ye, params, a))
            pc, sc = point_residuals(X.iloc[cal], models, days, z['L'][cal], z['FP'][cal], h, a)
            pt, st = point_residuals(X.iloc[te], models, days, z['L'][te], z['FP'][te], h, a)
            pn, sn = point_residuals(novel_X[h], models, days, novel_last[h], novel_points[h], h, a)
            for level in (.68,.95):
                qglobal = quantile(pc.z.to_numpy(), level)
                qgroups = {g: quantile(v.z.to_numpy(), level) if len(v) >= 100 else qglobal for g,v in pc.groupby('group')}
                for variant in ('global', 'lead_quality'):
                    q = np.full(len(pt), qglobal) if variant == 'global' else pt.group.map(qgroups).fillna(qglobal).to_numpy()
                    width = q*pt.scale.to_numpy(); truth = y[te[pt.i.to_numpy(dtype=int)]]
                    for c in [-1]+list(range(6)):
                        take = np.ones(len(pt),bool) if c == -1 else truth == c
                        results.append(dict(task='forecast', h=h, fold=fi, variant=variant, level=level,
                            cls='ALL' if c == -1 else SIX[c], n=int(take.sum()),
                            coverage=float((pt.error.to_numpy()[take] <= width[take]).mean()),
                            mean_width=float(2*width[take].mean()), median_error=float(np.median(pt.error.to_numpy()[take]))))
            # Narrow-context energy, including host/Gaia-WISE but not photo-z/position.
            cols = [c for c in X.columns if c not in POS_COLS+['M_pseudo']]
            yt5 = Y5[y[tr]]; cnt = np.bincount(yt5, minlength=5)
            ad = fit(X.iloc[tr][cols], yt5, X.iloc[es][cols], Y5[y[es]], TOP, a, ((len(tr)/(5*cnt))**.8)[yt5])
            energy = lambda xx: -logsumexp(ad.predict(xx[cols], raw_score=True, num_threads=a.threads), axis=1)
            ec, et, en = energy(X.iloc[cal]), energy(X.iloc[te]), energy(novel_X[h])
            # Equal availability for delayed-vs-immediate comparison is retained in CSV.
            for pop, ids, families, e, s in [('known',df.obj_id.iloc[te].values,df.cls.iloc[te].values,et,st),
                                             ('novel',novel.obj_id.values,novel.bts_type.values,en,sn)]:
                ep, rp = pvalue(ec,e), pvalue(sc,s)
                combined = np.minimum(1., 2*np.minimum(ep,rp))
                for oid,fam,pe,pr,co in zip(ids,families,ep,rp,combined):
                    ad_rows.append(dict(obj_id=oid, family=fam, population=pop, h=h, fold=fi,
                                        energy_p=pe, residual_p=pr, combined_p=co, residual_available=bool(np.isfinite(pr))))
            # Checkpoint after each expensive fold/horizon; files can be inspected live.
            pd.DataFrame(results).to_csv(a.out/'metrics.csv', index=False)
            pd.DataFrame(cls_rows).to_csv(a.out/'classification_oof.csv', index=False)
            pd.DataFrame(pe_rows).to_csv(a.out/'pe_oof.csv', index=False)
            pd.DataFrame(ad_rows).to_csv(a.out/'ad_oof.csv', index=False)
            print(f'completed fold={fi} horizon={h}', flush=True)
    (a.out/'complete.json').write_text(json.dumps(json_safe(provenance), indent=2))


if __name__ == '__main__':
    main()
