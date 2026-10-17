"""Registered follow-up: fixed classification ensemble and dated forecast stress test.

E1: 3-member hierarchy ensemble (existing seed42 + new seeds17,83), averaging
top probabilities and branch probabilities separately, then using the existing
hierarchical decoder. New members share original five outer folds and inner
fit/stop partitions. Each uses one .30 context-dropout view; seed-specific
fitting masks, but the original stopping mask is common. Same hyperparameters,
weights, patience and branch-threshold grid; threshold chosen on stopping CC
objects only. No ensemble size/seed/weight search. Baseline is the exact saved
paired refit from improve_supervised.py, verified by replay before scoring.
E1 requires pooled full-context macroF1 gain >=.005 and paired stratified 95%
bootstrap lower >0; E2 accuracy loss <=.002 and each class F1 loss <=.03;
E3 all-context-absent macroF1 loss <=.01. ADOPTION all three. No fallback decoder.

T: fixed existing 50/50 forecast blend, no retraining-rule changes. Two calendar
peak-date cohorts: 2024 and 2025. For each, only objects with catalog peak at
least100 days before Jan1 enter the historical pool; split76/12/12 fit/stop/cal
seed2026+year. Test peaks in that calendar year. Horizons10,20; grid and real
delta refit on history only with the already adopted parameters. Same real
future detections, equal-object error metric, plain68% calibration as before.
T1 pooled gain >=5%/3% at h10/h20 with positive paired bootstrap95 lower bounds.
T2 nonnegative primary gain in EACH year/horizon. T3 pooled SLSN/TDE median
object MAE cost <=.05mag each horizon. T4 pooled point-median cost <=.01mag.
All required for temporal robustness. Per-year rare counts/errors are reported.

These are SAME previously inspected objects, not an external test. Catalog
peakt is JD-2458000 (official BTS explorer documentation); it is not discovery
time. Absolute observation timestamps and historical context/label snapshots
are absent locally. The100-day peak-date buffer is a cohort separation rule,
NOT proof that all training measurements/labels were available at that date.
No new production weights/defaults are changed by this research script.
All intervals use2000 paired object draws conditional on fitted models.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import improve_supervised as api
from train_bts6 import TOP, BR


def decode(p, q, threshold):
    top = p.argmax(1)
    prediction = np.array([0, 1, 3, 4, 5])[top]
    prediction[(top == 1) & (q > threshold)] = 2
    return prediction


def chronology(peak, year):
    start = pd.Timestamp(f'{year}-01-01').to_julian_date() - 2458000
    end = pd.Timestamp(f'{year+1}-01-01').to_julian_date() - 2458000
    return np.flatnonzero(peak < start-100), np.flatnonzero((peak >= start) & (peak < end))


def classification_job(args, fold):
    data, root, source = (Path(args[k]) for k in ['data', 'out', 'source'])
    out = root / f'classification_fold{fold}'; out.mkdir()
    df, y = api.population(data); x = api.classification_features(data, df)
    parent = source / f'classification_fold{fold}'
    parts = np.load(parent / 'partitions.npz'); tr, es, te = (parts[k] for k in ['fit', 'early_stop', 'test'])
    np.savez_compressed(out/'partitions.npz', **{k: parts[k] for k in parts.files})
    tri = np.sort(np.r_[tr, es])
    original = api.dropout(x.iloc[tri].reset_index(drop=True), np.random.default_rng(fold), .3)
    xe = original.iloc[np.searchsorted(tri, es)]
    baseline_top = lgb.Booster(model_file=str(parent/'baseline_top.txt'))
    baseline_branch = lgb.Booster(model_file=str(parent/'baseline_branch.txt'))
    baseline_threshold = json.loads((parent/'metadata.json').read_text())['models']['baseline']['threshold']
    models = [(baseline_top, baseline_branch)]; details = {}
    yt = api.Y5[y[tr]]; count = np.bincount(yt, minlength=5)
    weights = (len(tr)/(5*count))**.8
    cc = np.isin(y[tr], [1,2]); ec = np.isin(y[es], [1,2])
    wb = np.where(y[tr][cc] == 2, (cc.sum()/(y[tr] == 2).sum())**.5, 1.)
    for seed in [17, 83]:
        xt = api.dropout(x.iloc[tr].reset_index(drop=True), np.random.default_rng(100*seed+fold), .3)
        top = api.fit(xt, yt, xe, api.Y5[y[es]], dict(TOP, seed=seed), args['threads'], weights[yt])
        branch = api.fit(xt.loc[cc], (y[tr][cc] == 2).astype(int), xe.loc[ec],
            (y[es][ec] == 2).astype(int), dict(BR, seed=seed), args['threads'], wb)
        top.save_model(str(out/f'seed{seed}_top.txt')); branch.save_model(str(out/f'seed{seed}_branch.txt'))
        models.append((top, branch)); details[str(seed)] = dict(top_rounds=top.best_iteration, branch_rounds=branch.best_iteration)
        print(f'classification fold{fold} seed{seed} fitted', flush=True)
    qes = np.mean([b.predict(xe.loc[ec], num_threads=args['threads']) for _, b in models], axis=0)
    threshold = float(max(np.linspace(.05,.9,35), key=lambda t:f1_score(y[es][ec] == 2, qes > t, zero_division=0)))
    records = []; probabilities = {}
    old = pd.read_csv(parent/'predictions.csv')
    for context in ['available', 'absent']:
        xx = x.iloc[te].copy()
        if context == 'absent': xx[sum(api.BLOCKS, [])] = np.nan
        pp = [m.predict(xx, num_threads=args['threads']) for m, _ in models]
        qq = [m.predict(xx, num_threads=args['threads']) for _, m in models]
        predictions = dict(baseline=decode(pp[0], qq[0], baseline_threshold),
                           ensemble=decode(np.mean(pp,axis=0), np.mean(qq,axis=0), threshold))
        expected = old[(old.method == 'baseline') & (old.context == context)].set_index('obj_id').loc[df.obj_id.iloc[te]]
        np.testing.assert_array_equal(predictions['baseline'], expected.predicted)
        probabilities[f'{context}_top'] = np.stack(pp)
        probabilities[f'{context}_branch'] = np.stack(qq)
        for method, pred in predictions.items():
            records.append(pd.DataFrame(dict(obj_id=df.obj_id.iloc[te].to_numpy(), truth=y[te],
                predicted=pred, fold=fold, context=context, method=method)))
    pd.concat(records,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    np.savez_compressed(out/'probabilities.npz', **probabilities)
    (out/'metadata.json').write_text(json.dumps(dict(seeds=[42,17,83], models=details,
        threshold=threshold, baseline_threshold=baseline_threshold, baseline_replay_passed=True,
        baseline_hashes={name:api.sha(parent/name) for name in ['baseline_top.txt','baseline_branch.txt','partitions.npz']}),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(fold=fold)))
    return f'classification fold{fold} complete'


def forecast_job(args, year, h):
    data, root, source = (Path(args[k]) for k in ['data','out','source'])
    out = root/f'forecast_y{year}_h{h}';out.mkdir()
    df,y = api.population(data);cache=data/'results/reconstruction/cache'
    catalog=pd.read_csv(data/'data/bts_catalog.csv').set_index('ZTFID')
    peak=pd.to_numeric(catalog.reindex(df.obj_id).peakt,errors='raise').to_numpy()
    if not np.isfinite(peak).all():raise ValueError('Missing catalog peak dates')
    history,te=chronology(peak,year)
    tr,rest=train_test_split(history,test_size=.24,stratify=y[history],random_state=2026+year)
    es,cal=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=2026+year)
    np.savez_compressed(out/'partitions.npz',obj_id=df.obj_id.to_numpy(dtype=str),fit=tr,early_stop=es,
                        calibration=cal,test=te,peak_jd_minus_2458000=peak)
    x=pd.read_parquet(cache/f'X_pool_h{h}.parquet').replace([np.inf,-np.inf],np.nan)
    z=np.load(cache/f'TL_pool_h{h}.npz',allow_pickle=True);last=z['L'];points=z['FP']
    state=np.load(source/'recency.npz')[f'h{h}']
    params=json.loads((data/'results/reconstruction/final_choice.json').read_text())['forecaster']
    models={};details={}
    for method in ['baseline','real_delta']:
        if method=='baseline':
            xt,yt,oi,bi=api.grid_long(x,z['T'],tr,h);xe,ye,oe,be=api.grid_long(x,z['T'],es,h);wt=we=None
        else:
            xt,yt,oi,bi=api.real_long(x,points,last,tr,h);xe,ye,oe,be=api.real_long(x,points,last,es,h)
            xt=api.add_recency(xt,oi,bi,state);xe=api.add_recency(xe,oe,be,state)
            yt=yt-last[oi,bi];ye=ye-last[oe,be];wt=api.object_weights(oi);we=api.object_weights(oe)
        model=api.fit(xt,yt,xe,ye,params,args['threads'],wt,we,rounds=4000,patience=100)
        model.save_model(str(out/f'{method}.txt'));models[method]=model
        details[method]=dict(rounds=model.best_iteration,features=list(xt.columns))
        print(f'temporal year{year} h{h} {method}: {model.best_iteration} rounds',flush=True)
        del xt,xe
    widths={};records=[]
    for split,ids in [('calibration',cal),('test',te)]:
        table,truth,oi,bi,pb=api.anchored(models['baseline'],x,ids,h,points,last,args['threads'])
        pr=models['real_delta'].predict(api.add_recency(table,oi,bi,state),num_threads=args['threads'])+last[oi,bi]
        for method,pred in dict(baseline=pb,real_delta=pr,blend=(pb+pr)/2).items():
            error=abs(pred-truth)
            if split=='calibration':
                ordered=np.sort(error);k=min(int(np.ceil((len(ordered)+1)*.68)),len(ordered))-1
                widths[method]=float(ordered[k])
            else:
                records.append(pd.DataFrame(dict(obj_id=df.obj_id.iloc[oi].to_numpy(),truth_class=y[oi],year=year,
                    h=h,method=method,day=table.fc_day,band=bi,truth=truth,prediction=pred,error=error,
                    halfwidth=widths[method],covered=error<=widths[method])))
    pd.concat(records,ignore_index=True).to_parquet(out/'points.parquet',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(models=details,calibration=widths,year=year,horizon=h),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(year=year,h=h)))
    return f'temporal year{year} h{h} complete'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=api.PKG.parent/'bts_dataset')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((a.source/'provenance.json').read_text())
    assert api.sha(api.__file__)==parent['script_sha256']
    inputs={}
    for name,expected in parent['input_sha256'].items():
        path=a.data/Path(name).relative_to(Path(parent['arguments']['data']))
        if api.sha(path)!=expected:raise ValueError(f'Parent input changed: {path}')
        inputs[str(path)]=expected
    arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    provenance=dict(script_sha256=api.sha(__file__),helper_sha256=api.sha(api.__file__),arguments=arguments,
        parent_provenance_sha256=api.sha(a.source/'provenance.json'),recency_sha256=api.sha(a.source/'recency.npz'),
        catalog_sha256=api.sha(a.data/'data/bts_catalog.csv'),input_sha256=inputs,
        date_reference='https://sites.astro.caltech.edu/ztf/bts/explorer_info.html',
        scope='Historical peak-date stress test on previously inspected population, not external/prospective validation')
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(forecast_job,arguments,year,h) for year in [2024,2025] for h in [10,20]]
        jobs += [pool.submit(classification_job,arguments,fold) for fold in range(5)]
        for job in as_completed(jobs):print(job.result(),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(script_sha256=api.sha(__file__),classification_folds=5,temporal_jobs=4)))


if __name__=='__main__':main()
