"""Registered follow-up after real_delta's day10 SLSN regression, 2026-09-16.

The original primary remains failed if any gate fails; this is a NEW development
experiment, not a rescue of its registered verdict. Motivation observed before
this registration: day10 real_delta reduces median object MAE 0.2602->0.2277 but
SLSN worsens 0.4488->0.5260. No blend test predictions have been evaluated yet.

Fixed candidate: arithmetic 50/50 mean of adopted-grid and real_delta forecasts.
No class routing or weight search. New 5-fold object partitions seed2026; inner
76/12/12 fit/stop/cal seed100+fold. SAME previously inspected 10497 objects: new
partitions are NOT a fresh population or independent external confirmation.
Both component models retrain in each fold using unchanged registered helpers
in improve_supervised.py. Horizons10,20. No classification changes or novel data.

B1: median object MAE reduction vs paired adopted-grid >=5% at h10 and >=3% at
h20, positive paired 95% object-bootstrap lower bounds at BOTH (2000 draws).
B2: SLSN/TDE median object MAE deterioration <=.05mag at either horizon.
B3: global point-median error deterioration <=.01mag at either horizon.
ADOPTION B1&B2&B3. Unblended real_delta is a control, never a fallback candidate.
Report real points, per-class, lead bins, coverage and day50 grid metrics.
Separate final train-only fitting/calibration is required before optional use.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split

import improve_supervised as api


def fold_job(args,fold,h):
    data=Path(args['data']);root=Path(args['out']);out=root/f'forecast_h{h}_fold{fold}';out.mkdir()
    df,y=api.population(data);cache=data/'results/reconstruction/cache'
    x=pd.read_parquet(cache/f'X_pool_h{h}.parquet').replace([np.inf,-np.inf],np.nan)
    z=np.load(cache/f'TL_pool_h{h}.npz',allow_pickle=True);truth=z['T'];last=z['L'];points=z['FP']
    state=np.load(Path(args['source'])/'recency.npz')[f'h{h}']
    np.testing.assert_allclose(state[:,:,0],last,atol=2e-5,equal_nan=True)
    tri,te=list(StratifiedKFold(5,shuffle=True,random_state=2026).split(x,y))[fold]
    tr,rest=train_test_split(tri,test_size=.24,stratify=y[tri],random_state=100+fold)
    es,cal=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=100+fold)
    np.savez_compressed(out/'partitions.npz',obj_id=df.obj_id.to_numpy(dtype=str),fit=tr,early_stop=es,calibration=cal,test=te)
    params=json.loads((data/'results/reconstruction/final_choice.json').read_text())['forecaster']
    models={};details={}
    for name in ['baseline','real_delta']:
        if name=='baseline':
            xt,yt,oi,bi=api.grid_long(x,truth,tr,h);xe,ye,oe,be=api.grid_long(x,truth,es,h);wt=we=None
        else:
            xt,yt,oi,bi=api.real_long(x,points,last,tr,h);xe,ye,oe,be=api.real_long(x,points,last,es,h)
            xt=api.add_recency(xt,oi,bi,state);xe=api.add_recency(xe,oe,be,state)
            yt=yt-last[oi,bi];ye=ye-last[oe,be];wt=api.object_weights(oi);we=api.object_weights(oe)
        start=time.monotonic()
        m=api.fit(xt,yt,xe,ye,params,args['threads'],wt,we,rounds=4000,patience=100)
        models[name]=m;m.save_model(str(out/f'{name}.txt'))
        details[name]=dict(rounds=m.best_iteration,features=list(xt.columns))
        print(f'blend h{h} fold{fold} {name}: {m.best_iteration} rounds, {time.monotonic()-start:.1f}s',flush=True)
        del xt,xe
    records=[];grid_records=[];calibration={}
    for split,ids in [('calibration',cal),('test',te)]:
        table,target,oi,bi,baseline=api.anchored(models['baseline'],x,ids,h,points,last,args['threads'])
        real=models['real_delta'].predict(api.add_recency(table,oi,bi,state),num_threads=args['threads'])+last[oi,bi]
        predictions=dict(baseline=baseline,real_delta=real,blend=(baseline+real)/2)
        for name,p in predictions.items():
            error=np.abs(p-target)
            if split=='calibration':
                s=np.sort(error[np.isfinite(error)]);k=int(np.ceil((len(s)+1)*.68));calibration[name]=float(s[min(k,len(s))-1])
            else:
                records.append(pd.DataFrame(dict(obj_id=df.obj_id.iloc[oi].to_numpy(),truth_class=y[oi],h=h,fold=fold,
                    method=name,day=table.fc_day,band=bi,truth=target,prediction=p,error=error,
                    covered=error<=calibration[name],halfwidth=calibration[name])))
        if split=='test':
            for b in [0,1]:
                g=x.iloc[ids].assign(fc_day=50.,fc_band=float(b));bands=np.full(len(ids),b)
                base=models['baseline'].predict(g,num_threads=args['threads'])
                real=models['real_delta'].predict(api.add_recency(g,ids,bands,state),num_threads=args['threads'])+last[ids,b]
                for name,pred in dict(baseline=base,real_delta=real,blend=(base+real)/2).items():
                    grid_records.append(pd.DataFrame(dict(obj_id=df.obj_id.iloc[ids].to_numpy(),truth_class=y[ids],h=h,
                        fold=fold,method=name,band=b,truth=truth[ids,b,10],prediction=pred)))
    pd.concat(records,ignore_index=True).to_parquet(out/'points.parquet',index=False)
    pd.concat(grid_records,ignore_index=True).to_parquet(out/'day50.parquet',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(models=details,blend_weight=.5,calibration=calibration),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(h=h,fold=fold)))
    return f'blend h{h} fold{fold} complete'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=api.PKG.parent/'bts_dataset')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((a.source/'provenance.json').read_text())
    assert api.sha(api.__file__)==parent['script_sha256']
    provenance=dict(script_sha256=api.sha(__file__),helper_sha256=api.sha(api.__file__),
        parent_provenance_sha256=api.sha(a.source/'provenance.json'),recency_sha256=api.sha(a.source/'recency.npz'),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        note='Follow-up motivated by day10 SLSN regression; same inspected population, new partitions, fixed .5 blend.')
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(fold_job,provenance['arguments'],f,h) for h in [10,20] for f in range(5)]
        for f in as_completed(futures):print(f.result(),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(folds=5,horizons=[10,20],script_sha256=api.sha(__file__)),indent=2))


if __name__=='__main__':main()
