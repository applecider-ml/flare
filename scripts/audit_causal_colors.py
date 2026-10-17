"""Audit and rescore fixed OOF models after correcting prefix color leakage.

No refitting, tuning, interval recalibration or new model adoption. Recompute
only the color features of each prefix, retain all other cached features, and
rescore the already registered seed2026 five-fold fixed50/50 blend. Report the
original and corrected paired metrics, class errors and empirical coverage.
These models were TRAINED with historical colors; rescoring isolates corrected
test-time preprocessing, not fully causal retraining. Prior headline results
using the historical caches must be identified as such.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import improve_supervised as api
from flare.analysis import observed_events,recompute_observed_colors
from flare import data as D
from flare.features import _color_features
from summarize_supervised import forecast_interval


def future_partners(events,h):
    count=0
    for first,second in [(0,1),(1,2)]:
        for i in np.flatnonzero(events[:,0]<=h):
            band=events[i,2]
            if band not in (first,second):continue
            other=np.flatnonzero(events[:,2]==(second if band==first else first))
            if not len(other):continue
            j=other[np.argmin(abs(events[other,0]-events[i,0]))]
            if abs(events[j,0]-events[i,0])<=1.5 and events[j,0]>h:count+=1
    return count


def rescore(args,h,fold):
    root=Path(args['out']);source=Path(args['source']);data=Path(args['data'])
    df,y=api.population(data);x=pd.read_parquet(root/f'X_causal_h{h}.parquet')
    cached=np.load(data/f'results/reconstruction/cache/TL_pool_h{h}.npz',allow_pickle=True)
    state=np.load(Path(args['initial'])/'recency.npz')[f'h{h}']
    old=source/f'forecast_h{h}_fold{fold}';parts=np.load(old/'partitions.npz');ids=parts['test']
    base=lgb.Booster(model_file=str(old/'baseline.txt'));real=lgb.Booster(model_file=str(old/'real_delta.txt'))
    table,truth,oi,bi,pb=api.anchored(base,x,ids,h,cached['FP'],cached['L'],2)
    pr=real.predict(api.add_recency(table,oi,bi,state),num_threads=2)+cached['L'][oi,bi]
    widths=json.loads((old/'metadata.json').read_text())['calibration']
    result=[]
    for method,pred in [('baseline',pb),('blend',(pb+pr)/2)]:
        result.append(pd.DataFrame(dict(obj_id=df.obj_id.iloc[oi].to_numpy(),truth_class=y[oi],h=h,fold=fold,
            method=method,day=table.fc_day,band=bi,truth=truth,prediction=pred,error=abs(pred-truth),
            covered=abs(pred-truth)<=widths[method])))
    pd.concat(result,ignore_index=True).to_parquet(root/f'points_h{h}_fold{fold}.parquet',index=False)
    return f'corrected inference h{h} fold{fold} complete'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=api.PKG.parent/'bts_dataset')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    df,y=api.population(a.data);audit=[];tables={}
    for h in [10,20]:tables[h]=pd.read_parquet(a.data/f'results/reconstruction/cache/X_pool_h{h}.parquet').replace([np.inf,-np.inf],np.nan)
    for i,row in df.iterrows():
        events=D.load_events(a.data/'data'/row.filepath)
        for h in [10,20]:
            arr=observed_events(events,h);clean=recompute_observed_colors(arr)
            features=_color_features(D.color_series(clean))
            for key,value in features.items():tables[h].loc[i,key]=value
            audit.append(dict(obj_id=row.obj_id,cls=row.cls,h=h,future_color_partners=future_partners(events,h)))
        if (i+1)%2500==0:print(f'prefix color audit {i+1}/{len(df)}',flush=True)
    pd.DataFrame(audit).to_csv(a.out/'affected_objects.csv',index=False)
    for h,x in tables.items():x.to_parquet(a.out/f'X_causal_h{h}.parquet',index=False)
    args={k:str(v) for k,v in vars(a).items()}
    provenance=dict(script_sha256=api.sha(__file__),inference_sha256=api.sha(api.PKG/'flare/analysis.py'),arguments=args,
        model_sha256={str(path):api.sha(path) for path in sorted(a.source.glob('forecast_h*_fold*/*.txt'))})
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(rescore,args,h,f) for h in [10,20] for f in range(5)]
        for job in as_completed(jobs):print(job.result(),flush=True)
    points=pd.concat([pd.read_parquet(path) for path in a.out.glob('points_h*_fold*.parquet')],ignore_index=True)
    objects=points.groupby(['h','method','obj_id','truth_class'],as_index=False).error.mean()
    objects.to_parquet(a.out/'object_errors.parquet',index=False)
    rows=[];classes=[];comparisons={}
    for h in [10,20]:
        d=objects[objects.h==h];wide=d.pivot(index='obj_id',columns='method',values='error').sort_index()
        gain=float(wide.baseline.median()-wide.blend.median())
        comparisons[str(h)]=dict(gain=gain,relative_gain=gain/float(wide.baseline.median()),
                                 paired_95ci=forecast_interval(wide.baseline,wide.blend))
        for method in ['baseline','blend']:
            o=d[d.method==method];p=points[(points.h==h)&(points.method==method)]
            rows.append(dict(h=h,method=method,objects=len(o),median_object_mae=float(o.error.median()),
                             point_median_error=float(p.error.median()),coverage68=float(p.covered.mean())))
            for cls,g in o.groupby('truth_class'):
                classes.append(dict(h=h,method=method,cls=api.CLASSES[cls],objects=len(g),median_object_mae=float(g.error.median())))
    pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(a.out/'class_metrics.csv',index=False)
    affected=pd.DataFrame(audit).assign(affected=lambda d:d.future_color_partners>0).groupby('h').affected.sum().to_dict()
    summary=dict(affected_objects={str(k):int(v) for k,v in affected.items()},population=len(df),comparisons=comparisons,
        caveat='Frozen legacy-trained models, corrected test prefixes only. Historical cached-color metrics are not fully causal validation.')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.out/'complete.json').write_text(json.dumps(dict(script_sha256=api.sha(__file__))))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
