"""Registered supervised improvements, 2026-09-16; development, not fresh test.

Two fixed primary experiments on the 10497 original known BTS objects. No novel
objects used. Compare against adopted recipes, not full-trained early inference.
No model/weight/score selection on outer folds. Source/input hashes retained.

CLASSIFICATION: reproduce config F hierarchy, 190 features, original five folds
(StratifiedKFold seed0), 1/8 training carve-out seed=fold, same TOP/BR hyperparams,
beta=.8 and branch threshold chosen on the carve-out only. Baseline: one copy,
independent external-block dropout .30 including stopping rows (historical rule).
PRIMARY multiview: four copies of fitting objects, original plus three independent
.40 block-dropout views. Expected overall dropout remains .30. Each copy has
1/4 object weight, min_child_samples multiplied by4 to preserve effective counts.
Same stopping objects/inputs and branch-selection rule as baseline. No new inputs.
Evaluate full context and all external blocks absent, same held-out objects.
C1 pooled six-class macro-F1 gain >=.005 and paired stratified 95% bootstrap
lower bound >0. C2 accuracy loss <=.002 and each class F1 loss <=.03. C3 fully
missing-context macro-F1 loss <=.01. ADOPTION C1&C2&C3. Fold mean/std also reported.

FORECAST: five paired folds seed42, original 76/12/12 fitting/stopping/calibration
inside each outer training fold, seed=fold. Horizons10 and20, predictions <=day90.
Baseline: adopted final_choice.json pooled L1 forecaster trained on interpolated
5-day-grid truths with fc_day/fc_band, scored using original horizon anchoring.
PRIMARY real_delta: train on actual future detections (<=90d), target change from
last visible band magnitude, predicting directly at the real timestamp. Add four
band-local inputs from observations <=h ONLY: last relative magnitude, age of last
point, last-five-point slope and scatter. Each training object has total weight1
(rescaled to mean row weight1); all its points stay in its fold. Stopping weights
follow the same rule. Fixed grid_delta control uses the same recency inputs/target
but grid labels and uniform row weights, isolating the real-detection change.
All models use the adopted tuned parameters, max4000 rounds, patience100. No tuning.
Same eligible real detections (finite truth, band last magnitude, h<t<=90) for all.
Primary metric: median over objects of mean absolute real-detection error (mag).
F1 real_delta reduces primary error >=5% at h10 and >=3% at h20; paired object
bootstrap gain lower >0 at BOTH horizons. F2 median object MAE for SLSN and TDE
does not worsen >.05mag at either horizon. F3 global point-median error does not
worsen >.01mag at either horizon. ADOPTION F1&F2&F3. Report day50 grid error and
per-lead errors as secondary measures, never alternate adoption targets.
Plain 68% residual bands independently calibrated on real calibration points for
all methods, reported descriptively; not a claim to fix rare-class coverage.

2000 paired bootstrap draws over unique objects (classification stratifies by
true class). Fixed primary candidates; controls cannot rescue failed primaries.
Models/checkpoints are research artifacts; no production default overwritten.
Outputs stage on local scratch to avoid known project inode-quota interruptions.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold,train_test_split

PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG))
from flare.config import BTS6,HOST_COLS,GW_COLS,POS_COLS
from flare.context import lumdist_mpc
from flare.data import load_events
from train_bts6 import TOP,BR

CLASSES=list(BTS6.classes); Y5=np.array([0,1,1,2,3,4])
BLOCKS=[HOST_COLS,['M_pseudo'],GW_COLS,POS_COLS]
DAYS=[10,15,20,25,30,40,50,60,75,90]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def population(data):
    d=pd.concat([pd.read_csv(data/'data/splits'/f'{s}.csv').assign(split=s)
                 for s in ['train','val','test']],ignore_index=True)
    if not d.obj_id.is_unique: raise ValueError('duplicate objects')
    return d,np.array([CLASSES.index(c) for c in d.cls])


def classification_features(data,df):
    f=pd.read_parquet(data/'data/features_all.parquet').set_index('obj_id').reindex(df.obj_id)
    f=f.reset_index(drop=True)
    for name,cols in [('hosts.csv',HOST_COLS),('context_gaia_wise.csv',GW_COLS),('ps1_position.csv',POS_COLS)]:
        t=pd.read_csv(data/'data'/name).drop_duplicates('obj_id').set_index('obj_id').reindex(df.obj_id)
        for c in cols: f[c]=t[c].to_numpy()
    pz=pd.read_csv(data/'data/hosts_photoz.csv').drop_duplicates('obj_id').set_index('obj_id').reindex(df.obj_id)
    cat=pd.read_csv(data/'data/bts_catalog.csv').set_index('ZTFID').reindex(df.obj_id)
    f['M_pseudo']=pd.to_numeric(cat.peakmag,errors='coerce').to_numpy()-5*np.log10(lumdist_mpc(pz.z_phot)*1e5)
    # Match the historical config F column order (including feature subsampling).
    lc=[c for c in f if c not in sum(BLOCKS,[])]
    return f[lc+HOST_COLS+['M_pseudo']+GW_COLS+POS_COLS].replace([np.inf,-np.inf],np.nan)


def dropout(x,rng,rate):
    x=x.copy()
    for cols in BLOCKS: x.loc[rng.random(len(x))<rate,cols]=np.nan
    return x


def views(x,seed):
    rng=np.random.default_rng(seed)
    return pd.concat([x]+[dropout(x,rng,.4) for _ in range(3)],ignore_index=True)


def fit(x,y,e,ey,params,threads,weight=None,eweight=None,rounds=3000,patience=120):
    ok=np.isfinite(y);oke=np.isfinite(ey)
    p=dict(params,num_threads=threads,verbosity=-1)
    d=lgb.Dataset(x.loc[ok],label=y[ok],weight=None if weight is None else weight[ok])
    v=lgb.Dataset(e.loc[oke],label=ey[oke],weight=None if eweight is None else eweight[oke],reference=d)
    return lgb.train(p,d,rounds,valid_sets=[v],callbacks=[lgb.early_stopping(patience,verbose=False)])


def classify(top,branch,x,threshold,threads):
    p=top.predict(x,num_threads=threads);q=branch.predict(x,num_threads=threads)
    pred=np.array([0,1,3,4,5])[p.argmax(1)]
    pred[(p.argmax(1)==1)&(q>threshold)]=2
    return pred


def classification_fold(args,fold):
    data=Path(args['data']);out=Path(args['out'])/f'classification_fold{fold}';out.mkdir()
    df,y=population(data);x=classification_features(data,df)
    tri,te=list(StratifiedKFold(5,shuffle=True,random_state=0).split(x,y))[fold]
    itr,ies=train_test_split(np.arange(len(tri)),test_size=1/8,stratify=y[tri],random_state=fold)
    tr,es=tri[itr],tri[ies]
    np.savez_compressed(out/'partitions.npz',obj_id=df.obj_id.to_numpy(dtype=str),fit=tr,early_stop=es,test=te)
    # Historical dropout is drawn over the outer train before its carve-out.
    xd=dropout(x.iloc[tri].reset_index(drop=True),np.random.default_rng(fold),.3)
    xe=xd.iloc[ies]; ye=Y5[y[es]]; yt=Y5[y[tr]]
    count=np.bincount(yt,minlength=5);w=(len(tr)/(5*count))**.8
    records=[];metadata={}
    for variant in ['baseline','multiview']:
        copies=1 if variant=='baseline' else 4
        xt=xd.iloc[itr].reset_index(drop=True) if copies==1 else views(x.iloc[tr].reset_index(drop=True),1000+fold)
        yy=np.tile(y[tr],copies);ww=np.tile(w[yt]/copies,copies)
        top=fit(xt,Y5[yy],xe,ye,dict(TOP,min_child_samples=TOP['min_child_samples']*copies),args['threads'],ww)
        cc=np.isin(yy,[1,2]); ec=np.isin(y[es],[1,2]);yb=(yy==2).astype(int)
        base_cc=np.isin(y[tr],[1,2]);pos=(y[tr]==2).sum()
        wb=np.where(yb[cc]==1,(base_cc.sum()/max(pos,1))**.5,1.)/copies
        branch=fit(xt.loc[cc],yb[cc],xe.loc[ec],(y[es][ec]==2).astype(int),
            dict(BR,min_child_samples=BR['min_child_samples']*copies),args['threads'],wb)
        pb=branch.predict(xe.loc[ec],num_threads=args['threads'])
        thresholds=np.linspace(.05,.9,35)
        threshold=float(max(thresholds,key=lambda t:f1_score(y[es][ec]==2,pb>t,zero_division=0)))
        for context in ['available','absent']:
            test=x.iloc[te].copy()
            if context=='absent': test[sum(BLOCKS,[])]=np.nan
            pred=classify(top,branch,test,threshold,args['threads'])
            records.extend(dict(obj_id=df.obj_id[i],truth=int(y[i]),predicted=int(p),fold=fold,
                                method=variant,context=context) for i,p in zip(te,pred))
        top.save_model(str(out/f'{variant}_top.txt'));branch.save_model(str(out/f'{variant}_branch.txt'))
        metadata[variant]=dict(threshold=threshold,top_rounds=top.best_iteration,branch_rounds=branch.best_iteration)
    pd.DataFrame(records).to_csv(out/'predictions.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(models=metadata,feature_names=list(x.columns)),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(fold=fold)))
    return f'classification fold {fold} complete'


def recency(events,h):
    """Only <=h detections; no full-curve colors or future normalization."""
    a=np.asarray(events,dtype=float)
    a=a[(a[:,0]>=0)&(a[:,0]<=h)&np.isfinite(a[:,[0,2,3]]).all(1)]
    result=np.full((2,4),np.nan)
    if not len(a): return result
    peak=23.9-2.5*a[:,3].max()
    for b in [0,1]:
        v=a[a[:,2]==b];v=v[np.argsort(v[:,0],kind='stable')]
        if not len(v): continue
        # Historical anchor uses the first observation at an exact tied epoch.
        v=v[np.r_[True,np.diff(v[:,0])>0]]
        t=v[-5:,0];m=23.9-2.5*v[-5:,3]-peak
        slope=float(np.polyfit(t-t[-1],m,1)[0]) if len(np.unique(t))>=2 else np.nan
        result[b]=[m[-1],h-t[-1],slope,float(m.std())]
    return result


def grid_long(x,t,ids,h):
    days=[d for d in DAYS if d>h];rows=[];targets=[];objects=[];bands=[]
    for b in [0,1]:
        for day in days:
            rows.append(x.iloc[ids].assign(fc_day=float(day),fc_band=float(b)))
            targets.append(t[ids,b,day//5]);objects.append(ids);bands.append(np.full(len(ids),b))
    return pd.concat(rows,ignore_index=True),np.concatenate(targets),np.concatenate(objects),np.concatenate(bands)


def real_long(x,points,last,ids,h):
    objects=[];times=[];bands=[];targets=[]
    for i in ids:
        a=np.asarray(points[i],float).reshape(-1,3)
        if not len(a): continue
        b=a[:,1].astype(int)
        good=np.isfinite(a).all(1)&(a[:,0]>h)&(a[:,0]<=90)&np.isin(b,[0,1])
        good &= np.isfinite(last[i,np.clip(b,0,1)])
        a=a[good];b=b[good]
        objects.extend([i]*len(a));times.extend(a[:,0]);bands.extend(b);targets.extend(a[:,2])
    objects=np.array(objects,dtype=int);bands=np.array(bands,dtype=int)
    table=x.iloc[objects].reset_index(drop=True).assign(fc_day=times,fc_band=bands).astype(float)
    return table,np.array(targets),objects,bands


def add_recency(table,objects,bands,state):
    table=table.copy()
    for j,c in enumerate(['fc_last_mag','fc_age','fc_slope_last5','fc_scatter_last5']): table[c]=state[objects,bands,j]
    return table


def object_weights(objects):
    _,inverse,count=np.unique(objects,return_inverse=True,return_counts=True)
    w=1/count[inverse].astype(float)
    return w/w.mean()


def anchored(model,x,ids,h,points,last,threads):
    days=[d for d in DAYS if d>h]
    table,_,_,_=grid_long(x,np.zeros((len(x),2,21)),ids,h)
    p=model.predict(table,num_threads=threads).reshape(2,len(days),len(ids)).transpose(2,0,1)
    rows,truth,obj,band=real_long(x,points,last,ids,h);values=np.empty(len(obj))
    lookup={i:j for j,i in enumerate(ids)}
    for i in ids:
        for b in [0,1]:
            mask=(obj==i)&(band==b)
            values[mask]=np.interp(rows.fc_day.to_numpy()[mask],[h]+days,np.r_[last[i,b],p[lookup[i],b]])
    return rows,truth,obj,band,values


def forecast_fold(args,fold,h):
    data=Path(args['data']);root=Path(args['out']);out=root/f'forecast_h{h}_fold{fold}';out.mkdir()
    df,y=population(data);cache=data/'results/reconstruction/cache'
    x=pd.read_parquet(cache/f'X_pool_h{h}.parquet').replace([np.inf,-np.inf],np.nan)
    z=np.load(cache/f'TL_pool_h{h}.npz',allow_pickle=True);t=z['T'];last=z['L'];points=z['FP']
    state=np.load(root/'recency.npz')[f'h{h}']
    np.testing.assert_allclose(state[:,:,0],last,atol=2e-5,equal_nan=True)
    tri,te=list(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y))[fold]
    tr,rest=train_test_split(tri,test_size=.24,stratify=y[tri],random_state=fold)
    es,cal=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=fold)
    np.savez_compressed(out/'partitions.npz',obj_id=df.obj_id.to_numpy(dtype=str),fit=tr,early_stop=es,calibration=cal,test=te)
    params=json.loads((data/'results/reconstruction/final_choice.json').read_text())['forecaster']
    models={};details={}
    for name in ['baseline','grid_delta','real_delta']:
        if name=='real_delta':
            xt,yt,oi,bi=real_long(x,points,last,tr,h);xe,ye,oe,be=real_long(x,points,last,es,h)
            wt=object_weights(oi);we=object_weights(oe)
        else:
            xt,yt,oi,bi=grid_long(x,t,tr,h);xe,ye,oe,be=grid_long(x,t,es,h);wt=we=None
        if name!='baseline':
            xt=add_recency(xt,oi,bi,state);xe=add_recency(xe,oe,be,state)
            yt=yt-last[oi,bi];ye=ye-last[oe,be]
        start=time.monotonic()
        model=fit(xt,yt,xe,ye,params,args['threads'],wt,we,rounds=4000,patience=100)
        model.save_model(str(out/f'{name}.txt'));models[name]=model
        details[name]=dict(rounds=model.best_iteration,seconds=time.monotonic()-start,features=list(xt.columns))
        print(f'h{h} fold{fold} {name} rounds={model.best_iteration} seconds={details[name]["seconds"]:.1f}',flush=True)
        del xt,xe
    rows=[];grid_rows=[];calibration={}
    for split,ids in [('calibration',cal),('test',te)]:
        table,truth,oi,bi,base=anchored(models['baseline'],x,ids,h,points,last,args['threads'])
        extra=add_recency(table,oi,bi,state)
        predictions=dict(baseline=base,grid_delta=models['grid_delta'].predict(extra,num_threads=args['threads'])+last[oi,bi],
                         real_delta=models['real_delta'].predict(extra,num_threads=args['threads'])+last[oi,bi])
        for name,pred in predictions.items():
            error=np.abs(pred-truth)
            if split=='calibration':
                s=np.sort(error[np.isfinite(error)]);k=int(np.ceil((len(s)+1)*.68));calibration[name]=float(s[min(k,len(s))-1])
            else:
                rows.append(pd.DataFrame(dict(obj_id=df.obj_id.iloc[oi].to_numpy(),truth_class=y[oi],h=h,fold=fold,
                    method=name,day=table.fc_day,band=bi,truth=truth,prediction=pred,error=error,
                    covered=error<=calibration[name],halfwidth=calibration[name])))
        if split=='test':
            for b in [0,1]:
                g=x.iloc[ids].assign(fc_day=50.,fc_band=float(b));bs=np.full(len(ids),b)
                ge=add_recency(g,ids,bs,state)
                for name,m in models.items():
                    pred=m.predict(g if name=='baseline' else ge,num_threads=args['threads'])
                    if name!='baseline':pred+=last[ids,b]
                    grid_rows.append(pd.DataFrame(dict(obj_id=df.obj_id.iloc[ids].to_numpy(),truth_class=y[ids],h=h,
                        fold=fold,method=name,band=b,truth=t[ids,b,10],prediction=pred)))
    pd.concat(rows,ignore_index=True).to_parquet(out/'points.parquet',index=False)
    pd.concat(grid_rows,ignore_index=True).to_parquet(out/'day50.parquet',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(models=details,calibration=calibration),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(h=h,fold=fold)))
    return f'forecast h{h} fold{fold} complete'


def prepare_run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    df,_=population(a.data)
    files=[a.data/'data/splits'/f'{s}.csv' for s in ['train','val','test']]
    files += [a.data/'data'/s for s in ['features_all.parquet','hosts.csv','hosts_photoz.csv','bts_catalog.csv','context_gaia_wise.csv','ps1_position.csv']]
    files += [a.data/'results/reconstruction/final_choice.json']
    for h in [10,20]:
        files.extend([a.data/f'results/reconstruction/cache/{name}_pool_h{h}.{ext}' for name,ext in [('X','parquet'),('TL','npz')]])
    provenance=dict(script_sha256=sha(__file__),input_sha256={str(p):sha(p) for p in files},
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},classes=CLASSES,
        note='Development on previously inspected original known-class BTS population; no novel examples.')
    state={f'h{h}':[] for h in [10,20]};hashes={}
    for i,p in enumerate(df.filepath):
        path=a.data/'data'/p;hashes[p]=sha(path);events=load_events(path)
        for h in [10,20]:state[f'h{h}'].append(recency(events,h))
        if (i+1)%2500==0:print(f'prepared {i+1} early states',flush=True)
    np.savez_compressed(a.out/'recency.npz',obj_id=df.obj_id.to_numpy(dtype=str),**{k:np.stack(v) for k,v in state.items()})
    for h in [10,20]:
        cached=np.load(a.data/f'results/reconstruction/cache/TL_pool_h{h}.npz',allow_pickle=True)
        np.testing.assert_allclose(np.stack(state[f'h{h}'])[:,:,0],cached['L'],atol=2e-5,equal_nan=True)
    provenance['event_sha256']=hashes
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    df[['obj_id','cls','split']].to_csv(a.out/'objects.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=PKG.parent/'bts_dataset');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();prepare_run(a);args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(classification_fold,args,f) for f in range(5)]
        futures += [pool.submit(forecast_fold,args,f,h) for h in [10,20] for f in range(5)]
        for f in as_completed(futures): print(f.result(),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(classification_folds=5,forecast_folds=5,
        horizons=[10,20],script_sha256=sha(__file__)),indent=2))


if __name__=='__main__':main()
