"""Registered object-disjoint forecast evaluation, 2026-09-17 UTC.

Freeze 200 BTS snapshot objects with type '-', ZTF25/26 identifiers, absent
from ALL local event files and feature/split IDs. Sort by SHA256 of
'flare-external-20260917:'+object_id; choose first200, no replacements based
on retrieval/quality/errors. Newly downloaded light curves have not been used
to fit, calibrate, choose or evaluate this forecast recipe. Catalogue-level
unclassified-population summaries were inspected before; no claim of a blind
catalogue. This is an unlabeled population and cannot validate classification
or rare-class forecast performance. It may contain out-of-distribution objects.

Fixed shipped forecast_blend checkpoints; compare 50/50 mixture to its own
grid component trained on exactly the same original7347 objects. No refitting,
recalibration, context lookup, hyperparameter changes or weight selection.
Color channels are recomputed from each observed prefix; cached colors from
full light curves must not enter early features. Training used historical
cached colors, so this additionally tests the corrected inference distribution.
ALL external context absent in BOTH methods, deliberately specified before
any new photometry is fetched. No spectroscopic redshift or class input.
Both horizons10,20; identical anchored real detections h<t<=90. Raw ALeRCE
magpsf is the target. Preserve raw MJD and JSON, unlike older relative arrays.

Eligibility: valid finite MJD, magpsf, positive sigmapsf, fid1/2/3; first MJD
>=2025-01-01 and <=snapshot UTC MJD-90; full100-day curve >=8 detections with
>=2 in g and r. No extra early-cadence cut. A future point needs an observed
anchor in its band. Require >=100 eligible objects per horizon before a
confirmation verdict. Report all exclusion/fetch failures and cadence counts.
Primary: median over objects of mean absolute actual-detection error. X1
relative reduction>=5% at h10 and>=3% at h20, paired2000-object-bootstrap95
lower>0 at both. X2 point-median error cost<=.01mag at both. X3 the200 fixed
IDs and model hashes must match registration, no overlap with local objects,
and >=100 eligible per horizon. Confirmation ALL, ONLY for this missing-context
unclassified population. Missing labels preclude a rare-class safeguard.
Nominal68/95 residual-band coverage is descriptive, no calibration updates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG))
from flare.analysis import feature_frame, observed_events
from flare.fetch import to_events
from flare.forecast_blend import BlendForecaster


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_rows(payload):
    rows=[]
    for d in (payload.get('detections') or []) if isinstance(payload,dict) else []:
        try:
            row=[float(d[k]) for k in ['mjd','fid','magpsf','sigmapsf']]
        except (TypeError,ValueError,KeyError):continue
        if np.isfinite(row).all() and row[1] in (1,2,3) and row[3]>0:rows.append(row)
    return np.array(sorted(rows),dtype=float).reshape(-1,4)


def eligible(rows,snapshot_mjd):
    if not len(rows):return 'no_valid_detections'
    start=rows[:,0].min()
    if start<pd.Timestamp('2025-01-01').to_julian_date()-2400000.5:return 'first_detection_before_2025'
    if start>snapshot_mjd-90:return 'less_than_90_days_since_first_detection'
    v=rows[rows[:,0]<=start+100]
    if len(v)<8 or (v[:,1]==1).sum()<2 or (v[:,1]==2).sum()<2:return 'fails_original_quality'
    return 'eligible'


def freeze(a):
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'raw').mkdir()
    cat=pd.read_csv(a.catalog)
    seen={p.stem for p in (a.data/'data/events').glob('*.npz')}
    seen.update(pd.read_parquet(a.data/'data/features_all.parquet',columns=['obj_id']).obj_id)
    for path in (a.data/'data/splits').glob('*.csv'):
        seen.update(pd.read_csv(path).obj_id)
    choices=cat[(cat.type=='-')&cat.ZTFID.str.match(r'^ZTF2[56][a-z]{7}$')&~cat.ZTFID.isin(seen)].copy()
    choices['selection_hash']=choices.ZTFID.map(lambda s:hashlib.sha256(('flare-external-20260917:'+s).encode()).hexdigest())
    selected=choices.sort_values('selection_hash').head(200)
    if len(selected)!=200:raise ValueError('Fewer than200 eligible catalogue candidates')
    selected.to_csv(a.out/'candidates.csv',index=False)
    pd.Series(sorted(seen),name='obj_id').to_csv(a.out/'excluded_local_ids.csv',index=False)
    import shutil
    shutil.copy2(a.catalog,a.out/'catalog.csv')
    now=datetime.now(timezone.utc)
    modeldir=PKG/'models/bts6/forecast_blend'
    manifest=json.loads((modeldir/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha(modeldir/name)!=expected:raise ValueError(f'Model checksum mismatch {name}')
    registration=dict(script_sha256=sha(__file__),created_utc=now.isoformat(),
        snapshot_mjd=float(pd.Timestamp(now).to_julian_date()-2400000.5),
        catalog_sha256=sha(a.out/'catalog.csv'),candidates_sha256=sha(a.out/'candidates.csv'),
        excluded_ids_sha256=sha(a.out/'excluded_local_ids.csv'),candidate_pool=len(choices),selected=200,
        model_dir=str(modeldir),model_manifest_sha256=sha(modeldir/'manifest.json'),
        model_hashes=manifest['files'],catalog_url='https://sites.astro.caltech.edu/ztf/bts/explorer.php?f=s&format=csv',
        inference_sha256={str(PKG/'flare'/name):sha(PKG/'flare'/name) for name in ['analysis.py','forecast_blend.py','fetch.py','features.py','data.py']},
        photometry_url='https://api.alerce.online/ztf/v1/objects/{obj_id}/lightcurve',
        scope='Unclassified objects, all external context absent; no classification or rare-class validation')
    (a.out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    config=['silent','show-error','location','fail','connect-timeout = 10','max-time = 30',
            'write-out = "%{url_effective} %{http_code} %{exitcode}\\n"']
    for oid in selected.ZTFID:
        config.extend([f'url = "https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve"',
                       f'output = "{a.out}/raw/{oid}.json"'])
    (a.out/'downloads.curl').write_text('\n'.join(config)+'\n')
    print(json.dumps(registration,indent=2))


def evaluate(a):
    root=a.out;reg=json.loads((root/'registration.json').read_text())
    for name,key in [('candidates.csv','candidates_sha256'),('excluded_local_ids.csv','excluded_ids_sha256'),('catalog.csv','catalog_sha256')]:
        if sha(root/name)!=reg[key]:raise ValueError(f'Frozen input changed: {name}')
    if sha(__file__)!=reg['script_sha256']:raise ValueError('Registered script changed')
    for path,expected in reg['inference_sha256'].items():
        if sha(path)!=expected:raise ValueError(f'Inference implementation changed: {path}')
    modeldir=Path(reg['model_dir'])
    if sha(modeldir/'manifest.json')!=reg['model_manifest_sha256']:raise ValueError('Models changed')
    model=BlendForecaster(modeldir,num_threads=2)
    for name,expected in reg['model_hashes'].items():
        if sha(modeldir/name)!=expected:raise ValueError(f'Model changed: {name}')
    selected=pd.read_csv(root/'candidates.csv');seen=set(pd.read_csv(root/'excluded_local_ids.csv').obj_id)
    assert len(selected)==200 and selected.ZTFID.is_unique and not set(selected.ZTFID)&seen
    records=[];audit=[];raw_hashes={}
    import lightgbm as lgb
    baselines={h:lgb.Booster(model_file=str(modeldir/f'h{h}_baseline.txt')) for h in [10,20]}
    for oid in selected.ZTFID:
        path=root/'raw'/f'{oid}.json'
        info=dict(obj_id=oid)
        if not path.exists() or path.stat().st_size==0:
            audit.append(dict(info,status='download_missing'));continue
        raw_hashes[oid]=sha(path)
        try:payload=json.loads(path.read_text())
        except json.JSONDecodeError:
            audit.append(dict(info,status='invalid_json'));continue
        rows=valid_rows(payload);status=eligible(rows,reg['snapshot_mjd'])
        info.update(status=status,valid_detections=len(rows))
        if len(rows):info.update(first_mjd=float(rows[:,0].min()),last_mjd=float(rows[:,0].max()))
        if status!='eligible':audit.append(info);continue
        events=to_events(rows.tolist(),horizon_days=100)
        # Future labels are native observed magnitudes, not fitted grid values.
        relative=rows[:,0]-rows[:,0].min()
        for h in [10,20]:
            arr=observed_events(events,h);frame,peak=feature_frame(arr)
            meta=json.loads((modeldir/f'h{h}.json').read_text())
            frame=frame.reindex(columns=meta['feature_names'])
            test=(relative>h)&(relative<=90)&np.isin(rows[:,1],[1,2])
            anchor_bands=set(arr[:,2].astype(int))
            test &= np.isin(rows[:,1]-1,list(anchor_bands))
            target_rows=rows[test];days=relative[test]
            info[f'h{h}_points']=len(days)
            if not len(days):continue
            times=np.unique(days)
            output=model.predict(arr,frame,peak,h,days=times)
            grid=np.array(meta['grid_days'],float)
            for b,label in enumerate(['g','r']):
                mask=target_rows[:,1]==b+1
                if not mask.any():continue
                band=output['bands'][label]
                indices=np.searchsorted(times,days[mask])+1
                blend=np.array(band['magnitude'])[indices]
                xx=pd.concat([frame.assign(fc_day=d,fc_band=float(b)) for d in grid],ignore_index=True)
                grid_prediction=baselines[h].predict(xx,num_threads=2)+peak
                baseline=np.interp(days[mask],np.r_[h,grid],np.r_[band['magnitude'][0],grid_prediction])
                truth=target_rows[mask,2]
                for method,pred in [('baseline',baseline),('blend',blend)]:
                    values=dict(obj_id=oid,h=h,method=method,day=days[mask],band=b,truth=truth,prediction=pred,error=abs(pred-truth))
                    if method=='blend':
                        values['covered68']=abs(pred-truth)<=meta['residual_halfwidth']['0.68']
                        values['covered95']=abs(pred-truth)<=meta['residual_halfwidth']['0.95']
                    records.append(pd.DataFrame(values))
        audit.append(info)
    pd.DataFrame(audit).to_csv(root/'eligibility.csv',index=False)
    if records:pd.concat(records,ignore_index=True).to_parquet(root/'points.parquet',index=False)
    (root/'raw_sha256.json').write_text(json.dumps(raw_hashes,indent=2)+'\n')
    (root/'complete.json').write_text(json.dumps(dict(script_sha256=sha(__file__),downloaded=len(raw_hashes),objects_evaluated=sum(r['status']=='eligible' for r in audit)),indent=2)+'\n')
    print(pd.DataFrame(audit).status.value_counts().to_string())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','evaluate'])
    p.add_argument('--catalog',type=Path);p.add_argument('--data',type=Path,default=PKG.parent/'bts_dataset')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.stage=='freeze':freeze(a)
    else:evaluate(a)


if __name__=='__main__':main()
