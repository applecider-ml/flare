"""Audit and report the fixed external unclassified forecast experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_forecast import sha,valid_rows,to_events,observed_events,feature_frame,BlendForecaster
from summarize_supervised import forecast_interval


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    complete=json.loads((root/'complete.json').read_text());reg=json.loads((root/'registration.json').read_text())
    assert complete['script_sha256']==reg['script_sha256']==sha(Path(__file__).with_name('external_forecast.py'))
    for name,key in [('candidates.csv','candidates_sha256'),('excluded_local_ids.csv','excluded_ids_sha256'),('catalog.csv','catalog_sha256')]:
        assert sha(root/name)==reg[key]
    selected=pd.read_csv(root/'candidates.csv');excluded=set(pd.read_csv(root/'excluded_local_ids.csv').obj_id)
    assert len(selected)==200 and selected.ZTFID.is_unique and not set(selected.ZTFID)&excluded
    for oid,expected in json.loads((root/'raw_sha256.json').read_text()).items():assert sha(root/'raw'/f'{oid}.json')==expected
    for path,expected in reg['inference_sha256'].items():assert sha(path)==expected
    for name,expected in reg['model_hashes'].items():assert sha(Path(reg['model_dir'])/name)==expected
    points=pd.read_parquet(root/'points.parquet');eligibility=pd.read_csv(root/'eligibility.csv')
    assert set(eligibility.obj_id)==set(selected.ZTFID) and eligibility.obj_id.is_unique
    assert set(points.obj_id)<=set(eligibility.loc[eligibility.status=='eligible','obj_id'])
    assert np.isfinite(points[['day','truth','prediction','error']]).all().all()
    np.testing.assert_allclose(points.error,abs(points.prediction-points.truth))
    objects=points.groupby(['h','method','obj_id'],as_index=False).error.mean()
    rows=[];comparisons={};primary=[];point_checks=[];size_checks=[]
    for h in [10,20]:
        d=points[points.h==h];o=objects[objects.h==h]
        tables={m:d[d.method==m].reset_index(drop=True) for m in ['baseline','blend']}
        pd.testing.assert_frame_equal(tables['baseline'][['obj_id','day','band','truth']],tables['blend'][['obj_id','day','band','truth']])
        assert (d.day>h).all() and (d.day<=90).all()
        wide=o.pivot(index='obj_id',columns='method',values='error').sort_index();assert not wide.isna().any().any()
        gain=float(wide.baseline.median()-wide.blend.median());relative=gain/float(wide.baseline.median())
        ci=forecast_interval(wide.baseline,wide.blend)
        comparisons[str(h)]=dict(objects=len(wide),gain=gain,relative_gain=relative,paired_95ci=ci)
        primary.append(relative>=(.05 if h==10 else .03) and ci[0]>0);size_checks.append(len(wide)>=100)
        point_checks.append(float(tables['blend'].error.median()-tables['baseline'].error.median())<=.01)
        for method in ['baseline','blend']:
            v=o[o.method==method];p=tables[method]
            rows.append(dict(h=h,method=method,objects=len(v),points=len(p),median_object_mae=float(v.error.median()),
                mean_object_mae=float(v.error.mean()),point_median_error=float(p.error.median()),
                coverage68=float(p.covered68.mean()) if method=='blend' else None,
                coverage95=float(p.covered95.mean()) if method=='blend' else None))
    gates=dict(X1=bool(all(primary)),X2=bool(all(point_checks)),X3=bool(all(size_checks)))
    summary=dict(confirmed=all(gates.values()),hypotheses=gates,comparisons=comparisons,
        exclusions={str(k):int(v) for k,v in eligibility.status.value_counts().items()},
        selected=200,scope=reg['scope'])
    table=pd.DataFrame(rows);table.to_csv(root/'metrics.csv',index=False)
    objects.to_parquet(root/'object_errors.parquet',index=False)
    # Real-object boundary test rebuilds the entire event/color cache after changing future magnitudes.
    model=BlendForecaster(reg['model_dir'],num_threads=2);checks=[]
    for oid in eligibility.loc[eligibility.status=='eligible','obj_id'].head(4):
        raw=valid_rows(json.loads((root/'raw'/f'{oid}.json').read_text()))
        for h in [10,20]:
            changed=raw.copy();changed[changed[:,0]-changed[:,0].min()>h,2]+=3
            predictions=[]
            for r in [raw,changed]:
                arr=observed_events(to_events(r.tolist()),h);frame,peak=feature_frame(arr)
                predictions.append(model.predict(arr,frame,peak,h,days=[h+.5,50.,90.]))
            for band in ['g','r']:
                if predictions[0]['bands'][band]['status']=='ok':
                    np.testing.assert_array_equal(predictions[0]['bands'][band]['magnitude'],predictions[1]['bands'][band]['magnitude'])
            checks.append(dict(obj_id=oid,h=h,future_cache_mutation_invariant=True))
    verification=dict(object_disjoint=True,frozen_models_and_candidates_verified=True,raw_hashes_verified=True,
                      common_future_targets=True,finite_predictions=True,future_boundary_checks=checks)
    (root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Frozen external forecast evaluation','',
        '**Object-disjoint unclassified2025/2026 light curves; all external context absent; observed-prefix colors recomputed.**','',
        'The sample was fixed before retrieval and scoring:200 objects, deterministic hash order, no replacements. '
        'No model fitting, mixture search, context lookup or interval recalibration. Baseline and blend use '
        'the same original7347 training objects. The comparison is against the bundle’s paired grid component, '
        'not a newly fitted baseline or the older production checkpoint trained on more objects.','',
        '| Horizon | Method | Objects | Median object MAE | Point-median error | Coverage68 | Coverage95 |','|---|---|---|---|---|---|---|']
    for _,r in table.iterrows():
        coverage=[f'{r[k]:.3f}' if pd.notna(r[k]) else '—' for k in ['coverage68','coverage95']]
        lines.append(f'| {int(r.h)} | {r.method} | {int(r.objects)} | {r.median_object_mae:.4f} | {r.point_median_error:.4f} | {coverage[0]} | {coverage[1]} |')
    for h,r in comparisons.items():lines+=['',f'Day{h}: reduction {r["relative_gain"]:.1%}; gain {r["gain"]:.4f}mag, paired95% interval [{r["paired_95ci"][0]:.4f}, {r["paired_95ci"][1]:.4f}].']
    lines+=['',f'**Registered confirmation criterion: {"MET" if summary["confirmed"] else "NOT MET"}.** '+str(gates),'',
        'X1: >=5%/3% improvements at day10/20 and positive bootstrap lower bounds at BOTH. X2: point-median cost<=0.01mag. X3: frozen objects/models, no local overlap and>=100 eligible evaluation objects per horizon.','',
        'Exclusions: '+str(summary['exclusions'])+'. Eligible full curves can still have no anchored future points at a particular horizon.','',
        '## Scope and limitations','',
        '- No class labels: this does not validate classification, rare-class errors, or label-conditional coverage.',
        '- The cohort is unclassified and all external context is absent. It is a separate deployment condition from the context-rich known-class BTS benchmark.',
        '- These frozen models were trained with historical color caches. The new inference explicitly rebuilds colors from observed prefixes; no future counterpart contributes to a test feature.',
        '- Raw MJD, photometry JSON, candidate selection, every exclusion and source/model hashes are retained. Four real objects at both horizons pass a full-cache future-mutation test.',
        '- Coverage is empirical pointwise coverage with original fixed residual widths; no calibration updates, no simultaneous curve guarantee.',
        '- The paired2000-draw bootstrap is conditional on this sample and fixed models. Catalogue selection and absent context limit generalization.',
        '- Sources: [BTS catalogue](https://sites.astro.caltech.edu/ztf/bts/explorer_info.html), [ALeRCE public API](https://science.alerce.online/services/ztf-database-api/).']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
