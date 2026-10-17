"""Summarize the preregistered ensemble and peak-date forecast follow-up."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import improve_supervised as api
from summarize_supervised import classification_interval, forecast_interval, metrics
from robustness_followup import chronology, decode


def classification(root):
    data=pd.concat([pd.read_csv(root/f'classification_fold{f}/predictions.csv') for f in range(5)],ignore_index=True)
    rows=[];contrasts={};confusions=[]
    for context in ['available','absent']:
        d=data[data.context==context]
        wide=d.pivot(index='obj_id',columns='method',values='predicted').sort_index()
        assert len(wide)==10497 and not wide.isna().any().any()
        truth=d.drop_duplicates('obj_id').set_index('obj_id').loc[wide.index,'truth'].to_numpy(int)
        result={m:metrics(truth,wide[m].to_numpy(int)) for m in ['baseline','ensemble']}
        ci=classification_interval(truth,wide.baseline.to_numpy(int),wide.ensemble.to_numpy(int))
        contrasts[context]=dict(gain=result['ensemble']['macro_f1']-result['baseline']['macro_f1'],paired_95ci=ci)
        for method,m in result.items():
            foldf1=[metrics(g.truth.to_numpy(),g.predicted.to_numpy())['macro_f1'] for _,g in d[d.method==method].groupby('fold')]
            rows.append(dict(context=context,method=method,macro_f1=m['macro_f1'],accuracy=m['accuracy'],
                fold_mean_f1=float(np.mean(foldf1)),fold_std_f1=float(np.std(foldf1)),
                **dict(zip(api.CLASSES,m['per_class']))))
            matrix=np.bincount(truth*6+wide[method].to_numpy(int),minlength=36).reshape(6,6)
            for i in range(6):
                for j in range(6):confusions.append(dict(context=context,method=method,truth=api.CLASSES[i],predicted=api.CLASSES[j],count=int(matrix[i,j])))
        if context=='available':
            accuracy=result['ensemble']['accuracy']-result['baseline']['accuracy']
            per=np.array(result['ensemble']['per_class'])-result['baseline']['per_class']
    verdict=dict(E1=bool(contrasts['available']['gain']>=.005 and contrasts['available']['paired_95ci'][0]>0),
                 E2=bool(accuracy>=-.002 and (per>=-.03).all()),E3=bool(contrasts['absent']['gain']>=-.01))
    table=pd.DataFrame(rows);table.to_csv(root/'classification_metrics.csv',index=False)
    pd.DataFrame(confusions).to_csv(root/'classification_confusions.csv',index=False)
    data.to_parquet(root/'classification_oof.parquet',index=False)
    return table,dict(hypotheses=verdict,adopt=all(verdict.values()),contrasts=contrasts,
                      accuracy_delta=accuracy,per_class_delta=dict(zip(api.CLASSES,per.tolist())))


def forecast(root):
    points=pd.concat([pd.read_parquet(root/f'forecast_y{year}_h{h}/points.parquet') for year in [2024,2025] for h in [10,20]],ignore_index=True)
    objects=points.groupby(['year','h','method','obj_id','truth_class'],as_index=False).error.mean()
    rows=[];classes=[];contrasts={};year_checks=[];rare_checks=[];point_checks=[];primary_checks=[]
    for cohort in ['pooled',2024,2025]:
        for h in [10,20]:
            o=objects[objects.h==h];p=points[points.h==h]
            if cohort!='pooled':o=o[o.year==cohort];p=p[p.year==cohort]
            wide=o.pivot(index='obj_id',columns='method',values='error').sort_index()
            assert not wide.isna().any().any()
            gain=float(wide.baseline.median()-wide.blend.median());relative=gain/float(wide.baseline.median())
            ci=forecast_interval(wide.baseline,wide.blend)
            contrasts[f'{cohort}_h{h}']=dict(objects=len(wide),gain=gain,relative_gain=relative,paired_95ci=ci)
            for method in ['baseline','real_delta','blend']:
                v=o[o.method==method];q=p[p.method==method]
                rows.append(dict(cohort=cohort,h=h,method=method,objects=len(v),points=len(q),
                    median_object_mae=float(v.error.median()),mean_object_mae=float(v.error.mean()),
                    point_median_error=float(q.error.median()),coverage68=float(q.covered.mean()),
                    mean_width68=float(2*q.halfwidth.mean())))
                for cls,g in v.groupby('truth_class'):
                    qp=q[q.truth_class==cls]
                    classes.append(dict(cohort=cohort,h=h,method=method,cls=api.CLASSES[cls],objects=len(g),
                        median_object_mae=float(g.error.median()),coverage68=float(qp.covered.mean())))
            if cohort=='pooled':
                primary_checks.append(relative>=(.05 if h==10 else .03) and ci[0]>0)
                for cls in [2,4]:
                    m=o[o.truth_class==cls].groupby('method').error.median()
                    rare_checks.append(float(m.blend-m.baseline)<=.05)
                m=p.groupby('method').error.median();point_checks.append(float(m.blend-m.baseline)<=.01)
            else:year_checks.append(gain>=0)
    verdict=dict(T1=bool(all(primary_checks)),T2=bool(all(year_checks)),T3=bool(all(rare_checks)),T4=bool(all(point_checks)))
    table=pd.DataFrame(rows);cl=pd.DataFrame(classes)
    table.to_csv(root/'forecast_metrics.csv',index=False);cl.to_csv(root/'forecast_class_metrics.csv',index=False)
    objects.to_parquet(root/'forecast_object_errors.parquet',index=False)
    points['lead_bin']=pd.cut(points.day-points.h,[0,5,15,30,60,100])
    points.groupby(['year','h','method','lead_bin'],observed=True).agg(points=('error','size'),median_error=('error','median'),coverage68=('covered','mean')).to_csv(root/'forecast_lead_metrics.csv')
    return table,cl,dict(hypotheses=verdict,robust=all(verdict.values()),contrasts=contrasts)


def verify(root):
    provenance=json.loads((root/'provenance.json').read_text())
    assert api.sha(Path(__file__).with_name('robustness_followup.py'))==provenance['script_sha256']
    assert api.sha(api.__file__)==provenance['helper_sha256']
    source=Path(provenance['arguments']['source']);data=Path(provenance['arguments']['data'])
    df,_=api.population(data);features=api.classification_features(data,df)
    seen=[]
    replays=[]
    for f in range(5):
        out=root/f'classification_fold{f}';p=np.load(out/'partitions.npz');seen.extend(p['test'])
        sets=[set(p[k]) for k in ['fit','early_stop','test']]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
        assert len(set.union(*sets))==10497
        meta=json.loads((out/'metadata.json').read_text());assert meta['baseline_replay_passed']
        probs=np.load(out/'probabilities.npz');pred=pd.read_csv(out/'predictions.csv')
        for context in ['available','absent']:
            expected=decode(probs[f'{context}_top'].mean(0),probs[f'{context}_branch'].mean(0),meta['threshold'])
            np.testing.assert_array_equal(expected,pred[(pred.method=='ensemble')&(pred.context==context)].predicted)
            xx=features.iloc[p['test'][:20]].copy()
            if context=='absent':xx[sum(api.BLOCKS,[])]=np.nan
            for member,seed in enumerate([42,17,83]):
                directory=source/f'classification_fold{f}' if seed==42 else out
                prefix='baseline' if seed==42 else f'seed{seed}'
                for role in ['top','branch']:
                    model=lgb.Booster(model_file=str(directory/f'{prefix}_{role}.txt'))
                    np.testing.assert_allclose(model.predict(xx,num_threads=2),probs[f'{context}_{role}'][member,:20],rtol=1e-12,atol=1e-12)
            replays.append(dict(task='classification',fold=f,context=context,objects=20,members=3))
    assert sorted(seen)==list(range(10497))
    cohorts=[]
    for year in [2024,2025]:
        for h in [10,20]:
            out=root/f'forecast_y{year}_h{h}';p=np.load(out/'partitions.npz')
            history,test=chronology(p['peak_jd_minus_2458000'],year)
            sets=[set(p[k]) for k in ['fit','early_stop','calibration','test']]
            assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
            assert set.union(*sets[:3])==set(history);assert sets[3]==set(test)
            point=pd.read_parquet(out/'points.parquet')
            tables={m:point[point.method==m].reset_index(drop=True) for m in ['baseline','real_delta','blend']}
            for v in tables.values():
                pd.testing.assert_frame_equal(v[['obj_id','day','band','truth']],tables['baseline'][['obj_id','day','band','truth']])
                assert np.isfinite(v[['truth','prediction','error','halfwidth']]).all().all()
                assert (v.day>h).all() and (v.day<=90).all()
                assert set(v.obj_id)<=set(p['obj_id'][p['test']])
                np.testing.assert_allclose(v.error,abs(v.prediction-v.truth))
            np.testing.assert_allclose(tables['blend'].prediction,(tables['baseline'].prediction+tables['real_delta'].prediction)/2,rtol=0,atol=1e-12)
            x=pd.read_parquet(data/f'results/reconstruction/cache/X_pool_h{h}.parquet').replace([np.inf,-np.inf],np.nan)
            cached=np.load(data/f'results/reconstruction/cache/TL_pool_h{h}.npz',allow_pickle=True)
            state=np.load(source/'recency.npz')[f'h{h}'];ids=p['test'][:10]
            mb=lgb.Booster(model_file=str(out/'baseline.txt'));mr=lgb.Booster(model_file=str(out/'real_delta.txt'))
            xx,truth,oi,bi,pb=api.anchored(mb,x,ids,h,cached['FP'],cached['L'],2)
            pr=mr.predict(api.add_recency(xx,oi,bi,state),num_threads=2)+cached['L'][oi,bi]
            for method,values in [('baseline',pb),('real_delta',pr),('blend',(pb+pr)/2)]:
                saved=tables[method][tables[method].obj_id.isin(df.obj_id.iloc[ids])]
                np.testing.assert_array_equal(saved.obj_id,df.obj_id.iloc[oi])
                np.testing.assert_allclose(saved.prediction,values,rtol=1e-12,atol=1e-12)
            replays.append(dict(task='forecast',year=year,h=h,points=len(truth),methods=3))
            cohorts.append(dict(year=year,h=h,history=len(history),test=len(test)))
    result=dict(registered_sources_unchanged=True,disjoint_partitions=True,classification_baseline_replayed=True,
                ensemble_decode_replayed=True,peak_date_split_verified=True,common_forecast_points=True,
                fixed_blend_all_points_verified=True,cohorts=cohorts,model_replays=replays)
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
    complete=json.loads((root/'complete.json').read_text());assert complete['classification_folds']==5 and complete['temporal_jobs']==4
    verify(root);ct,cr=classification(root);ft,fc,fr=forecast(root)
    (root/'summary.json').write_text(json.dumps(dict(classification=cr,forecast=fr),indent=2)+'\n')
    lines=['# Fixed ensemble and chronological forecast follow-up','',
        '**Previously inspected BTS population: development stress tests, not external confirmation.**','',
        '## Classification: three fixed seeds','',
        '| Context | Method | Pooled macro F1 | Accuracy | Fold mean ± SD |','|---|---|---|---|---|']
    for _,r in ct.iterrows():lines.append(f'| {r.context} | {r.method} | {r.macro_f1:.4f} | {r.accuracy:.4f} | {r.fold_mean_f1:.4f} ± {r.fold_std_f1:.4f} |')
    for context,r in cr['contrasts'].items():lines+=['',f'{context}: F1 gain {r["gain"]:+.4f}, paired 95% interval [{r["paired_95ci"][0]:+.4f}, {r["paired_95ci"][1]:+.4f}].']
    lines+=['',f'**Classification adoption: {"MET" if cr["adopt"] else "NOT MET"}.** '+str(cr['hypotheses']),
        '', 'Same fitting/stopping/test objects as the previous paired baseline. Seeds42/17/83, equal weights, no seed or decoder selection on outer results. All saved baseline predictions replay exactly.','',
        f'These are pooled F1 values. The historical paper mean0.832 uses different exact stopping partitions; the new ensemble fold mean is {ct[(ct.context=="available")&(ct.method=="ensemble")].fold_mean_f1.iloc[0]:.4f}. This does not establish a new paper headline best.','',
        '| Class | Baseline F1 | Ensemble F1 |','|---|---|---|']
    cm=ct[ct.context=='available'].set_index('method')
    for c in api.CLASSES:lines.append(f'| {c} | {cm.loc["baseline",c]:.4f} | {cm.loc["ensemble",c]:.4f} |')
    lines += ['',
        '## Forecast: older catalog peaks to 2024/2025','',
        '| Peak cohort | Horizon | Baseline object MAE | Blend object MAE | Relative gain | 95% gain interval |',
        '|---|---|---|---|---|---|']
    for (cohort,h),g in ft.groupby(['cohort','h'],sort=False):
        v=g.set_index('method');r=fr['contrasts'][f'{cohort}_h{h}'];ci=r['paired_95ci']
        lines.append(f'| {cohort} | {h} | {v.loc["baseline","median_object_mae"]:.4f} | {v.loc["blend","median_object_mae"]:.4f} | {r["relative_gain"]:.1%} | [{ci[0]:.4f}, {ci[1]:.4f}] |')
    lines+=['',f'**Temporal robustness criteria: {"MET" if fr["robust"] else "NOT MET"}.** '+str(fr['hypotheses']),
        '', 'T1: pooled reductions >=5%/3% at day10/20, positive bootstrap lower bounds. T2: no aggregate regression in either year/horizon. T3: pooled SLSN/TDE median-object cost <=0.05mag. T4: pooled point-median cost <=0.01mag. All are required.','',
        '## Per-class forecast errors','', '| Cohort | Horizon | Class | Objects | Baseline | Blend |','|---|---|---|---|---|---|']
    for (cohort,h,cls),g in fc.groupby(['cohort','h','cls'],sort=False):
        v=g.set_index('method');lines.append(f'| {cohort} | {h} | {cls} | {int(v.loc["baseline","objects"])} | {v.loc["baseline","median_object_mae"]:.4f} | {v.loc["blend","median_object_mae"]:.4f} |')
    lines+=['','## Scope and reproducibility','',
        '- No production weights/defaults were changed by this evaluation.',
        '- [BTS dates](https://sites.astro.caltech.edu/ztf/bts/explorer_info.html) are catalog peak JD−2458000, not discovery dates. Historical pool peaks must precede Jan1 by >100days; test peaks fall in that year.',
        '- The100-day peak buffer does not prove measurement or label availability. Local event arrays lack absolute start timestamps; historical context/label snapshots are absent. This is not a fully prospective backtest.',
        '- Both cohort populations appeared in previous random-fold development. Refitting only earlier cohorts prevents test-object fitting, not historical model-selection reuse.',
        '- Small rare-class counts limit interpretation. Pointwise residual-band coverage is descriptive; no conditional coverage guarantee.',
        '- Bootstrap intervals resample objects2000 times conditional on fitted models. Retraining and historical selection uncertainty are excluded.',
        '- See classification_confusions.csv, forecast_class_metrics.csv, forecast_lead_metrics.csv, verification.json and the complete artifact archive.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    ax=axes[0];v=ct[ct.context=='available'].set_index('method')
    bars=ax.bar([0,1],[v.loc[m,'macro_f1'] for m in ['baseline','ensemble']],color=['#a8bccb','#386e98'])
    ax.bar_label(bars,fmt='%.4f');ax.set(xticks=[0,1],xticklabels=['Single model','Three seeds'],ylim=(0,1),ylabel='Pooled macro F1',title='Classification: criterion not met')
    for ax,h in zip(axes[1:],[10,20]):
        xpos=np.arange(3);cohorts=[2024,2025,'pooled']
        for method,shift,color in [('baseline',-.18,'#a8bccb'),('blend',.18,'#386e98')]:
            values=[float(ft[(ft.cohort==cohort)&(ft.h==h)&(ft.method==method)].median_object_mae.iloc[0]) for cohort in cohorts]
            bars=ax.bar(xpos+shift,values,.36,label=method,color=color);ax.bar_label(bars,fmt='%.3f',fontsize=8)
        ax.set(xticks=xpos,xticklabels=['2024','2025','Pooled'],ylabel='Median object MAE (mag)',title=f'Forecast from day{h}')
        ax.legend(frameon=False,fontsize=8);ax.set_ylim(0,float(ft[ft.h==h].median_object_mae.max())*1.18)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(classification=cr,forecast=fr),indent=2))


if __name__=='__main__':main()
