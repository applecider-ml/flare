"""Summarize registered supervised comparisons with paired object uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from improve_supervised import CLASSES,sha


def metrics(y,p):
    c=np.bincount(np.asarray(y,dtype=int)*6+np.asarray(p,dtype=int),minlength=36).reshape(6,6)
    per=2*np.diag(c)/np.maximum(c.sum(0)+c.sum(1),1)
    return dict(macro_f1=float(per.mean()),accuracy=float(np.trace(c)/c.sum()),per_class=per.tolist())


def classification_interval(y,a,b,draws=2000):
    rng=np.random.default_rng(827);groups=[np.flatnonzero(y==c) for c in range(6)];values=[]
    for _ in range(draws):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        values.append(metrics(y[ix],b[ix])['macro_f1']-metrics(y[ix],a[ix])['macro_f1'])
    return np.quantile(values,[.025,.975]).tolist()


def forecast_interval(a,b,draws=2000):
    a=np.asarray(a);b=np.asarray(b);rng=np.random.default_rng(828);values=[]
    for _ in range(draws):
        ix=rng.integers(0,len(a),len(a));values.append(float(np.median(a[ix])-np.median(b[ix])))
    return np.quantile(values,[.025,.975]).tolist()


def summarize_classification(root):
    data=pd.concat([pd.read_csv(root/f'classification_fold{f}/predictions.csv') for f in range(5)],ignore_index=True)
    rows=[];results={}
    for context in ['available','absent']:
        d=data[data.context==context];pivot=d.pivot(index='obj_id',columns='method',values='predicted').sort_index()
        truth=d.drop_duplicates('obj_id').set_index('obj_id').loc[pivot.index,'truth'].to_numpy(dtype=int)
        if len(pivot)!=10497 or pivot.isna().any().any():raise ValueError('incomplete classification OOF')
        result={m:metrics(truth,pivot[m].to_numpy()) for m in ['baseline','multiview']}
        results[context]=result
        for method,value in result.items():
            folds=[metrics(g.truth.to_numpy(),g.predicted.to_numpy())['macro_f1']
                   for _,g in d[d.method==method].groupby('fold')]
            rows.append(dict(context=context,method=method,macro_f1=value['macro_f1'],accuracy=value['accuracy'],
                fold_mean_f1=float(np.mean(folds)),fold_std_f1=float(np.std(folds)),
                **{c:value['per_class'][i] for i,c in enumerate(CLASSES)}))
        if context=='available':
            ci=classification_interval(truth,pivot.baseline.to_numpy(),pivot.multiview.to_numpy())
        else:
            missing_ci=classification_interval(truth,pivot.baseline.to_numpy(),pivot.multiview.to_numpy())
    base=results['available']['baseline'];new=results['available']['multiview']
    gain=new['macro_f1']-base['macro_f1'];acc=new['accuracy']-base['accuracy']
    per=np.array(new['per_class'])-np.array(base['per_class'])
    missing=results['absent']['multiview']['macro_f1']-results['absent']['baseline']['macro_f1']
    verdict=dict(C1=bool(gain>=.005 and ci[0]>0),C2=bool(acc>=-.002 and (per>=-.03).all()),C3=bool(missing>=-.01))
    result=dict(hypotheses=verdict,adopt=all(verdict.values()),macro_f1_delta=gain,paired_95ci=ci,
                accuracy_delta=acc,missing_context_f1_delta=missing,missing_context_paired_95ci=missing_ci,
                per_class_delta=dict(zip(CLASSES,per.tolist())))
    pd.DataFrame(rows).to_csv(root/'classification_metrics.csv',index=False)
    data.to_parquet(root/'classification_oof.parquet',index=False)
    return pd.DataFrame(rows),result


def summarize_forecast(root,primary='real_delta',methods=('baseline','grid_delta','real_delta')):
    points=pd.concat([pd.read_parquet(root/f'forecast_h{h}_fold{f}/points.parquet')
                      for h in [10,20] for f in range(5)],ignore_index=True)
    grid=pd.concat([pd.read_parquet(root/f'forecast_h{h}_fold{f}/day50.parquet')
                    for h in [10,20] for f in range(5)],ignore_index=True)
    grid['error']=abs(grid.prediction-grid.truth)
    # Repeated observations never become independent bootstrap objects.
    objects=points.groupby(['h','method','obj_id','truth_class','fold'],as_index=False).error.mean()
    rows=[];deltas={};rare_checks=[];point_checks=[];primary_checks=[];class_rows=[]
    for h in [10,20]:
        d=points[points.h==h];o=objects[objects.h==h]
        pivot=o.pivot(index='obj_id',columns='method',values='error').sort_index()
        if pivot.isna().any().any():raise ValueError('forecast populations differ')
        gain=float(np.median(pivot.baseline)-np.median(pivot[primary]))
        relative=gain/float(np.median(pivot.baseline));ci=forecast_interval(pivot.baseline,pivot[primary])
        primary_checks.append(relative>=(.05 if h==10 else .03) and ci[0]>0)
        deltas[str(h)]=dict(object_n=len(pivot),median_object_mae_gain=gain,relative_gain=relative,paired_gain_95ci=ci)
        for name in methods:
            p=d[d.method==name];v=o[o.method==name];g=grid[(grid.h==h)&(grid.method==name)]
            rows.append(dict(h=h,method=name,objects=len(v),points=len(p),median_object_mae=float(v.error.median()),
                mean_object_mae=float(v.error.mean()),point_median_error=float(p.error.median()),
                day50_grid_median_error=float(g.error.median()),coverage68=float(p.covered.mean()),
                mean_width68=float(2*p.halfwidth.mean())))
            for cls,gg in v.groupby('truth_class'):
                class_rows.append(dict(h=h,method=name,cls=CLASSES[cls],objects=len(gg),median_object_mae=float(gg.error.median())))
        for cls in [2,4]:
            v=o[o.truth_class==cls].groupby('method').error.median()
            rare_checks.append(float(v[primary]-v.baseline)<=.05)
        med=d.groupby('method').error.median()
        point_checks.append(float(med[primary]-med.baseline)<=.01)
    verdict=dict(F1=bool(all(primary_checks)),F2=bool(all(rare_checks)),F3=bool(all(point_checks)))
    result=dict(hypotheses=verdict,adopt=all(verdict.values()),comparisons=deltas)
    points['lead_bin']=pd.cut(points.day-points.h,[0,5,15,30,60,100],right=True)
    lead=points.groupby(['h','method','lead_bin'],observed=True).agg(points=('error','size'),
        median_error=('error','median'),coverage68=('covered','mean')).reset_index()
    table=pd.DataFrame(rows);table.to_csv(root/'forecast_metrics.csv',index=False)
    pd.DataFrame(class_rows).to_csv(root/'forecast_class_metrics.csv',index=False)
    lead.to_csv(root/'forecast_lead_metrics.csv',index=False)
    objects.to_parquet(root/'forecast_object_errors.parquet',index=False)
    return table,result,pd.DataFrame(class_rows)


def report(root,ct,cr,ft,fr,classes):
    lines=['# Supervised improvements: paired development evaluation','',
        'Two fixed primary candidates tested against the adopted classification and '
        'forecast recipes. Five object folds each. This population has been inspected '
        'in earlier research; results are development evidence, not a fresh external test.','',
        '## Classification: multiple context views','',
        'The baseline reproduces config F: full-curve hierarchy, 190 features, '
        'class weighting, external-context dropout and a stopping-only branch threshold. '
        'The candidate keeps a complete view plus three independently masked views per '
        'fitting object, preserving expected dropout and total object weight. The test '
        'objects and stopping inputs are paired. This is not the earlier comparison '
        'of a full-trained classifier against horizon-specific models.','',
        '| Context | Method | Pooled macro F1 | Accuracy | Fold mean F1 ± SD |',
        '|---|---|---|---|---|']
    for _,r in ct.iterrows():
        lines.append(f'| {r.context} | {r.method} | {r.macro_f1:.4f} | {r.accuracy:.4f} | {r.fold_mean_f1:.4f} ± {r.fold_std_f1:.4f} |')
    ci=cr['paired_95ci']
    lines+=['',f'Available-context F1 delta **{cr["macro_f1_delta"]:+.4f}**, paired stratified '
            f'95% bootstrap **[{ci[0]:+.4f}, {ci[1]:+.4f}]**.','',
            'The comparison uses new paired refits of the adopted recipe. Their exact '
            'stopping partitions are saved; the historical paper mean of 0.832 is not '
            'substituted for this measured baseline. Numerical-label stratification in '
            'the stopping carve-out differs from the historical string-label ordering.','',
            f'Absent-context stress-test delta **{cr["missing_context_f1_delta"]:+.4f}**, '
            f'95% paired interval **[{cr["missing_context_paired_95ci"][0]:+.4f}, '
            f'{cr["missing_context_paired_95ci"][1]:+.4f}]**. This is a prespecified '
            'secondary condition, not a substitute for the primary available-context '
            'adoption rule. It compares the same context-trained classifiers under '
            'missingness; it does not establish superiority over a dedicated LC-only classifier.','',
            '| Class | Baseline F1 | Multiview F1 |','|---|---|---|']
    a=ct[ct.context=='available'].set_index('method')
    for c in CLASSES:lines.append(f'| {c} | {a.loc["baseline",c]:.4f} | {a.loc["multiview",c]:.4f} |')
    lines+=['',f'**Classification adoption: {"MET" if cr["adopt"] else "NOT MET"}.**',
            '- C1: >=0.005 pooled macro-F1 gain and positive paired lower bound.',
            '- C2: accuracy loss <=0.002 and no class F1 loss >0.03.',
            '- C3: absent-context macro-F1 loss <=0.01.',
            'Verdicts: '+', '.join(f'{k}={"HOLDS" if v else "FAILS"}' for k,v in cr['hypotheses'].items())+'.','',
            '## Forecast: real future detections and recent band state','',
            'Baseline parameters are the adopted tuned pooled L1 model, refitted in each '
            'fold on the original grid targets. It uses the historical horizon anchor '
            'when interpolated to real detection times. The primary real_delta model '
            'learns change from the last observed band magnitude directly on real '
            'future detections, with equal total weight per training object. It also '
            'receives band-local last magnitude, observation age, recent slope and '
            'scatter. grid_delta is a fixed ablation with grid targets.','',
            '| Horizon | Method | Median object MAE | Point-median error | Day50 grid error |',
            '|---|---|---|---|---|']
    for _,r in ft.iterrows():
        lines.append(f'| {int(r.h)} | {r.method} | {r.median_object_mae:.4f} | {r.point_median_error:.4f} | {r.day50_grid_median_error:.4f} |')
    lines+=['','Errors in magnitudes. Primary: mean absolute error over each object’s '
            'actual future detections through day90, then median over objects. The '
            'point-median and interpolated day50 metrics are distinct secondary targets. '
            'All methods share exactly the same eligible detections.','']
    for h,r in fr['comparisons'].items():
        ci=r['paired_gain_95ci']
        lines.append(f'- Day {h}: primary reduction {100*r["relative_gain"]:.2f}%; '
            f'gain {r["median_object_mae_gain"]:+.4f} mag, 95% paired object-bootstrap '
            f'[{ci[0]:+.4f}, {ci[1]:+.4f}]; {r["object_n"]} objects.')
    lines+=['',f'**Forecast adoption: {"MET" if fr["adopt"] else "NOT MET"}.**',
            '- F1: primary reduction >=5% at day10 and >=3% at day20, positive lower bounds at both.',
            '- F2: SLSN/TDE median object MAE deterioration <=0.05 mag at both horizons.',
            '- F3: point-median deterioration <=0.01 mag at both horizons.',
            'Verdicts: '+', '.join(f'{k}={"HOLDS" if v else "FAILS"}' for k,v in fr['hypotheses'].items())+'.','',
            '## Limits and artifacts','',
            '- No production weights are replaced by this experiment. A successful candidate needs a separately identified final fit and calibration.',
            '- Both experiments have fixed primaries. An ablation cannot become a winner retrospectively.',
            '- Bootstrap intervals use 2,000 paired object resamples, conditional on these fitted folds. They omit retraining and historical benchmark-selection uncertainty.',
            '- Forecast residual bands are plain 68% bands recalibrated on real calibration points for each model. Their coverage and widths are descriptive; they are not the shipped adaptive intervals.',
            '- Fold models, partitions, thresholds, input hashes, recent-state arrays and OOF outputs are retained. No anomaly labels were used.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')


def plots(root,ct,ft):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    c=ct[ct.context=='available'].set_index('method')
    for i,m in enumerate(['baseline','multiview']):axs[0].bar(i,c.loc[m,'macro_f1'],color=['#a8bccb','#386e98'][i])
    axs[0].set_xticks([0,1],['Adopted recipe','Multiview'],rotation=25,ha='right')
    axs[0].set(title='Full-curve classification',ylabel='OOF macro F1',ylim=(0,1))
    for ax,h in zip(axs[1:],[10,20]):
        v=ft[ft.h==h].set_index('method')
        for i,m in enumerate(['baseline','grid_delta','real_delta']):
            ax.bar(i,v.loc[m,'median_object_mae'],color='#386e98' if m=='real_delta' else '#a8bccb')
        ax.set_xticks(range(3),['Adopted grid','Grid delta','Real delta'],rotation=25,ha='right')
        ax.set(title=f'Forecast from day {h}',ylabel='Median object MAE (mag)')
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
    complete=json.loads((root/'complete.json').read_text())
    if complete['classification_folds']!=5 or complete['forecast_folds']!=5:raise ValueError('incomplete run')
    ct,cr=summarize_classification(root);ft,fr,classes=summarize_forecast(root)
    result=dict(classification=cr,forecast=fr,script_sha256=complete['script_sha256'])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    report(root,ct,cr,ft,fr,classes);plots(root,ct,ft)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
