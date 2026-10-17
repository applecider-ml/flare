"""Report the separately registered fixed-blend follow-up."""
import argparse,json
from pathlib import Path

from summarize_supervised import summarize_forecast


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
    complete=json.loads((root/'complete.json').read_text())
    if complete['folds']!=5:raise ValueError('incomplete run')
    table,result,classes=summarize_forecast(root,primary='blend',methods=('baseline','real_delta','blend'))
    result['hypotheses']={k.replace('F','B'):v for k,v in result['hypotheses'].items()}
    result['script_sha256']=complete['script_sha256']
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Fixed 50/50 forecast blend: registered follow-up','',
        'The preceding real-detection forecaster improved aggregate day10 error but '
        'worsened SLSN error beyond its registered tolerance. This follow-up fixed a '
        '50/50 average with the adopted-grid baseline before evaluating any blend '
        'predictions. Both components retrain on new five-fold partitions, seed2026, '
        'with disjoint fitting/stopping/calibration/test objects.','',
        '**The objects are the same inspected BTS population. New partitions are '
        'not an untouched population or external confirmation.** The original '
        'experiment retains its own verdict. No mixture weights or class routes '
        'were searched. The unblended method is a control.','',
        '| Horizon | Method | Median object MAE | Point-median error | Day50 grid error |',
        '|---|---|---|---|---|']
    for _,r in table.iterrows():
        lines.append(f'| {int(r.h)} | {r.method} | {r.median_object_mae:.4f} | {r.point_median_error:.4f} | {r.day50_grid_median_error:.4f} |')
    lines+=['','Errors are in magnitudes. Primary: mean error across each object’s '
            'real future detections through day90, then median over objects. All '
            'methods share the same eligible detections.','']
    for h,r in result['comparisons'].items():
        ci=r['paired_gain_95ci']
        lines.append(f'- Day {h}: **{100*r["relative_gain"]:.2f}%** reduction; gain '
            f'**{r["median_object_mae_gain"]:.4f} mag**, paired 95% interval '
            f'**[{ci[0]:.4f}, {ci[1]:.4f}]**; {r["object_n"]} objects.')
    lines+=['','## Class-level primary error','',
            '| Horizon | Class | Objects | Baseline | Blend |','|---|---|---|---|---|']
    for (h,c),d in classes.groupby(['h','cls']):
        v=d.set_index('method')
        lines.append(f'| {h} | {c} | {int(v.loc["baseline","objects"])} | '
                     f'{v.loc["baseline","median_object_mae"]:.4f} | {v.loc["blend","median_object_mae"]:.4f} |')
    lines+=['','## Registered verdict','',
            '- B1: >=5% primary improvement at day10 and >=3% at day20; positive lower bounds at both.',
            '- B2: SLSN/TDE median object MAE deterioration <=0.05 mag at both horizons.',
            '- B3: global point-median deterioration <=0.01 mag at both horizons.']
    lines += [f'- {k}: {"HOLDS" if v else "FAILS"}.' for k,v in result['hypotheses'].items()]
    lines+=['',f'**Adoption criterion: {"MET" if result["adopt"] else "NOT MET"}.**','',
            'Uncertainty uses 2,000 paired object bootstrap draws conditional on these '
            'fitted models. It does not include retraining or historical benchmark reuse. '
            'The baseline is a paired refit of adopted tuned parameters, not the exact '
            'production checkpoint. Plain residual-band coverage is descriptive, not '
            'a guarantee for individual classes or whole light curves.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,h in zip(axs,[10,20]):
        v=table[table.h==h].set_index('method')
        for i,m in enumerate(['baseline','real_delta','blend']):
            ax.bar(i,v.loc[m,'median_object_mae'],color='#386e98' if m=='blend' else '#a8bccb')
        ax.set_xticks(range(3),['Adopted grid','Real delta','50/50 blend'],rotation=25,ha='right')
        ax.set(title=f'Forecast from day {h}',ylabel='Median object MAE (mag)');ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
