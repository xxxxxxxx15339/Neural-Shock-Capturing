"""Derived statistics from immutable full-validation measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from .artifacts import create_output,write_json,finish_manifest
from .model_io import REPO_ROOT,sha256_file


def derive(source,output):
    rows=json.loads((source/'per_case.json').read_text());config=json.loads((source/'configuration.json').read_text())
    baseline={r['case_id']:r for r in rows if r['mode']=='soft' and r['strength']==0}
    cases=[];groups={}
    for row in rows:
        base=baseline[row['case_id']]
        r={k:row[k] for k in ['case_id','physics','family','cells','mode','strength','valid']}
        for norm in ['l1','l2','linf']:
            r['absolute_'+norm]=row[norm];r['baseline_'+norm]=base[norm]
            r['relative_'+norm]=row[norm]/base[norm] if row['valid'] and base['valid'] and row['family']!='constant' and base[norm]>config['ratio_error_floor'] else None
        cases.append(r)
        groups.setdefault((row['physics'],row['mode'],row['strength'],row['family'],row['cells']),[]).append(row)
    summaries=[]
    for key,values in sorted(groups.items()):
        valid=[r for r in values if r['valid'] and baseline[r['case_id']]['valid']]
        result=dict(zip(['physics','mode','strength','family','cells'],key));result['count']=len(values);result['failures']=len(values)-len(valid)
        result['statistics_condition_on_valid_cases']=True
        for norm in ['l1','l2','linf']:
            a=np.array([r[norm] for r in valid]);b=np.array([baseline[r['case_id']][norm] for r in valid])
            result[norm]=dict(mean=float(a.mean()) if len(a) else None,
                              quantiles=np.quantile(a,[0,.05,.25,.5,.75,.95,1]).tolist() if len(a) else None,
                              relative_to_classical=float(a.mean()/b.mean()) if len(a) and key[3]!='constant' and b.mean()>config['ratio_error_floor'] else None)
        for metric in ['total_variation','overshoot','undershoot','state_min','state_max']:
            a=np.array([r[metric] for r in valid]);result[metric]=dict(mean=a.mean(axis=0).tolist(),minimum=a.min(axis=0).tolist(),maximum=a.max(axis=0).tolist()) if len(a) else None
        if key[0]=='euler':
            result['minimum_density']=min((r['minimum_density'] for r in valid),default=None)
            result['minimum_pressure']=min((r['minimum_pressure'] for r in valid),default=None)
        summaries.append(result)
    overall=[]
    for physics in config['physics']:
        for mode in config['modes']:
            for strength in config['strengths']:
                selected=[r for r in cases if r['physics']==physics and r['mode']==mode and r['strength']==strength and r['valid']]
                gs=[g for g in summaries if g['physics']==physics and g['mode']==mode and g['strength']==strength]
                result=dict(physics=physics,mode=mode,strength=strength)
                for norm in ['l1','l2','linf']:
                    eligible=[r for r in selected if r['relative_'+norm] is not None]
                    result[norm]=dict(macro_absolute=float(np.mean([g[norm]['mean'] for g in gs if g[norm]['mean'] is not None])),
                         trajectory_weighted_absolute=float(np.mean([r['absolute_'+norm] for r in selected])),
                         macro_relative=float(np.mean([g[norm]['relative_to_classical'] for g in gs if g[norm]['relative_to_classical'] is not None])),
                         trajectory_weighted_relative=sum(r['absolute_'+norm] for r in eligible)/sum(r['baseline_'+norm] for r in eligible) if eligible else None)
                overall.append(result)
    write_json(output/'statistics.json',dict(source_sha256=sha256_file(source/'per_case.json'),
          definitions='L1=mean absolute; L2=RMS; Linf=max absolute over cells and components. TV includes periodic jump only for scalar. Overshoot/undershoot compare final reference extrema componentwise. Euler variables have different units; unscaled aggregates retain historical metric.',
          bootstrap='L1 confidence intervals in immutable full-validation/summary.json; this derived report supplies additional norms and distribution quantiles, not new uncertainty estimates.',
          overall=overall,groups=summaries))
    write_json(output/'per_case_relative.json',cases)
    finish_manifest(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=REPO_ROOT/'docs/research_release/results/full-validation');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();derive(a.source,create_output(a.output))


if __name__=='__main__':main()
