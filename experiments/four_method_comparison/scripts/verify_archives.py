"""Check archived numerical values against original entry-point reproductions."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/four_method_comparison'

def leaves(x,path=''):
    if isinstance(x,dict):
        for k,v in x.items():yield from leaves(v,path+'/'+k)
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from leaves(v,path+'/'+str(i))
    elif isinstance(x,(int,float)) and not isinstance(x,bool):yield path,x

def main():
    comparisons=[('models/nn_teno5_scalar_comparison.json','reproduced_original_scalar.json'),
                 ('models/nn_teno5euler_jax_comparison_phase18.json','reproduced_phase18_euler.json')]
    records=[]
    for original,new in comparisons:
        path=OUT/'results/raw'/new
        if not path.exists():continue
        old=dict(leaves(json.loads((ROOT/original).read_text())))
        fresh=dict(leaves(json.loads(path.read_text())))
        for key,value in old.items():
            if any(term in key for term in ('runtime','seconds')):continue
            if key in fresh and any(term in key for term in ('l1','l2','linf','minimum','conservation','total_variation')):
                delta=abs(value-fresh[key]);records.append(dict(status='REPRODUCED' if delta<1e-10 else 'UNAVAILABLE',
                    original=original,reproduced=str(path.relative_to(ROOT)),json_pointer=key,archived=value,recomputed=fresh[key],absolute_difference=delta))
    with (OUT/'results/archive_reproduction.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report=dict(compared_values=len(records),max_abs_difference=max(r['absolute_difference'] for r in records),
                mismatches=sum(r['status']!='REPRODUCED' for r in records),
                warning='Scalar archived burgers_smooth_to_shock at t=.6 is pre-shock (shock time2/pi); name is not post-shock evidence. Archived fidelity-factor pass is not a strict improvement.')
    (OUT/'results/raw/archive_reproduction_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
