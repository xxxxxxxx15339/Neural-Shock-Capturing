"""Finest-grid dt/2 control, declared before execution; no model or parameter tuning."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts.run_matched_benchmarks import OUT,RAW,MANIFEST,evolve,save_run,write
from experiments.four_method_comparison.scripts.calculate_metrics import norms
from experiments.four_method_comparison.scripts.collect_existing_results import csv_write
import numpy as np

def main():
    manifest=json.loads(MANIFEST.read_text());case=next(c for c in manifest['cases'] if c['id']=='advection_smooth')
    config=dict(case='advection_smooth',cells=256,dt_refinements=[1,2],methods=manifest['methods'],
        purpose='Measure temporal sensitivity separately from observed full-solution grid convergence; no fifth-order temporal claim')
    cfg=OUT/'configs/smooth_time_control.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:write(cfg,config)
    rows=[]
    for method in config['methods']:
        base=RAW/f'temporal_advection_smooth_256_{method}_dt2'
        if base.with_suffix('.json').exists():d=json.loads(base.with_suffix('.json').read_text())
        else:
            d,a=evolve(case,256,method,manifest,dt_refine=2,timing=False);save_run(base,d,a)
        assert d['method']==method and d['success']
        with np.load(base.with_suffix('.npz')) as fine,np.load(RAW/f'run_advection_smooth_256_{method}.npz') as coarse:
            a=norms(coarse['final'],coarse['reference']);b=norms(fine['final'],fine['reference']);delta=norms(fine['final'],coarse['final'])
            rows.append(dict(case='advection_smooth',cells=256,method=method,dt=d['dt']*2,dt_half=d['dt'],
                dt_l1=a['l1'],dt_half_l1=b['l1'],temporal_change_l1=delta['l1'],temporal_change_fraction=delta['l1']/a['l1'],
                status='NEW MATCHED EVALUATION',source_file=str(base.relative_to(ROOT))+'.json'))
        print('SMOOTH DT CONTROL',method,rows[-1],flush=True)
    csv_write(OUT/'results/smooth_time_control.csv',rows)

if __name__=='__main__':main()
