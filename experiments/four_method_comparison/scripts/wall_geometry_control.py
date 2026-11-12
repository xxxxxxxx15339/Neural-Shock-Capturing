"""Additional predeclared wall-aligned point-grid stability control (no new model)."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts.run_matched_benchmarks import OUT,RAW,MANIFEST,evolve,save_run,write,clean
from experiments.four_method_comparison.scripts.collect_existing_results import csv_write

def main():
    manifest=json.loads(MANIFEST.read_text());config=dict(
        purpose='Check physical wall alignment independently of historical x=i*dx grid; all methods, no result-based parameter selection',
        sampling='point-cell-centers',physical_wall_locations=[0.,1.],grids=[128],times=[.001,.038],
        methods=manifest['methods'],boundary='reflective-wall',pressure_regions=[1000.,.01,100.],
        density=1.,velocity=0.,pressure_jumps=[.1,.9],repair_policy='none',reference='none; stability only')
    cfg=OUT/'configs/wall_geometry_control.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:write(cfg,config) # Freeze before running the additional wall cases.
    records=[]
    for end in config['times']:
        case=dict(id='blast_wall_aligned_'+str(end),physics='euler',equation='euler',time=end,length=1.,
                  boundary='reflective-wall',reference='none_stability_only',sampling='point-cell-centers')
        for method in config['methods']:
            # Path.with_suffix treats a decimal in a stem as a suffix separator.
            # Use an unambiguous token so methods/times never share a cache path.
            token=str(end).replace('.','p')
            base=RAW/f'geometry_blast_{token}_128_{method}'
            if base.with_suffix('.json').exists():r=json.loads(base.with_suffix('.json').read_text())
            else:
                r,a=evolve(case,128,method,manifest,timing=False);r['supplemental_protocol']='configs/wall_geometry_control.json';save_run(base,r,a)
            assert r['method']==method and r['requested_time']==end and r['case']==case['id']
            records.append(dict(case=case['id'],cells=128,method=method,success=r['success'],requested_time=end,
                actual_time=r['actual_time'],raw_min_density=r['raw_min_density'],raw_min_pressure=r['raw_min_pressure'],
                first_invalid_step=None if r['failure'] is None else r['failure']['step'],
                first_invalid_stage=None if r['failure'] is None else r['failure']['stage'],
                conservation_residual=json.dumps(clean(r['conservation_residual'])),status='NEW MATCHED EVALUATION',source_file=str(base.relative_to(ROOT))+'.json'))
            print('WALL GEOMETRY',end,method,r['success'],flush=True)
    csv_write(OUT/'results/wall_geometry_control.csv',records)

if __name__=='__main__':main()
