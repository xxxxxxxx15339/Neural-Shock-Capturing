"""Audit exact metadata threshold precision and repeat initial Sod timings in isolation.

This is a correctness/timing audit, not parameter selection. First-pass raw
artifacts are copied intact. The fixed protocol and trained weights never change.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts.run_matched_benchmarks import OUT,RAW,MANIFEST,evolve,save_run,write
import numpy as np

def main():
    manifest=json.loads(MANIFEST.read_text());audit=[]
    archive=RAW/'preflight_float32_thresholds';archive.mkdir(exist_ok=True)
    # Preserve exact JSON threshold values when comparing with float32 outputs.
    # The first adapter cast thresholds to float32, unlike original inference.
    for case in manifest['cases']:
        for n in case['grids']:
            base=RAW/f'run_{case["id"]}_{n}_nn_teno5'
            oldbase=archive/base.name
            if not oldbase.with_suffix('.json').exists():
                for ext in ('.json','.npz'):shutil.copy2(base.with_suffix(ext),oldbase.with_suffix(ext))
            row,arr=evolve(case,n,'nn_teno5',manifest)
            with np.load(oldbase.with_suffix('.npz')) as old:delta=float(np.max(abs(old['final']-arr['final'])))
            previous=json.loads(oldbase.with_suffix('.json').read_text())
            row['audit_note']='Exact float64 metadata thresholds against float32 classifier outputs; first-pass artifact preserved.'
            save_run(base,row,arr)
            audit.append(dict(case=case['id'],cells=n,old_success=previous['success'],new_success=row['success'],
                maximum_final_state_difference=delta,old_source=str(oldbase.relative_to(ROOT))+'.json',
                final_source=str(base.relative_to(ROOT))+'.json'))
            print('THRESHOLD AUDIT',case['id'],n,row['success'],delta,flush=True)
    # The first Sod smoke overlapped archival reproduction. Retime it after
    # all numerical/reference jobs, without changing ICs or numerical parameters.
    archive=RAW/'preflight_sod_timings';archive.mkdir(exist_ok=True)
    case=next(c for c in manifest['cases'] if c['id']=='sod')
    for n in case['grids']:
        methods=['teno5','neno6','learned_head']+(['head_backbone_only','head_fixed_blend'] if n==128 else [])
        for method in methods:
            base=RAW/f'run_sod_{n}_{method}';oldbase=archive/base.name
            if not oldbase.with_suffix('.json').exists():
                for ext in ('.json','.npz'):shutil.copy2(base.with_suffix(ext),oldbase.with_suffix(ext))
            row,arr=evolve(case,n,method,manifest)
            with np.load(oldbase.with_suffix('.npz')) as old:np.testing.assert_array_equal(arr['final'],old['final'])
            row['audit_note']='Initial Sod timing repeated after archival jobs; identical final numerical state.'
            save_run(base,row,arr);print('SOD ISOLATED TIMING',n,method,flush=True)
    write(RAW/'threshold_precision_audit.json',dict(reason='Preserve original float64 metadata threshold comparisons; no test-based selection',cases=audit,
        any_success_changed=any(r['old_success']!=r['new_success'] for r in audit),
        maximum_final_state_difference=max(r['maximum_final_state_difference'] for r in audit)))

if __name__=='__main__':main()
