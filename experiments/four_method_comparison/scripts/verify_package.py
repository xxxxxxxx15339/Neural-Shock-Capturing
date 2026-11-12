"""Fail loudly on missing cases, corrupted provenance, repaired states or invented scores."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts import methods as m
from experiments.four_method_comparison.scripts.calculate_metrics import primitives,norms
from experiments.four_method_comparison.scripts.run_matched_benchmarks import clean
import numpy as np

OUT=m.OUT;RAW=OUT/'results/raw'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest_path=OUT/'configs/benchmark_manifest.json';manifest=json.loads(manifest_path.read_text())
    assert sha(manifest_path)==manifest_path.with_suffix('.sha256').read_text().split()[0]
    protected=list(csv.DictReader((RAW/'archived_files.csv').open()))
    changed=[r['path'] for r in protected if not (ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    assert not changed,changed
    for cfg in manifest['models'].values():assert sha(ROOT/cfg['path'])==cfg['sha256']
    rows=json.loads((RAW/'evaluated_rows.json').read_text())
    expected={(c['id'],n,method) for c in manifest['cases'] for n in c['grids'] for method in manifest['methods']}
    expected|={(c,n,method) for c in manifest['ablation']['cases'] for n in manifest['ablation']['grids'] for method in ('head_backbone_only','head_fixed_blend')}
    lookup={(r['case'],r['cells'],r['method']):r for r in rows}
    assert len(rows)==len(lookup) and set(lookup)==expected,(len(rows),len(expected),expected-set(lookup))
    failures=0;max_relative_budget=0.;max_initial_scale_budget=0.
    for r in rows:
        source=ROOT/r['source_file'];raw=json.loads(source.read_text())
        assert raw['manifest_sha256']==sha(manifest_path)
        assert raw['repair_count']==raw['retry_count']==0
        with np.load(ROOT/r['trajectory_file']) as a:
            initial,final=a['initial'],a['final'];res=np.asarray(r['conservation_residual'])
            np.testing.assert_allclose(res,r['dx']*np.sum(final-initial,axis=0)+np.asarray(r['boundary_exchange']),atol=1e-13,rtol=1e-13)
            initial_scale=np.maximum(1.,np.maximum(np.abs(r['dx']*np.sum(initial,axis=0)),np.abs(r['boundary_exchange'])))
            max_initial_scale_budget=max(max_initial_scale_budget,float(np.max(abs(res)/initial_scale)))
            # Floating cancellation scales with magnitudes, not signed total mass.
            # The finite explosive scalar ablation is still explicitly flagged.
            scale=np.maximum(initial_scale,np.maximum(r['dx']*np.sum(abs(final),axis=0),r['dx']*np.sum(abs(initial),axis=0)))
            rel=float(np.max(abs(res)/scale));max_relative_budget=max(max_relative_budget,rel)
            assert rel<1e-10,(r['case'],r['method'],rel)
            if r['success']:
                assert abs(r['actual_time']-r['requested_time'])<1e-12
                assert np.all(np.isfinite(final))
                if r['physics']=='euler':assert np.min(final[:,0])>0 and np.min(primitives(final)[2])>0
                assert len(r['rollout_repeats_seconds'])==3
                if r.get('l1') is not None:
                    ref=a['reference']
                    if r['case']=='shu_osher':
                        with np.load(RAW/'reference_shu_2048_dt2.npz') as b:ref=np.column_stack([np.interp(a['x'],b['x'],b['final'][:,k]) for k in range(3)])
                    np.testing.assert_allclose(r['l1'],norms(final,ref)['l1'],atol=1e-15,rtol=1e-14)
            else:
                failures+=1;assert r.get('l1') is None and r.get('runtime_median_s') is None
                assert r['failure'] and r['failure']['stage'] in (1,2,3)
                candidate=a['invalid_candidate']
                valid=np.isfinite(candidate).all()
                if candidate.ndim==2:valid=valid and np.all(candidate[:,0]>0) and np.all(primitives(candidate)[2]>0)
                assert not valid
                assert np.isfinite(a['last_admissible_stage']).all()
    for r in csv.DictReader((OUT/'results/percentage_improvements.csv').open()):
        case,n,metric=r['benchmark'],int(r['cells']),r['metric']
        pairs=[('neno6_vs_teno5_percent','neno6','teno5'),('nn_vs_teno5_percent','nn_teno5','teno5'),
               ('v2_vs_teno5_percent','learned_head','teno5'),('v2_vs_nn_percent','learned_head','nn_teno5')]
        for key,a,b in pairs:
            va=lookup[(case,n,a)].get(metric);vb=lookup[(case,n,b)].get(metric)
            if va is not None and vb is not None and vb>0:np.testing.assert_allclose(float(r[key]),100*(1-va/vb),atol=1e-12)
            else:assert r[key]==''
    # Exported parameters must equal the original HDF5 arrays, without a model rewrite.
    import h5py
    values=[]
    with h5py.File(ROOT/'models/nn_teno5.h5') as f:
        f['model_weights'].visititems(lambda name,obj:values.append(np.asarray(obj)) if isinstance(obj,h5py.Dataset) else None)
    with np.load(RAW/'original_keras_export.npz') as z:
        for i in range(10):assert any(v.shape==z[f'a{i}'].shape and np.array_equal(v,z[f'a{i}']) for v in values)
    required=['01_sod_density_comparison','02_lax_density_comparison','03_shu_osher_density_zoom','04_burgers_comparison',
              '05_accuracy_bar_chart','06_error_by_grid_resolution','07_mixing_head_lambda_distribution','08_accuracy_vs_runtime','09_training_evolution_timeline']
    for name in required:
        for extension in ('.png','.pdf'):assert (OUT/'figures'/(name+extension)).stat().st_size>5000
        from PIL import Image
        with Image.open(OUT/'figures'/(name+'.png')) as picture:assert picture.size==(3072,1728)
    geometry=list(csv.DictReader((OUT/'results/wall_geometry_control.csv').open()))
    assert len(geometry)==8
    for r in geometry:
        d=json.loads((ROOT/r['source_file']).read_text())
        assert d['method']==r['method'] and d['case']==r['case'] and d['requested_time']==float(r['requested_time'])
        assert d['repair_count']==0 and d['retry_count']==0
        with np.load((ROOT/r['source_file']).with_suffix('.npz')) as a:
            assert a['x'][0]==.5/128 and a['x'][-1]==1-.5/128
            np.testing.assert_allclose(d['conservation_residual'],np.sum(a['final']-a['initial'],axis=0)/128+np.asarray(d['boundary_exchange']),atol=1e-13)
            if not d['success']:assert np.min(primitives(a['invalid_candidate'])[2])<=0 or not np.isfinite(a['invalid_candidate']).all()
    parity=json.loads((RAW/'actual_input_parity.json').read_text())
    assert parity['mask_mismatch_count']==0 and parity['stencil_count']==8448
    timecontrols=list(csv.DictReader((OUT/'results/smooth_time_control.csv').open()))
    assert len(timecontrols)==4
    for r in timecontrols:
        with np.load((ROOT/r['source_file']).with_suffix('.npz')) as a,np.load(RAW/f'run_advection_smooth_256_{r["method"]}.npz') as b:
            np.testing.assert_allclose(float(r['temporal_change_l1']),norms(a['final'],b['final'])['l1'],atol=1e-16)
    previous=RAW/'preflight_wave_window_diagnostics/evaluated_rows.json'
    if previous.exists():
        old={(r['case'],r['cells'],r['method']):r for r in json.loads(previous.read_text())}
        assert all(r.get('l1')==old[(r['case'],r['cells'],r['method'])].get('l1') for r in rows)
    report=dict(passed=True,protected_files_verified=len(protected),checkpoint_identities_verified=4,
        expected_rollouts=len(expected),measured_rollouts=len(rows),failures_preserved=failures,
        failed_endpoint_errors_absent=True,percentage_checks=True,exact_export_weights=True,
        maximum_relative_boundary_budget_residual=max_relative_budget,required_figures_verified=len(required),
        maximum_initial_scale_boundary_budget_residual=max_initial_scale_budget,
        finite_explosive_runs=sum(bool(r.get('finite_growth_warning')) for r in rows),
        supplementary_wall_runs_verified=len(geometry),supplementary_wall_failures_preserved=sum(r['success']=='False' for r in geometry),
        actual_input_mask_parity_stencils=parity['stencil_count'],
        supplementary_dt_half_controls_verified=len(timecontrols),
        new_test_source_sha256=sha(ROOT/'tests/test_four_method_comparison.py'),
        qualification='Tests and reference limitations are reported separately; integrity PASS is not an accuracy superiority claim.')
    fresh=OUT/'results/fresh_reproduction_sod'
    if fresh.exists():
        checks=[]
        for p in sorted(fresh.glob('run_*.json')):
            a=json.loads(p.read_text());b=json.loads((RAW/p.name).read_text())
            assert a['method']==b['method'] and a['success']==b['success']
            assert a['failure']==b['failure']
            with np.load(p.with_suffix('.npz')) as x,np.load((RAW/p.name).with_suffix('.npz')) as y:np.testing.assert_array_equal(x['final'],y['final'])
            checks.append(dict(status='REPRODUCED',case=a['case'],cells=a['cells'],method=a['method'],
                maximum_final_state_difference=0.,success_status_matches=True,first_invalid_stage_matches=True,
                source_file=str(p.relative_to(ROOT)),original_file=str((RAW/p.name).relative_to(ROOT))))
        assert len(checks)==10
        from experiments.four_method_comparison.scripts.collect_existing_results import csv_write
        csv_write(OUT/'results/fresh_reproduction_check.csv',checks)
        report['fresh_sod_reproductions_verified']=len(checks)
    (RAW/'package_verification.json').write_text(json.dumps(clean(report),indent=2)+'\n')
    active_logs={'logs/verification.log','logs/end_to_end_reproduction.log'}
    checksums={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='package_checksums.json' and str(p.relative_to(OUT)) not in active_logs}
    (RAW/'package_checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
