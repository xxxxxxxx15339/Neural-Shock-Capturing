"""Calculate consistent errors and improvements from saved successful trajectories."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts import methods as m
from experiments.four_method_comparison.scripts.run_matched_benchmarks import reference,exact_module,write
from experiments.four_method_comparison.scripts.collect_existing_results import csv_write
import numpy as np

OUT=m.OUT;RAW=OUT/'results/raw'

def norms(a,b):
    d=np.abs(a-b)
    out=dict(l1=float(np.mean(d)),l2=float(np.sqrt(np.mean(d*d))),linf=float(np.max(d)))
    if a.ndim==2:
        out.update(density_l1=float(np.mean(d[:,0])),momentum_l1=float(np.mean(d[:,1])),energy_l1=float(np.mean(d[:,2])))
    return out

def tv(a,periodic):return float(np.sum(np.abs(np.diff(a)))+(abs(a[-1]-a[0]) if periodic else 0.))

def crossing(x,u,value,center,length,radius=None):
    radius=.12*length if radius is None else radius
    edges=np.flatnonzero((u[:-1]-value)*(u[1:]-value)<=0)
    candidates=[]
    for i in edges:
        if u[i+1]!=u[i]:
            p=x[i]+(x[i+1]-x[i])*(value-u[i])/(u[i+1]-u[i])
            if abs(p-center)<radius:candidates.append(p)
    return min(candidates,key=lambda p:abs(p-center)) if candidates else None

def wave_metrics(x,u,center,left,right,length,radius=None):
    radius=.12*length if radius is None else radius
    edges=.5*(x[1:]+x[:-1]);mask=np.abs(edges-center)<radius
    out={}
    if mask.any():
        indices=np.flatnonzero(mask);k=indices[np.argmax(abs(np.diff(u))[mask])]
        out.update(position=float(edges[k]),position_error=float(abs(edges[k]-center)))
    low=crossing(x,u,right+.1*(left-right),center,length,radius)
    high=crossing(x,u,right+.9*(left-right),center,length,radius)
    out['width_10_90']=None if low is None or high is None else float(abs(high-low))
    return out

def euler_wave_metrics(x,rho,ex,jump,end,length):
    """Isolate known Riemann waves before measuring jumps/crossings.

    A broad contact window can accidentally select the stronger nearby shock.
    Restrict each radius to45% of its nearest exact wave separation, independent
    of the tested method or error. Unresolved crossings remain unavailable.
    """
    out={}
    if ex.right_shock:
        center=jump+ex.R[-1]*end
        radius=min(.12*length,.45*(ex.R[-1]-ex.vel)*end)
        for key,value in wave_metrics(x,rho,center,ex.rsr,ex.rp[0],length,radius).items():out['shock_'+key]=value
    center=jump+ex.vel*end
    radius=min(.12*length,.45*min(ex.vel-ex.L[1],ex.R[0]-ex.vel)*end)
    for key,value in wave_metrics(x,rho,center,ex.rsl,ex.rsr,length,radius).items():out['contact_'+key]=value
    return out

def primitives(u):
    rho=u[:,0];v=u[:,1]/rho;p=.4*(u[:,2]-.5*u[:,1]**2/rho)
    return rho,v,p

def corr(a,b):
    return None if np.std(a)<1e-14 or np.std(b)<1e-14 else float(np.corrcoef(a,b)[0,1])

def load_rows():
    manifest=json.loads((OUT/'configs/benchmark_manifest.json').read_text())
    cases={c['id']:c for c in manifest['cases']};rows=[];stability=[];runtime=[];ref_report=[];lambda_report=[]
    finest=max(manifest['shu_reference']['grids'])
    ref_paths=[RAW/f'reference_shu_{n}_dt1' for n in manifest['shu_reference']['grids']]+[RAW/f'reference_shu_{finest}_dt2']
    refs=[]
    for base in ref_paths:
        if base.with_suffix('.json').exists():
            d=json.loads(base.with_suffix('.json').read_text())
            if d['success']:
                with np.load(base.with_suffix('.npz')) as a:refs.append((base.name,np.array(a['x']),np.array(a['final'])))
    def interp(x,ref):return np.column_stack([np.interp(x,ref[1],ref[2][:,k]) for k in range(3)])
    for file in sorted(RAW.glob('run_*.json')):
        row=json.loads(file.read_text());case=cases[row['case']]
        with np.load(file.with_suffix('.npz')) as z:
            arr={k:np.array(z[k]) for k in z.files}
        x,u,ref=arr['x'],arr['final'],arr['reference']
        if case['id']=='shu_osher' and refs:
            ref=interp(x,refs[-1]);arr['reference']=ref
        row['source_file']=str(file.relative_to(ROOT));row['trajectory_file']=str(file.with_suffix('.npz').relative_to(ROOT))
        row['invalid_stage_count']=0 if row['success'] else 1
        row['attempted_actual_stages']=3*row['accepted_steps']+(row['failure']['stage'] if row['failure'] else 0)
        row['failed_stage_padding']='Fixed-shape diagnostic slots after first_invalid_stage are unexecuted padding, not additional invalid stages'
        model_key=('learned_'+case['physics']) if row['method'] in ('learned_head','head_backbone_only','head_fixed_blend') else row['method']
        selected=manifest['models'].get(model_key)
        row['checkpoint_path']=selected['path'] if selected else 'none; classical deterministic source'
        row['checkpoint_sha256']=selected['sha256'] if selected else None
        row['conservation_max_abs']=float(np.max(np.abs(row['conservation_residual'])))
        scale=np.maximum(1.,np.maximum(abs(row['dx']*np.sum(arr['initial'],axis=0)),abs(np.asarray(row['boundary_exchange']))))
        row['conservation_relative']=float(np.max(abs(np.asarray(row['conservation_residual']))/scale))
        if case['physics']=='scalar':
            row['maximum_absolute_value']=float(np.max(abs(u)))
            row['amplification_factor']=row['maximum_absolute_value']/max(1.,float(np.max(abs(arr['initial']))))
            row['finite_growth_warning']=row['amplification_factor']>10
        if row['success'] and ref.size:
            row.update(norms(u,ref));q=u[:,0] if u.ndim==2 else u;r=ref[:,0] if ref.ndim==2 else ref
            row.update(total_variation=tv(q,case['boundary']=='periodic'),
                overshoot=max(0.,float(np.max(q)-np.max(r))),undershoot=max(0.,float(np.min(r)-np.min(q))))
            if u.ndim==2:
                _,vel,p=primitives(u);_,vr,pr=primitives(ref)
                row.update(velocity_l1=float(np.mean(abs(vel-vr))),pressure_l1=float(np.mean(abs(p-pr))),
                    final_min_density=float(np.min(u[:,0])),final_min_pressure=float(np.min(p)))
                if 'left' in case:
                    ex=exact_module().ExactEuler(case['left'],case['right'],case['jump'])
                    row.update(euler_wave_metrics(x,q,ex,case['jump'],case['time'],case['length']))
            elif case['id'] in ('burgers_hat','burgers_post'):
                center=.6+.5*case['time'] if case['id']=='burgers_hat' else (.5+.5*case['time'])%1
                states=(1.,0.) if case['id']=='burgers_hat' else (reference(case,np.array([center-1e-10]),case['time'])[0],reference(case,np.array([center+1e-10]),case['time'])[0])
                for k,v in wave_metrics(x,u,center,*states,case['length']).items():row['shock_'+k]=v
                row['exact_shock_speed']=.5
                if row.get('shock_position') is not None:
                    origin=.6 if case['id']=='burgers_hat' else .5
                    row['measured_shock_speed']=(row['shock_position']-origin)/case['time']
            if case['id']=='shu_osher':
                lo,hi=case['oscillatory_region'];mask=(x>=lo)&(x<=hi)
                row['regional_density_l1']=float(np.mean(abs(u[mask,0]-ref[mask,0])))
                row['regional_mean_l1']=float(np.mean(abs(u[mask]-ref[mask])))
                row['regional_density_amplitude']=float(np.ptp(u[mask,0]))
                sensitivity=[]
                for name,xr,ur in refs:
                    alt=interp(x,(name,xr,ur));sensitivity.append(dict(reference=name,**norms(u,alt)))
                row['reference_sensitivity']=sensitivity
                row['reference_grid_delta_l1']=norms(interp(x,refs[-2]),interp(x,refs[-3]))['l1'] if len(refs)>=3 else None
                row['reference_dt_delta_l1']=norms(interp(x,refs[-1]),interp(x,refs[-2]))['l1'] if len(refs)>=2 else None
                row['reference_warning']=row['reference_grid_delta_l1']>manifest['shu_reference']['warning_fraction']*row['l1'] if row['reference_grid_delta_l1'] is not None else True
            if case['id']!='shu_osher':
                row['error_over_time']=[dict(time=float(t),**norms(v,reference(case,x,float(t)))) for t,v in zip(arr['times'],arr['states'])]
        elif row['success']:
            q=u[:,0] if u.ndim==2 else u;row['total_variation']=tv(q,case['boundary']=='periodic')
        # Metrics are never computed at the requested endpoint for a failed trajectory.
        if arr.get('lambda_final',np.array([])).size:
            lam=arr['lambda_final'];mean=lam.mean(axis=(1,2));q=u[:,0]
            g=np.abs(np.gradient(q,x));smooth=g<np.quantile(g,.5);shock=g>=np.quantile(g,.9)
            stats=dict(case=case['id'],cells=row['cells'],method='learned_head',
                mean=float(lam.mean()),minimum=float(lam.min()),maximum=float(lam.max()),std=float(lam.std()),
                percentiles=np.percentile(lam,[0,5,25,50,75,95,100]).tolist(),
                mean_smooth=float(mean[smooth].mean()) if smooth.any() else None,
                mean_shock=float(mean[shock].mean()),correlation_density_gradient=corr(mean,g),
                interpretation='descriptive only; gradient masks are median and top-decile density gradient')
            p,mn=m.euler_split_flux_stencils(m.jnp.asarray(u),case['boundary'])
            pure=m.reconstruction('head_backbone_only','euler')
            correction=np.stack((np.asarray(pure(p)-m.teno5_reconstruct_jax(p)),np.asarray(pure(mn)-m.teno5_reconstruct_jax(mn))),axis=-1)
            stats['correlation_unmixed_characteristic_correction']=corr(lam.ravel(),abs(correction).ravel())
            base=RAW/f'run_{case["id"]}_{row["cells"]}_teno5.npz'
            if base.exists() and ref.size:
                with np.load(base) as a:
                    change=np.mean(abs(a['final']-ref)-abs(u-ref),axis=1)
                stats['correlation_local_error_reduction']=corr(mean,change)
            lambda_report.append(stats)
        if row['rollout_repeats_seconds']:
            vals=np.array(row['rollout_repeats_seconds']);single=np.array(row['warm_step_repeats_seconds'])
            row['runtime_to_stop_median_s']=float(np.median(vals))
            row['runtime_to_stop_iqr_s']=float(np.percentile(vals,75)-np.percentile(vals,25))
            row['runtime_median_s']=float(np.median(vals)) if row['success'] else None
            row['runtime_iqr_s']=row['runtime_to_stop_iqr_s'] if row['success'] else None
            row['warm_step_median_s']=float(np.median(single));row['warm_step_iqr_s']=float(np.percentile(single,75)-np.percentile(single,25))
        runtime.append({k:row.get(k) for k in ('case','cells','method','success','accepted_steps','planned_steps','compile_step_seconds','compile_rollout_seconds',
            'first_step_seconds','warm_step_median_s','warm_step_iqr_s','runtime_median_s','runtime_iqr_s','runtime_to_stop_median_s','runtime_to_stop_iqr_s','instrumented_rollout_seconds')})
        stability.append(dict(case=row['case'],cells=row['cells'],method=row['method'],success=row['success'],actual_time=row['actual_time'],
            first_invalid_step=None if row['failure'] is None else row['failure']['step'],first_invalid_stage=None if row['failure'] is None else row['failure']['stage'],
            raw_min_density=row['raw_min_density'],raw_min_pressure=row['raw_min_pressure'],conservation_residual=json.dumps(row['conservation_residual']),
            boundary_exchange=json.dumps(row['boundary_exchange']),conservation_max_abs=row['conservation_max_abs'],
            invalid_stage_count=row['invalid_stage_count'],attempted_actual_stages=row['attempted_actual_stages'],
            conservation_relative=row['conservation_relative'],finite_growth_warning=row.get('finite_growth_warning',False),
            maximum_absolute_value=row.get('maximum_absolute_value'),repair_count=0,retry_count=0))
        rows.append(row)
    for n in (128,256):
        x=np.arange(n)*10/n
        for i in range(1,len(refs)):
            a=interp(x,refs[i-1]);b=interp(x,refs[i]);ref_report.append(dict(cells=n,coarser=refs[i-1][0],finer=refs[i][0],**norms(a,b)))
    write(RAW/'reference_sensitivity.json',ref_report);write(RAW/'lambda_statistics.json',lambda_report)
    geometry_file=OUT/'results/wall_geometry_control.csv'
    if geometry_file.exists():
        for control in csv.DictReader(geometry_file.open()):
            r=json.loads((ROOT/control['source_file']).read_text())
            stability.append(dict(case=r['case'],cells=r['cells'],method=r['method'],success=r['success'],actual_time=r['actual_time'],
                first_invalid_step=None if r['failure'] is None else r['failure']['step'],
                first_invalid_stage=None if r['failure'] is None else r['failure']['stage'],
                raw_min_density=r['raw_min_density'],raw_min_pressure=r['raw_min_pressure'],
                conservation_residual=json.dumps(r['conservation_residual']),boundary_exchange=json.dumps(r['boundary_exchange']),
                conservation_max_abs=float(np.max(abs(np.asarray(r['conservation_residual'])))),repair_count=0,retry_count=0))
            runtime.append(dict(case=r['case'],cells=r['cells'],method=r['method'],success=r['success'],
                accepted_steps=r['accepted_steps'],planned_steps=r['planned_steps'],timing_status='UNAVAILABLE; supplementary stability control, no performance claim'))
    csv_write(OUT/'results/stability_results.csv',stability);csv_write(OUT/'results/runtime_results.csv',runtime)
    return manifest,rows

def main():
    manifest,rows=load_rows();write(RAW/'evaluated_rows.json',rows)
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
    csv_write(OUT/'results/matched_metrics_long.csv',flat)
    master=[];improvements=[];ablation=[]
    keys=['density_l1','momentum_l1','energy_l1','l1','l2','linf','regional_density_l1','regional_mean_l1','velocity_l1','pressure_l1',
          'total_variation','overshoot','undershoot','shock_position_error','shock_width_10_90','contact_width_10_90','runtime_median_s','conservation_max_abs']
    for case in manifest['cases']:
        for n in case['grids']:
            family={r['method']:r for r in rows if r['case']==case['id'] and r['cells']==n}
            for metric in keys:
                if not any(metric in r and r[metric] is not None for r in family.values()):continue
                record=dict(benchmark=case['id'],cells=n,final_time=case['time'],metric=metric,status='NEW MATCHED EVALUATION')
                for method in manifest['methods']:
                    r=family.get(method,{})
                    record[method]=r.get(metric) if r.get('success') else None
                    record[method+'_status']='SUCCESS' if r.get('success') else ('FAILED' if r else 'UNAVAILABLE')
                master.append(record)
                if metric not in ('l1','density_l1','momentum_l1','energy_l1','regional_density_l1'):continue
                def gain(a,b):
                    va=family.get(a,{}).get(metric);vb=family.get(b,{}).get(metric)
                    return 100*(1-va/vb) if va is not None and vb is not None and vb>0 else None
                improvements.append(dict(benchmark=case['id'],cells=n,metric=metric,status='NEW MATCHED EVALUATION',
                    neno6_vs_teno5_percent=gain('neno6','teno5'),nn_vs_teno5_percent=gain('nn_teno5','teno5'),
                    v2_vs_teno5_percent=gain('learned_head','teno5'),v2_vs_nn_percent=gain('learned_head','nn_teno5'),
                    interpretation='V2-vs-NN is whole-method difference; different backbones'))
                if 'head_backbone_only' in family:
                    ablation.append(dict(benchmark=case['id'],cells=n,metric=metric,
                        pure=family['head_backbone_only'].get(metric),fixed=family['head_fixed_blend'].get(metric),learned=family['learned_head'].get(metric),
                        learned_vs_pure_percent=gain('learned_head','head_backbone_only'),learned_vs_fixed_percent=gain('learned_head','head_fixed_blend'),
                        pure_success=family['head_backbone_only']['success'],fixed_success=family['head_fixed_blend']['success'],learned_success=family['learned_head']['success']))
    stress={r['method']:[v for v in rows if v['method']==r['method'] and next(c for c in manifest['cases'] if c['id']==v['case']).get('robustness')] for r in rows if r['method'] in manifest['methods']}
    master.append(dict(benchmark='declared_robustness_suite',cells='64,128',metric='successful_cases',status='NEW MATCHED EVALUATION',
        **{method:sum(v['success'] for v in rs) for method,rs in stress.items()},
        **{method+'_status':f'{sum(v["success"] for v in rs)}/{len(rs)} completed; see separate wall-alignment controls' for method,rs in stress.items()}))
    csv_write(OUT/'results/master_results.csv',master);csv_write(OUT/'results/percentage_improvements.csv',improvements)
    csv_write(OUT/'results/same_backbone_ablation.csv',ablation)
    provenance=[]
    for r in rows:
        for key in keys:
            if r.get(key) is not None:provenance.append(dict(status='NEW MATCHED EVALUATION',benchmark=r['case'],cells=r['cells'],method=r['method'],
                metric=key,value=r[key],source_file=r['source_file'],trajectory=r['trajectory_file'],
                source_sha256=hashlib.sha256((ROOT/r['source_file']).read_bytes()).hexdigest(),
                calculation='scripts/calculate_metrics.py; full values in results/raw/evaluated_rows.json'))
    for r in csv.DictReader((OUT/'results/archive_reproduction.csv').open()):
        provenance.append(dict(status=r['status'],metric=r['json_pointer'],value=r['recomputed'],
            source_file=r['reproduced'],source_sha256=hashlib.sha256((ROOT/r['reproduced']).read_bytes()).hexdigest(),
            calculation='Original entry point; compared against '+r['original']+'; delta='+r['absolute_difference']))
    provenance.append(dict(status='ARCHIVED',source_file=str((RAW/'archived_metrics.csv').relative_to(ROOT)),
        source_sha256=hashlib.sha256((RAW/'archived_metrics.csv').read_bytes()).hexdigest(),
        calculation='Complete archived numerical evidence catalog; each row has original JSON pointer and source hash; not merged with new matched tables'))
    provenance.append(dict(status='LITERATURE ONLY',source_file='https://arxiv.org/pdf/2207.08500',metric='NENO6 optimized background accuracy constraint',value=4,
        calculation='Section4.1: optimized six-point weights target fourth-order background accuracy. Not a matched experiment.'))
    geometry_file=OUT/'results/wall_geometry_control.csv'
    if geometry_file.exists():
        for r in csv.DictReader(geometry_file.open()):
            for key in ('success','raw_min_density','raw_min_pressure','actual_time'):
                provenance.append(dict(status='NEW MATCHED EVALUATION',benchmark=r['case'],cells=r['cells'],method=r['method'],
                    metric=key,value=r[key],source_file=r['source_file'],source_sha256=hashlib.sha256((ROOT/r['source_file']).read_bytes()).hexdigest(),
                    calculation='Supplementary predeclared wall-aligned control; no endpoint error/reference claimed'))
    time_file=OUT/'results/smooth_time_control.csv'
    if time_file.exists():
        for r in csv.DictReader(time_file.open()):
            for key in ('dt_half_l1','temporal_change_l1','temporal_change_fraction'):
                provenance.append(dict(status='NEW MATCHED EVALUATION',benchmark=r['case'],cells=r['cells'],method=r['method'],
                    metric=key,value=r[key],source_file=r['source_file'],source_sha256=hashlib.sha256((ROOT/r['source_file']).read_bytes()).hexdigest(),
                    calculation='Supplementary predeclared finest-grid dt/2 sensitivity control'))
    csv_write(OUT/'RESULT_PROVENANCE.csv',provenance)
    print('Measured',len(rows),'rollouts;',sum(not r['success'] for r in rows),'failures; no failed endpoint errors.')

if __name__=='__main__':main()
