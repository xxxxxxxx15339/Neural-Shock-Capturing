"""Declare cases and model identities before evaluating their matched results."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/four_method_comparison'

def main():
    path=OUT/'configs/benchmark_manifest.json'
    if path.exists():raise FileExistsError('Frozen manifest already exists; do not select models from test results')
    model_paths={
        'nn_teno5':'models/nn_teno5.h5',
        'neno6':'experiments/four_method_comparison/vendor/NNENO/mytrainmodel1',
        'learned_euler':'models/nn_teno5euler_unrolled_jax_params_curriculum_phase21_richgate.pkl',
        'learned_scalar':'models/nn_teno5_unrolled_jax_params_run6_learnedlambda.pkl'}
    cases=[]
    def add(id,physics,equation,time,length=1.,boundary=None,grids=(128,256),**kw):
        cases.append(dict(id=id,physics=physics,equation=equation,time=time,length=length,
            boundary=boundary or ('periodic' if physics=='scalar' else 'transmissive'),grids=list(grids),**kw))
    add('sod','euler','euler',.2,left=[1.,0.,1.],right=[.125,0.,.1],jump=.5,reference='exact_riemann')
    add('lax','euler','euler',.1,left=[.445,.698,3.528],right=[.5,0.,.571],jump=.5,reference='exact_riemann')
    add('shu_osher','euler','euler',1.8,length=10.,reference='teno5_resolution_ladder',oscillatory_region=[5.5,7.2])
    add('burgers_pre','scalar','burgers',.5,reference='exact_entropy_sine')
    add('burgers_post','scalar','burgers',.9,reference='exact_entropy_sine',shock_formation_time=2/3.141592653589793)
    add('burgers_hat','scalar','burgers',.2,reference='exact_entropy_top_hat')
    add('advection_square','scalar','advection',1.,reference='exact_periodic_transport')
    add('advection_smooth','scalar','advection',.2,grids=(64,128,256),reference='exact_periodic_transport',smooth=True)
    for id,left,right,time in [
        ('two_rarefactions',[1.,-2.,.4],[1.,2.,.4],.08),
        ('strong_contrast',[10.,0.,100.],[.01,0.,.01],.012),
        ('near_vacuum',[1e-5,-.01,1e-8],[1e-5,.01,1e-8],.1)]:
        add(id,'euler','euler',time,grids=(64,128),left=left,right=right,jump=.5,reference='exact_riemann',robustness=True)
    add('blast_smoke','euler','euler',.001,boundary='reflective-wall',grids=(64,128),reference='none_stability_only',robustness=True)
    add('blast_full','euler','euler',.038,boundary='reflective-wall',grids=(64,128),reference='none_stability_only',robustness=True)
    manifest=dict(schema_version=1,frozen_utc=datetime.now(timezone.utc).isoformat(),
        declaration='Frozen after architecture audit, before any matched numerical evaluation; no retraining/tuning.',
        methods=['teno5','neno6','nn_teno5','learned_head'],
        models={k:dict(path=v,sha256=hashlib.sha256((ROOT/v).read_bytes()).hexdigest()) for k,v in model_paths.items()},
        sampling='point-i-dx',precision='float64 states and JAX heads; original Keras classifier float32 verified adapter',
        euler_flux='historical characteristic split-flux; adjacent-state LF alpha; Roe basis',
        scalar_flux='historical state reconstruction + LF numerical flux, alpha=1',
        time_integrator='SSPRK3',cfl=.25,initial_wave_speed_safety=1.3,
        timestep='fixed dt=T/ceil(T*initial_speed*safety/(CFL*dx)); matched all methods',
        smooth_timestep='dt<=.15*dx**(5/3), then exact integer endpoint',
        repair_policy='none; stop at first invalid actual SSPRK3 stage; no retry or fallback',
        fixed_lambda=.04742587317756678,
        ablation=dict(cases=['sod','lax','shu_osher','burgers_pre','burgers_post','burgers_hat','advection_square'],
                      grids=[128],methods=['head_backbone_only','head_fixed_blend','learned_head']),
        shu_reference=dict(grids=[512,1024,2048],finest_dt_refinements=[1,2],
                           warning_fraction=.1,description='Resolution and dt sensitivity measured; no exact reference claimed'),
        runtime=dict(full_rollout_repeats=3,warm_step_repeats=7,synchronized=True,
                     summary='median, IQR; compilation separate; instrumented trajectory separate'),
        metrics=dict(l1='mean absolute point error',euler_mean='mean of rho,momentum,energy L1',
                     total_variation='sum abs adjacent differences, periodic closing difference included only for periodic BC',
                     failed_error=None,improvement_percent='100*(1-E_method/E_baseline)',
                     conservation='dx*sum(final-initial)+SSPRK3-weighted time integral of right-minus-left boundary flux'),
        cases=cases,selection='Euler phase21 rich head: richest trained finite validated architecture, selected before final matched tests; scalar run6 saved learned head',
        role='historically exposed benchmarks; replication and frozen comparative evaluation, not untouched holdout')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    (path.with_suffix('.sha256')).write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  benchmark_manifest.json\n')
    print(path)

if __name__=='__main__':main()
