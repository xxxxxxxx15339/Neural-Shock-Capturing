"""Unchanged scalar methods on analytic FV references; no scalar solver claims."""
import argparse
from pathlib import Path
import time
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from src.differentiable.v2_benchmark import exact_module
from src.differentiable.model_io import load_checkpoint
from src.differentiable.reconstruction import make_reconstruction
from src.differentiable.diagnostics import diagnostic_rollout
from src.differentiable.evaluation import errors
from experiments.accuracy_upgrade.study import write


def run(output):
    output.mkdir(exist_ok=False);ex=exact_module();rows=[]
    model=load_checkpoint('models/v2/scalar_gated_dataset_v2_seed2026.pkl')
    p0=load_checkpoint('models/nn_teno5_unrolled_jax_params_run6_learnedlambda.pkl')
    methods=dict(teno5=make_reconstruction(None,'teno5'),v2_soft25=make_reconstruction(model,'gated-soft',.25),
                 p0_learned=make_reconstruction(p0,'learned-blend'))
    for kind in ['sine','fourier','gaussian','square','burgers_smooth','burgers_hat']:
        system='burgers' if kind.startswith('burgers') else 'advection'
        for n in [64,128,256]:
            end=.2
            if kind=='sine':exact=lambda t:ex.exact_fourier(n,.3,[[.2,1,.1]],t,.83)
            elif kind=='fourier':exact=lambda t:ex.exact_fourier(n,.3,[[.2,1,.1],[.1,3,.3]],t,.83)
            elif kind=='gaussian':exact=lambda t:ex.exact_gaussian(n,.4,.07,1.,.1,t,.83)
            elif kind=='square':exact=lambda t:ex.exact_square(n,.17,.56,1.1,-.1,.83*t)
            elif kind=='burgers_smooth':exact=lambda t:ex.exact_burgers_smooth(n,.45,.2,1,.1,t)
            else:exact=lambda t:ex.exact_burgers_hat(n,.1,.9,.2,.6,t)
            u=exact(0.);steps=int(np.ceil(end*1.2/(.25/n)));dt=end/steps;reference=exact(end)
            for name,rec in methods.items():
                def execute():return diagnostic_rollout(u,rec,dt,1/n,steps,system,'periodic',wave_speed=.83,
                                   alpha='dynamic' if system=='burgers' else .83,return_device=True,record_states=True)
                jit=jax.jit(execute);start=time.perf_counter();compiled=jit.lower().compile();compile_time=time.perf_counter()-start
                final,history=compiled();final.block_until_ready();timings=[]
                for _ in range(3):
                    start=time.perf_counter();final,history=compiled();final.block_until_ready();timings.append(time.perf_counter()-start)
                final=np.asarray(final);history=jax.tree.map(np.asarray,history)
                row=dict(case=kind,system=system,cells=n,method=name,time=end,dt=dt,steps=steps,**errors(final,reference),
                    tv=float(np.sum(abs(final-np.roll(final,1)))),
                    overshoot=max(0.,float(final.max()-reference.max())),undershoot=max(0.,float(reference.min()-final.min())),
                    compile_s=compile_time,warm_simulation_s=float(np.median(timings)),runtime_repeats=timings,
                    conservation_residual=float(np.sum(final-u)/n),valid=bool(np.isfinite(final).all()),
                    error_over_time=[dict(time=float((i+1)*dt),**errors(v,exact((i+1)*dt)))
                                    for i,v in enumerate(history['state']) if i%(max(1,steps//10))==0])
                if kind=='burgers_hat':
                    # Declared right-going shock before rarefaction/shock interaction.
                    x=(np.arange(n)+.5)/n;ix=np.flatnonzero(x>.5)
                    shock=float(x[ix[np.argmin(np.diff(final,append=final[-1])[ix])]])
                    row.update(shock_position=shock,exact_shock_position=.6+.5*end,
                        shock_position_error=abs(shock-(.6+.5*end)),measured_average_shock_speed=(shock-.6)/end,
                        exact_shock_speed=.5)
                    band=(x>.5)&(x<.85);row['shock_width_cells']=int(np.sum(band&(final>.18)&(final<.82)))
                write(output/f'{kind}_{n}_{name}.json',row)
                np.savez_compressed(output/f'{kind}_{n}_{name}.npz',initial=u,reference=reference,final=final)
                rows.append(row);print('scalar',kind,n,name,row['l1'],flush=True)
    for r in rows:
        b=next(x for x in rows if x['case']==r['case'] and x['cells']==r['cells'] and x['method']=='teno5')
        r['l1_ratio']=r['l1']/b['l1']
    write(output/'summary.json',rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
