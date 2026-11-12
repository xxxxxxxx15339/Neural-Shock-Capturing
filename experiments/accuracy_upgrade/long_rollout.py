"""Frozen pilot checkpoints, longer Sod rollout and sampled exact-error curves."""
import argparse
from pathlib import Path
import json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from experiments.accuracy_upgrade import study
from experiments.accuracy_upgrade.study import OUT,evolve,write
from src.differentiable.model_io import load_checkpoint
from src.differentiable.reconstruction import make_reconstruction
from src.differentiable.v2_benchmark import state
from src.differentiable.euler_jax import spds_jax
from src.differentiable.evaluation import errors


def run(output):
    output.mkdir(exist_ok=False);variants=['historical_teno','historical_v2_soft25','fv_teno','safe_fv_teno']
    for seed in [3101,3102]:
        for v in ['A','B','C']:
            m=load_checkpoint(OUT/f'checkpoints/paired_pilot/{v}_seed{seed}_final.pkl')
            name=('fv_faceguard' if v=='C' else 'historical')+f'_pilot{v}_seed{seed}'
            study.EXTRA_RECONSTRUCTIONS[name]=make_reconstruction(m,'gated-soft',.25);variants.append(name)
    case=study.cases()[0].copy();case['time']=.2;rows=[]
    for n in [64,128]:
        u=state(case,n);speed=float(jnp.max(abs(spds_jax(jnp.asarray(u)))))
        dt=.2/int(np.ceil(.2*speed*1.5/(.25/n)))
        for v in variants:
            final,d,h,t,s=evolve(u,v,'transmissive',.2,1/n,dt,20)
            row=dict(cells=n,variant=v,case=case,**d)
            if d['success']:
                row.update(errors(final,state(case,n,.2)))
                row['error_over_time']=[dict(time=float(q),**errors(a,state(case,n,q))) for q,a in zip(t,s)]
            write(output/f'{n}_{v}.json',row);rows.append(row)
            print('long Sod',n,v,d['success'],row.get('l1'),flush=True)
    write(output/'summary.json',rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
