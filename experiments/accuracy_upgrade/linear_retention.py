"""Diagnose smooth selector order with all candidates retained; not a shock method."""
import argparse
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from experiments.accuracy_upgrade.study import averages,profile,write
from src.differentiable.accuracy_upgrade import flux_variant,state_stencils
from src.differentiable.euler_jax import flux_jax
from src.differentiable.teno5_jax import teno5_soft_reconstruct_jax,teno5_selection_values,DEFAULT_CUTOFF
from src.differentiable.evaluation import errors


def run(output):
    rec=lambda s:teno5_soft_reconstruct_jax(s,jnp.ones(s.shape[:-1]+(3,)))
    f=jax.jit(flux_variant(rec,'periodic','fv_state_flux'));rows=[]
    for kind in ['momentum','critical']:
        previous=None
        for n in [40,80,160,320]:
            u=jnp.asarray(averages(n,kind));faces=flux_jax(jnp.asarray(profile(np.arange(n+1)/n,kind)))
            exact=np.diff(np.asarray(faces),axis=0)*n;e=errors(np.asarray(f(u))*n,exact)
            plus,minus=state_stencils(u,'periodic');chi=teno5_selection_values(jnp.stack((plus,minus)))
            rows.append(dict(profile=kind,cells=n,**e,order=None if previous is None else np.log2(previous/e['l1']),
                teno_rejected_candidate_count=int(jnp.sum(chi<DEFAULT_CUTOFF)),
                diagnostic='always retain all candidates; not a shock method'))
            previous=e['l1']
    write(output,rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
