"""General smooth Euler finite-volume consistency diagnostic and explicit variant."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from .euler_jax import flux_jax,get_euler_flux_jax,get_euler_state_flux_jax
from .evaluation import errors


def nonlinear_state(x):
    rho=1+.1*np.sin(2*np.pi*x);v=.4+.1*np.cos(2*np.pi*x);p=1+.1*np.sin(4*np.pi*x)
    return np.stack((rho,rho*v,p/.4+.5*rho*v*v),axis=-1)


def nonlinear_spatial(n,rec,flux_factory=get_euler_flux_jax):
    z,w=np.polynomial.legendre.leggauss(16)
    xs=(np.arange(n)[:,None]+.5+.5*z)/n
    avg=.5*np.einsum('j,ijk->ik',w,nonlinear_state(xs))
    faces=np.asarray(flux_jax(jnp.asarray(nonlinear_state(np.arange(n+1)/n))))
    exact=np.diff(faces,axis=0)*n
    num=np.asarray(jax.jit(flux_factory(rec,'periodic'))(jnp.asarray(avg)))*n
    return dict(cells=n,**errors(num,exact))


def run(models,output):
    from .artifacts import write_json
    from .reconstruction import make_reconstruction
    from .v2_studies import add_orders
    rows=[]
    for solver,func in [('historical-split-flux',get_euler_flux_jax),('experimental-fv-state-flux',get_euler_state_flux_jax)]:
        for method in ['teno5','gated-soft','gated-hard']:
            rec=make_reconstruction(models['euler'],method,.25)
            data=[nonlinear_spatial(n,rec,func) for n in [40,80,160,320,640]];add_orders(data)
            rows.append(dict(solver=solver,method=method,rows=data))
    write_json(output/'nonlinear_euler.json',dict(definition='instantaneous exact cell-average flux divergence; no temporal error',
               initial='rho=1+.1sin(2pi x); v=.4+.1cos(2pi x); p=1+.1sin(4pi x)',
               limitation='state-flux variant has only a smooth consistency diagnostic; no shock/positivity validation or retraining',results=rows))
