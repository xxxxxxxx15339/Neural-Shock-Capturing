"""Read-only adapters. The existing TENO5/Euler implementations are unchanged."""
from functools import lru_cache
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from src.differentiable.teno5_jax import teno5_reconstruct_jax,teno5_soft_reconstruct_jax
from src.differentiable.euler_jax import (get_euler_flux_jax,flux_jax,spds_jax,
    roe_eigenbasis_jax,project_to_characteristic_jax,project_to_conservative_jax,
    euler_split_flux_stencils)
from src.differentiable.model_io import load_checkpoint
from src.differentiable.reconstruction import make_reconstruction,INITIAL_FIXED_BLEND
from src.differentiable.teno5_jax_toy import toy_keep_weight,toy_lambda,toy_lambda_v2

OUT=ROOT/'experiments/four_method_comparison'
HEADS={'euler':'models/nn_teno5euler_unrolled_jax_params_curriculum_phase21_richgate.pkl',
       'scalar':'models/nn_teno5_unrolled_jax_params_run6_learnedlambda.pkl'}
THRESHOLDS=jnp.array([.6599999999999997,.4999999999999998,.6899999999999997],dtype=jnp.float64)
MAIN=('teno5','neno6','nn_teno5','learned_head')
NAMES={'teno5':'TENO5','neno6':'NENO6','nn_teno5':'NN-TENO5',
       'learned_head':'NN-TENO5-V2 (Learned Mixing Head)',
       'head_backbone_only':'Same backbone: pure NN','head_fixed_blend':'Same backbone: fixed blend'}

def dense(values,layers):
    x=values
    for w,b in layers[:-1]:x=jax.nn.relu(x@w+b)
    w,b=layers[-1]
    return jax.nn.sigmoid(x@w+b)

@lru_cache(None)
def original_layers():
    with np.load(OUT/'results/raw/original_keras_export.npz') as z:
        return tuple((np.asarray(z[f'a{2*i}']),np.asarray(z[f'a{2*i+1}'])) for i in range(5))

def original_probabilities(s):
    minimum=jnp.min(s,axis=-1,keepdims=True)
    span=jnp.max(s,axis=-1,keepdims=True)-minimum
    x=((s-minimum)/jnp.where(span>0,span,1.)).astype(jnp.float32)
    return dense(x,original_layers())

def original_reconstruct(s):
    return teno5_soft_reconstruct_jax(s,(original_probabilities(s)<THRESHOLDS).astype(s.dtype))

@lru_cache(None)
def neno_layers():
    with np.load(OUT/'results/raw/neno6_author_export.npz') as z:
        return tuple((np.asarray(z[f'w{i}']),np.asarray(z[f'b{i}'])) for i in range(5))

def neno_probabilities(s):
    return dense(s/jnp.maximum(1.,jnp.max(jnp.abs(s),axis=-1,keepdims=True)),neno_layers())

@lru_cache(None)
def neno_mask_weights():
    # Only sixteen masks. CPU IEEE float32 divisions reproduce the author's
    # normalized weights exactly, avoiding XLA reciprocal-division rounding.
    coefficients=np.array([.4294317718960898,.1727270875843552,.0855682281039113,.312272912415645],np.float32)
    table=np.empty((16,4),np.float32)
    for code in range(16):
        active=coefficients*np.array([(code>>i)&1 for i in range(4)],np.float32)
        den=((active[0]+active[1])+active[2])+active[3]
        with np.errstate(invalid='ignore'):table[code]=active/den
    return table

def neno_reconstruct(s):
    """Author coefficients/order, 0.5 discard test and zero-denominator behavior."""
    if s.shape[-1]!=6:raise ValueError('NENO6 requires exactly six values')
    a,b,c,d,e,f=[s[...,i] for i in range(6)]
    variations=jnp.stack((-b/6+5*c/6+2*d/6-c,
                          2*c/6+5*d/6-e/6-c,
                          2*a/6-7*b/6+11*c/6-c,
                          3*c/12+13*d/12-5*e/12+f/12-c),axis=-1)
    # Author PyTorch multiplies int32 masks by Python floats, producing float32
    # weights even though its network/states are double. Preserve that behavior.
    keep=(neno_probabilities(s)<.5).astype(jnp.int32)
    code=keep[...,0]+2*keep[...,1]+4*keep[...,2]+8*keep[...,3]
    w=jnp.asarray(neno_mask_weights())[code]
    # The author method has no all-discard fallback; such a failure remains visible.
    result=c
    for i in range(4):result=result+w[...,i]*variations[...,i]
    return result

@lru_cache(None)
def reconstruction(method,physics):
    if method=='teno5':return teno5_reconstruct_jax
    if method=='nn_teno5':return original_reconstruct
    if method=='neno6':return neno_reconstruct
    c=load_checkpoint(HEADS[physics])
    policy={'learned_head':'learned-blend','head_backbone_only':'neural-only','head_fixed_blend':'fixed-blend'}[method]
    return make_reconstruction(c,policy,fixed_lambda=INITIAL_FIXED_BLEND)

def gather(u,offsets,boundary):
    indices=jnp.arange(u.shape[0])[:,None]+jnp.asarray(offsets)[None,:]
    indices=indices%u.shape[0] if boundary=='periodic' else jnp.clip(indices,0,u.shape[0]-1)
    return u[indices]

def neno_split(u,boundary):
    left,right=roe_eigenbasis_jax(u,boundary)
    f=flux_jax(u);c=jnp.max(jnp.abs(spds_jax(u)),axis=1)
    cr=jnp.roll(c,-1) if boundary=='periodic' else jnp.concatenate((c[1:],c[-1:]))
    alpha=jnp.maximum(c,cr)[:,None,None]
    def project(v,offsets):
        return project_to_characteristic_jax(jnp.transpose(gather(v,offsets,boundary),(0,2,1)),left)
    plus=(-2,-1,0,1,2,3);minus=(3,2,1,0,-1,-2)
    return .5*(project(f,plus)+alpha*project(u,plus)),.5*(project(f,minus)-alpha*project(u,minus)),right

def neno_euler_flux(u,boundary):
    n=u.shape[0]
    if boundary=='reflective-wall':
        sign=jnp.array([1.,-1.,1.])
        padded=jnp.concatenate((jnp.flip(u[:3],axis=0)*sign,u,jnp.flip(u[-3:],axis=0)*sign))
        p,m,b=neno_split(padded,'transmissive')
        faces=project_to_conservative_jax(neno_reconstruct(p)+neno_reconstruct(m),b)
        faces=faces.at[2,0].set(0.).at[2,2].set(0.).at[n+2,0].set(0.).at[n+2,2].set(0.)
        return faces[3:n+3]-faces[2:n+2],(faces[2],faces[n+2])
    p,m,b=neno_split(u,boundary)
    faces=project_to_conservative_jax(neno_reconstruct(p)+neno_reconstruct(m),b)
    left=faces[-1] if boundary=='periodic' else flux_jax(u[:1])[0]
    prev=jnp.roll(faces,1,axis=0) if boundary=='periodic' else jnp.concatenate((left[None,:],faces[:-1]))
    return faces-prev,(left,faces[-1])

def scalar_flux(u,method,equation):
    """Common historical scalar state-reconstruction/Rusanov path."""
    rec=reconstruction(method,'scalar')
    p=(-2,-1,0,1,2,3) if method=='neno6' else (-2,-1,0,1,2)
    m=(3,2,1,0,-1,-2) if method=='neno6' else (3,2,1,0,-1)
    ul=rec(gather(u,p,'periodic'));ur=rec(gather(u,m,'periodic'))
    if equation=='advection':
        fl,fr,alpha=ul,ur,1.
    else:
        fl,fr,alpha=.5*ul**2,.5*ur**2,1.
    face=.5*(fl+alpha*ul)+.5*(fr-alpha*ur)
    return face-jnp.roll(face,1),(face[-1],face[-1])

def make_flux(method,physics,boundary,equation):
    if physics=='scalar':
        reconstruction(method,'scalar')  # Construct checkpoint closures outside tracing.
        return lambda u:scalar_flux(u,method,equation)
    if method=='neno6':return lambda u:neno_euler_flux(u,boundary)
    return get_euler_flux_jax(reconstruction(method,physics),boundary,return_boundary_flux=True)

def pressure(u):return .4*(u[...,2]-.5*u[...,1]**2/u[...,0])

def admissible(u,physics):
    finite=jnp.all(jnp.isfinite(u))
    return finite if physics=='scalar' else finite&jnp.all(u[:,0]>0)&jnp.all(pressure(u)>0)

@lru_cache(None)
def checked_step(method,physics,boundary,equation):
    flux=make_flux(method,physics,boundary,equation)
    def step(u,dt,dx):
        shape=() if physics=='scalar' else (3,)
        def stage(v,base,weight):
            net,(left,right)=flux(v)
            trial=base+weight*(v-dt/dx*net)
            ok=admissible(trial,physics)
            lo=jnp.min(trial) if physics=='scalar' else jnp.min(trial[:,0])
            lp=jnp.array(0.) if physics=='scalar' else jnp.min(pressure(trial))
            return trial,ok,lo,lp,dt*(right-left)
        a,oa,ra,pa,ea=stage(u,jnp.zeros_like(u),1.)
        def second(_):return stage(a,.75*u,.25)
        b,ob,rb,pb,eb=jax.lax.cond(oa,second,lambda _:(a,jnp.array(False),ra,pa,jnp.zeros(shape)),None)
        c,oc,rc,pc,ec=jax.lax.cond(oa&ob,lambda _:stage(b,u/3,2/3),
                                  lambda _:(b,jnp.array(False),rb,pb,jnp.zeros(shape)),None)
        ok=oa&ob&oc
        bad=jnp.where(~oa,1,jnp.where(~ob,2,jnp.where(~oc,3,0)))
        candidate=jnp.where(~oa,a,jnp.where(~ob,b,c))
        last=jnp.where(oa,a,u);last=jnp.where(oa&ob,b,last)
        return jnp.where(ok,c,u),dict(success=ok,first_invalid_stage=bad,
            stage_min_density=jnp.stack((ra,rb,rc)),stage_min_pressure=jnp.stack((pa,pb,pc)),
            boundary_exchange=jnp.where(ok,(ea+eb)/6+2*ec/3,jnp.zeros(shape)),
            invalid_candidate=candidate,last_admissible_stage=last)
    return jax.jit(step)

@lru_cache(None)
def lambda_function(physics):
    c=load_checkpoint(HEADS[physics])
    backbone=tuple((jnp.asarray(w),jnp.asarray(b)) for w,b in c.backbone)
    if c.schema=='linear':head=tuple(jnp.asarray(v) for v in c.head)
    else:head=tuple(tuple(jnp.asarray(v) for v in layer) for layer in c.head)
    def evaluate(s):
        p=1-toy_keep_weight(backbone,s)
        return toy_lambda(head,p) if c.schema=='linear' else toy_lambda_v2(head,p,s)
    return jax.jit(evaluate)

def euler_lambda(u,boundary):
    p,m=euler_split_flux_stencils(u,boundary)
    return jnp.stack((lambda_function('euler')(p),lambda_function('euler')(m)),axis=-1)
