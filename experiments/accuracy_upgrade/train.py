"""Paired, fixed-budget input ablation. No historical checkpoint is overwritten.

CPU pilot, not a replication of the saved three-grid 300-update training.
Rejected/nonfinite updates are logged and never applied. No state clipping.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import optax
from scripts.unrolled_physics_loss.Train_Gated_NNTENO5_Dataset_V2 import (
    load_split, training_pool, FAMILY_PROBS)
from src.differentiable.accuracy_upgrade import (
    state_stencils, flux_variant, admissible, pressure)
from src.differentiable.euler_jax import euler_characteristic_state_stencils, euler_split_flux_stencils
from src.differentiable.teno5_jax import teno5_selection_values,DEFAULT_CUTOFF,teno5_reconstruct_jax
from src.differentiable.teno5_jax_toy import init_toy_params,toy_keep_weight
from src.differentiable.gated_teno5_jax import gated_nn_teno5_reconstruct
from experiments.accuracy_upgrade.study import OUT,ROOT,write


def canonical_inputs(u,variant):
    if variant=='A':return (euler_characteristic_state_stencils(u,'transmissive'),)
    if variant=='B':return euler_split_flux_stencils(u,'transmissive')
    if variant=='C':return state_stencils(u,'transmissive')
    raise ValueError(variant)


def make_pool(group,variant,seed,per_family=12,max_pool=15000):
    rng=np.random.default_rng(seed)
    picked=[]
    for f in sorted(group['indices']):
        ids=group['indices'][f];picked.extend(rng.choice(ids,min(per_family,len(ids)),replace=False))
    # Same trajectory sample for all representations; each supplies its actual
    # inference stencils. Legacy A deliberately retains forward-only warm-up.
    values=np.concatenate([np.asarray(jnp.stack(canonical_inputs(jnp.asarray(group['initial'][i]),variant))).reshape(-1,5) for i in picked])
    if len(values)>max_pool:values=values[rng.choice(len(values),max_pool,replace=False)]
    labels=(np.asarray(teno5_selection_values(jnp.asarray(values)))>=DEFAULT_CUTOFF).astype(float)
    troubled=np.flatnonzero(~np.all(labels==1,axis=1));smooth=np.flatnonzero(np.all(labels==1,axis=1))
    k=min(len(troubled),len(smooth));ix=np.r_[rng.choice(troubled,k,replace=False),rng.choice(smooth,k,replace=False)]
    return jnp.asarray(values[ix]),jnp.asarray(labels[ix]),[str(group['ids'][i]) for i in picked]


def predict_one(params,u,dt,variant,steps=15,neural=True):
    rec=(lambda s:gated_nn_teno5_reconstruct(params,s,'soft',.25)) if neural else teno5_reconstruct_jax
    # Prediction has explicit rejection. No eigenbasis is evaluated on an
    # invalid stage, and no nonconservative repair is applied.
    flux=flux_variant(rec,'transmissive','fv_state_flux' if variant=='C' else 'historical',
                      face_guard=variant=='C')
    def fe(v):
        result=v-dt*len(u)*flux(v)
        ok=jnp.all(admissible(result))
        # Invalid candidate contributes a finite pre-correction penalty.
        safe_p=pressure(result)
        penalty=jnp.mean(jnp.maximum(1e-12-result[:,0],0.)**2+jnp.maximum(1e-12-safe_p,0.)**2)
        result=jax.lax.cond(ok,lambda _:result,lambda _:v,None)
        return result,ok,penalty
    def step(carry,_):
        v,valid=carry
        def advance(_):
            a,oka,pa=fe(v)
            bfe,okb,pb=fe(a);b=.75*v+.25*bfe
            cfe,okc,pc=fe(b);result=v/3.+2*cfe/3.
            ok=oka&okb&okc&jnp.all(admissible(result))
            return (jax.lax.cond(ok,lambda _:result,lambda _:v,None),ok),pa+pb+pc
        return jax.lax.cond(valid,advance,lambda _:((v,valid),jnp.array(0.)),None)
    (final,valid),penalties=jax.lax.scan(step,(u,jnp.array(True)),None,length=steps)
    return final,valid,jnp.mean(penalties)


def train_all(output,seeds):
    output.mkdir(exist_ok=False)
    cfg=json.loads((OUT/'configs/protocol.json').read_text())
    data,trainids=load_split(ROOT/'data/nnteno_v2/data','euler','train')
    val,validids=load_split(ROOT/'data/nnteno_v2/data','euler','validation')
    assert not trainids.intersection(validids)
    group=data[64]
    results=[]
    # JIT executables reused across seeds and A/B; C is a different numerical path.
    predictor={}
    def build_predictor(v):
        return jax.jit(jax.vmap(lambda p,u,t:predict_one(p,u,t,v),in_axes=(None,0,0)))
    for v in ['A','B','C']:
        key='C' if v=='C' else 'H'
        if key not in predictor:
            predictor[key]=build_predictor('C' if key=='C' else 'A')
    def evaluate(params,variant):
        rows=[];key='C' if variant=='C' else 'H'
        for n in cfg['validation_grids']:
            g=val[n]
            for lo in range(0,len(g['ids']),8):
                final,ok,pen=predictor[key](params,jnp.asarray(g['initial'][lo:lo+8]),jnp.asarray(g['dt'][lo:lo+8]))
                final=np.asarray(final);ok=np.asarray(ok)
                target=g['target'][lo:lo+8]
                for k in range(len(final)):
                    scale=np.maximum(np.mean(target[k]**2,axis=0),.05)
                    rows.append(dict(id=str(g['ids'][lo+k]),family=str(g['family'][lo+k]),cells=n,valid=bool(ok[k]),
                        l1=float(np.mean(abs(final[k]-target[k]))),component_l1=np.mean(abs(final[k]-target[k]),axis=0).tolist(),
                        normalized_mse=float(np.mean((final[k]-target[k])**2/scale))))
        eligible=[r for r in rows if r['family']!='constant' and r['valid']]
        return dict(failures=sum(not r['valid'] for r in rows),
            mean_l1=float(np.mean([r['l1'] for r in eligible])),
            mean_normalized_mse=float(np.mean([r['normalized_mse'] for r in eligible])),cases=rows)
    # Warm-up classifier objective matches the original weighted teacher BCE.
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(cfg['learning_rate']))
    @jax.jit
    def warm_step(p,s,x,y):
        def loss(p):
            keep=jnp.clip(toy_keep_weight(p,x),1e-7,1-1e-7)
            return jnp.mean((-y*jnp.log(keep)-(1-y)*jnp.log1p(-keep))*jnp.where(y<.5,2.,1.))
        value,grad=jax.value_and_grad(loss)(p);updates,s=opt.update(grad,s,p)
        return optax.apply_updates(p,updates),s,value,optax.global_norm(grad)
    @jax.jit
    def classification(p,x,y):
        keep=jnp.clip(toy_keep_weight(p,x),1e-7,1-1e-7)
        return jnp.mean((-y*jnp.log(keep)-(1-y)*jnp.log1p(-keep))*jnp.where(y<.5,2.,1.)),jnp.mean((keep>=.5)==(y>=.5))
    physics_steps={}
    for variant in ['A','B','C']:
        key='C' if variant=='C' else 'H'
        if key in physics_steps:continue
        def build(v):
            def loss(p,u,target,dt):
                final,ok,pen=jax.vmap(lambda x,t:predict_one(p,x,t,v))(u,dt)
                scales=jnp.maximum(jnp.mean(target*target,axis=(0,1)),.05)
                mse=jnp.mean((final-target)**2/scales)
                return mse+.1*jnp.mean(pen),(mse,jnp.mean(pen),jnp.all(ok))
            @jax.jit
            def update(p,s,u,target,dt):
                (value,aux),grad=jax.value_and_grad(loss,has_aux=True)(p,u,target,dt)
                norm=optax.global_norm(grad);finite=jnp.isfinite(value)&jnp.isfinite(norm)&aux[2]
                delta,nexts=opt.update(grad,s,p)
                nextp=optax.apply_updates(p,delta)
                # Reject rather than apply an invalid/nonfinite optimizer step.
                nextp,nexts=jax.lax.cond(finite,lambda _:(nextp,nexts),lambda _:(p,s),None)
                return nextp,nexts,value,aux,norm,finite
            return update
        physics_steps[key]=build('C' if key=='C' else 'A')
    for seed in seeds:
        for variant in ['A','B','C']:
            started=time.perf_counter();rng=np.random.default_rng(seed)
            params=init_toy_params(jax.random.PRNGKey(seed),hidden_sizes=(64,32,16,8))
            pool,labels,ids=make_pool(group,variant,seed)
            # Always evaluate classification on canonical selected solver inputs.
            actual='C' if variant=='C' else 'B'
            probe,probe_y,_=make_pool(group,actual,seed+10000)
            initial_cls=tuple(float(x) for x in classification(params,probe,probe_y))
            state=opt.init(params);history=[]
            for it in range(cfg['warmup_steps']):
                ix=rng.integers(0,len(pool),size=cfg['warmup_batch'])
                params,state,value,norm=warm_step(params,state,pool[ix],labels[ix])
                history.append(dict(phase='warmup',step=it,loss=float(value),gradient_norm=float(norm)))
            cls=tuple(float(x) for x in classification(params,probe,probe_y))
            before=evaluate(params,variant)
            run=output/f'{variant}_seed{seed}'
            def save(name,p):
                with (output/f'{run.name}_{name}.pkl').open('xb') as f:pickle.dump(jax.tree.map(np.asarray,p),f)
            save('warmup',params)
            state=opt.init(params);families=list(FAMILY_PROBS['euler']);weights=np.array(list(FAMILY_PROBS['euler'].values()));weights/=weights.sum()
            # Independent RNG makes physics batches exactly paired across variants.
            prng=np.random.default_rng(seed+50000);nonfinite=0;violations=0
            for it in range(cfg['physics_updates']):
                family=prng.choice(families,p=weights);ix=prng.choice(group['indices'][family],cfg['physics_batch'],replace=True)
                params,state,value,aux,norm,ok=physics_steps['C' if variant=='C' else 'H'](params,state,
                    jnp.asarray(group['initial'][ix]),jnp.asarray(group['target'][ix]),jnp.asarray(group['dt'][ix]))
                nonfinite+=int(not np.isfinite(float(norm)) or not np.isfinite(float(value)));violations+=int(not aux[2])
                history.append(dict(phase='physics',step=it,family=str(family),loss=float(value),state_loss=float(aux[0]),
                    positivity_penalty=float(aux[1]),gradient_norm=float(norm),accepted=bool(ok)))
            after=evaluate(params,variant);save('final',params)
            row=dict(variant=variant,seed=seed,architecture=[5,64,32,16,8,3],parameter_count=3155,
                formulation='fv_state_flux' if variant=='C' else 'historical',
                inputs='forward characteristic states (legacy mismatch)' if variant=='A' else
                       'both characteristic split fluxes' if variant=='B' else 'both characteristic states',
                config=cfg,train_case_ids=sorted(str(x) for x in group['ids']),warmup_case_ids=ids,
                validation_case_ids=sorted(r['id'] for r in after['cases']),
                case_id_definition='eligible N=64 training pool and actually evaluated N=64/128 validation IDs; split-wide disjointness checked separately',
                initial_classifier_bce=initial_cls[0],initial_classifier_accuracy=initial_cls[1],
                warm_classifier_bce=cls[0],warm_classifier_accuracy=cls[1],
                nonfinite_gradient_count=nonfinite,physical_violation_updates=violations,
                warmup_validation=before,final_validation=after,history=history,
                wall_seconds=time.perf_counter()-started,selection='fixed final update; no checkpoint cherry-picking',
                training_guard='reject invalid stages/updates, no state repair; not exact original guarded procedure',
                face_guard=variant=='C',
                dataset_sha256=json.loads((OUT/'baseline_manifest.json').read_text())['dataset_identities']['files']['euler_train.h5']['sha256'],
                holdout_sha256=hashlib.sha256((OUT/'configs/final_holdout.json').read_bytes()).hexdigest())
            write(run.with_suffix('.json'),row);results.append({k:v for k,v in row.items() if k not in ('history','train_case_ids','validation_case_ids')})
            print('trained',variant,seed,'warm',before['mean_l1'],'final',after['mean_l1'],'failures',after['failures'],'seconds',row['wall_seconds'],flush=True)
    write(output/'summary.json',results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[3101,3102]);a=p.parse_args()
    train_all(a.output,a.seeds)
