"""Train the original NN-TENO5 backbone through the full conservative FV solver.

New checkpoints only. Original data/test files and earlier experiments are read
only. A fixed-time training rollout never substitutes a repaired average or a
shorter physical horizon after a failed step. Inference uses bounded retries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import resource
import subprocess
import sys
import time
from functools import lru_cache

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import optax

from src.differentiable.accuracy_upgrade import Admissibility, safeguarded_step
from src.differentiable.gated_teno5_jax import gated_nn_teno5_reconstruct
from src.differentiable.teno5_jax import teno5_reconstruct_jax
from src.differentiable.teno5_jax_toy import init_toy_params, toy_keep_weight
from src.differentiable.model_io import load_checkpoint, sha256_file
from scripts.unrolled_physics_loss.Train_Gated_NNTENO5_Dataset_V2 import load_split, FAMILY_PROBS
from experiments.accuracy_upgrade.train import make_pool
from experiments.accuracy_upgrade.study import ROOT, OUT, write

RUN=OUT/'training_2026_11_12'
CONFIG=RUN/'configs/training.json'


def configure_run(path):
    """Keep repeat runs in new experimental subdirectories."""
    global RUN,CONFIG
    path=Path(path).resolve()
    if not path.is_relative_to(OUT) or path==OUT:
        raise ValueError('run-dir must be a subdirectory of experiments/accuracy_upgrade')
    RUN=path;CONFIG=RUN/'configs/training.json'
    return RUN


def read_config(path=None):
    path=CONFIG if path is None else path
    config=json.loads(Path(path).read_text())
    if config['architecture'] != [5,64,32,16,8,3] or not 0<=config['strength']<=1:
        raise ValueError('Keep the declared architecture and a bounded neural strength')
    if config['physics']!='euler' or config['rollout_steps']<1:
        raise ValueError('Invalid Euler rollout configuration')
    return config


def conservative_rollout(params,u,dt,steps=15,strength=.25,neural=True,config=Admissibility()):
    """Actual safeguarded SSPRK3 stages with fixed total time and diagnostics.

    lax.map (below), rather than vmap, preserves lazy admissibility branches
    across independent trajectories. Failed steps stop this trajectory and are
    returned explicitly; callers reject the entire optimizer update.
    """
    rec=(lambda s:gated_nn_teno5_reconstruct(params,s,'soft',strength)) if neural else teno5_reconstruct_jax
    def advance(carry,_):
        state,valid=carry
        candidate,diag=safeguarded_step(state,dt,1/len(u),rec,'transmissive',config)
        ok=valid & diag['success']
        next_state=jax.lax.cond(ok,lambda _:candidate,lambda _:state,None)
        obs=dict(valid=ok,flux_limited=jnp.sum(diag['stages']['theta']<1.),
            face_limited=jnp.sum(diag['stages']['face_theta']<1.),
            raw_invalid_rows=jnp.sum(diag['stages']['raw_invalid']),
            min_density=jnp.min(diag['stages']['min_density']),
            min_pressure=jnp.min(diag['stages']['min_pressure']),
            max_cfl=jnp.max(diag['stages']['cfl']),
            boundary_exchange=jnp.where(ok,diag['exchange'],jnp.zeros(3)))
        return (next_state,ok),obs
    (final,valid),history=jax.lax.scan(advance,(u,jnp.array(True)),None,length=steps)
    return final,dict(valid=valid,flux_limited=jnp.sum(history['flux_limited']),
        face_limited=jnp.sum(history['face_limited']),
        raw_invalid_rows=jnp.sum(history['raw_invalid_rows']),
        min_density=jnp.min(history['min_density']),min_pressure=jnp.min(history['min_pressure']),
        max_cfl=jnp.max(history['max_cfl']),boundary_exchange=jnp.sum(history['boundary_exchange'],axis=0))


def batched_rollout(params,states,dt,steps=15,strength=.25,neural=True,config=Admissibility()):
    return jax.lax.map(lambda pair:conservative_rollout(params,pair[0],pair[1],steps,strength,neural,config),
                       (states,dt))


def loss_function(params,states,target,dt,steps=15,strength=.25,config=Admissibility()):
    final,diag=batched_rollout(params,states,dt,steps,strength,True,config)
    # Every physical component is normalized within its own trajectory.
    scales=jnp.maximum(jnp.mean(target*target,axis=1),.05)
    losses=jnp.mean((final-target)**2/scales[:,None,:],axis=(1,2))
    return jnp.mean(losses),diag


def optimizer_for(config):
    return optax.chain(optax.clip_by_global_norm(config['gradient_clip_norm']),
                       optax.adam(config['learning_rate']))


def build_update(config):
    optimizer=optimizer_for(config);safety=Admissibility(**config['safeguard_parameters'])
    @jax.jit
    def update(params,opt_state,states,target,dt):
        def loss(p):return loss_function(p,states,target,dt,config['rollout_steps'],config['strength'],safety)
        (value,diag),grad=jax.value_and_grad(loss,has_aux=True)(params)
        norm=optax.global_norm(grad)
        finite=jnp.isfinite(value)&jnp.isfinite(norm)
        accepted=finite&jnp.all(diag['valid'])
        def apply(_):
            delta,new_state=optimizer.update(grad,opt_state,params)
            next_params=optax.apply_updates(params,delta)
            return next_params,new_state
        next_params,next_state=jax.lax.cond(accepted,apply,lambda _:(params,opt_state),None)
        return next_params,next_state,value,norm,accepted,diag
    return optimizer,update


def build_predict(config,neural=True):
    safety=Admissibility(**config['safeguard_parameters'])
    return jax.jit(lambda params,u,t:batched_rollout(params,u,t,config['rollout_steps'],config['strength'],neural,safety))


def initial_params(seed):
    return init_toy_params(jax.random.PRNGKey(seed),hidden_sizes=(64,32,16,8))


def prepare(config):
    if (RUN/'configs/fresh_holdout.json').exists():raise FileExistsError('Holdout already declared')
    from src.differentiable.v2_benchmark import exact_module
    ex=exact_module();rng=np.random.default_rng(config['holdout_seed']);cases=[]
    for i,family in enumerate(['random_riemann','sod_like','contact','two_rarefactions']*2):
        times,states,meta=ex.euler_case(rng,128,family)
        case=dict(id=f'fresh_{i}_{family}',family=family,kind='riemann',system='euler',
            left=meta['left_primitive'],right=meta['right_primitive'],jump=meta['jump'],
            time=min(.1,float(times[-1])),boundary='transmissive',grids=config['holdout_grids'])
        cases.append(case)
    write(RUN/'configs/fresh_holdout.json',dict(seed=config['holdout_seed'],cases=cases,
        purpose='confirmation after validation-only checkpoint selection; never use for tuning'))
    write(RUN/'configs/provenance.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        python=sys.version,platform=platform.platform(),devices=[str(x) for x in jax.devices()],
        jax=jax.__version__,numpy=np.__version__,optax=optax.__version__,
        config_sha256=sha256_file(CONFIG),holdout_sha256=sha256_file(RUN/'configs/fresh_holdout.json'),
        earlier_holdout='already examined; not used as new confirmation',
        originals='no old checkpoint, dataset or benchmark output is an output target'))


def predict_cases(params,data,config,predict):
    rows=[]
    for n in config['validation_grids']:
        group=data[n];batch=config['validation_batch_size']
        for lo in range(0,len(group['ids']),batch):
            hi=min(lo+batch,len(group['ids']));u=group['initial'][lo:hi];target=group['target'][lo:hi]
            final,diag=predict(params,jnp.asarray(u),jnp.asarray(group['dt'][lo:hi]));final=np.asarray(final)
            diag=jax.tree.map(np.asarray,diag)
            for k in range(hi-lo):
                diff=final[k]-target[k];scale=np.maximum(np.mean(target[k]**2,axis=0),.05)
                rows.append(dict(case_id=str(group['ids'][lo+k]),family=str(group['family'][lo+k]),cells=n,
                    valid=bool(diag['valid'][k]),l1=float(np.mean(abs(diff))),l2=float(np.sqrt(np.mean(diff*diff))),
                    linf=float(np.max(abs(diff))),component_l1=np.mean(abs(diff),axis=0).tolist(),
                    normalized_mse=float(np.mean(diff*diff/scale)),
                    flux_limited=int(diag['flux_limited'][k]),face_limited=int(diag['face_limited'][k]),
                    raw_invalid_rows=int(diag['raw_invalid_rows'][k]),
                    minimum_density=float(diag['min_density'][k]),minimum_pressure=float(diag['min_pressure'][k]),
                    conservation_residual=(np.sum(final[k]-u[k],axis=0)/n+diag['boundary_exchange'][k]).tolist()))
    return rows


def validation_summary(rows,baseline):
    lookup={r['case_id']:r for r in baseline};groups={}
    for r in rows:
        b=lookup[r['case_id']]
        if r['family']=='constant' or b['l1']<=1e-10:continue
        key=f'{r["family"]}_N{r["cells"]}'
        groups.setdefault(key,[]).append(r['l1']/b['l1'])
    good=[r for r in rows if r['valid'] and r['family']!='constant']
    return dict(failures=sum(not r['valid'] for r in rows),
        macro_l1_ratio=float(np.mean([np.mean(v) for v in groups.values()])),
        group_l1_ratios={k:float(np.mean(v)) for k,v in groups.items()},
        mean_l1=float(np.mean([r['l1'] for r in good])),
        normalized_mse=float(np.mean([r['normalized_mse'] for r in good])),
        flux_limited=int(sum(r['flux_limited'] for r in rows)),face_limited=int(sum(r['face_limited'] for r in rows)),
        max_conservation_residual=np.max(np.abs([r['conservation_residual'] for r in rows]),axis=0).tolist())


def save_model(path,params,metadata):
    with path.open('xb') as f:pickle.dump(jax.tree.map(np.asarray,params),f)
    write(path.with_suffix('.json'),dict(**metadata,sha256=sha256_file(path)))


def smoke(config):
    data,ids=load_split(ROOT/'data/nnteno_v2/data','euler','train')
    group=data[64];ix=group['indices']['sod_like'][:config['physics_batch_size']]
    params=initial_params(config['seeds'][0]);optimizer,update=build_update(config);state=optimizer.init(params)
    started=time.perf_counter()
    params,state,value,norm,accepted,diag=update(params,state,jnp.asarray(group['initial'][ix]),
                 jnp.asarray(group['target'][ix]),jnp.asarray(group['dt'][ix]))
    jax.block_until_ready(params)
    result=dict(loss=float(value),gradient_norm=float(norm),accepted=bool(accepted),
        compile_and_first_update_seconds=time.perf_counter()-started,
        diagnostics=jax.tree.map(np.asarray,diag),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(RUN/'results/gradient_smoke.json',result);print(json.dumps({k:v for k,v in result.items() if k!='diagnostics'}),flush=True)
    if not bool(accepted):raise FloatingPointError('Do not launch training after a failed gradient smoke')


def train(config):
    if not (RUN/'results/gradient_smoke.json').exists():raise RuntimeError('Run the gradient smoke first')
    data,trainids=load_split(ROOT/'data/nnteno_v2/data','euler','train')
    val,validids=load_split(ROOT/'data/nnteno_v2/data','euler','validation')
    if trainids.intersection(validids):raise RuntimeError('Trajectory contamination')
    neural_predict=build_predict(config);classical_predict=build_predict(config,False)
    saved=load_checkpoint('models/v2/euler_gated_dataset_v2_seed2026.pkl')
    started=time.perf_counter()
    baseline=predict_cases(initial_params(config['seeds'][0]),val,config,classical_predict)
    if any(not r['valid'] for r in baseline):raise FloatingPointError('Classical validation baseline failed')
    write(RUN/'results/validation_classical_safe_fv.json',baseline)
    old_rows=predict_cases(saved.backbone,val,config,neural_predict)
    write(RUN/'results/validation_saved_v2_safe_fv.json',dict(summary=validation_summary(old_rows,baseline),cases=old_rows))
    print('Validation baselines complete',time.perf_counter()-started,'seconds',flush=True)
    optimizer,update=build_update(config)
    warm_optimizer=optimizer_for(config)
    @jax.jit
    def warm_step(p,s,x,y):
        def loss(q):
            keep=jnp.clip(toy_keep_weight(q,x),1e-7,1-1e-7)
            return jnp.mean((-y*jnp.log(keep)-(1-y)*jnp.log1p(-keep))*jnp.where(y<.5,2.,1.))
        value,grad=jax.value_and_grad(loss)(p);delta,s=warm_optimizer.update(grad,s,p)
        return optax.apply_updates(p,delta),s,value,optax.global_norm(grad)
    all_runs=[]
    for seed in config['seeds']:
        run_start=time.perf_counter();params=initial_params(seed);warm_state=warm_optimizer.init(params)
        pools=[];labels=[];warm_ids=[]
        for n in config['train_grids']:
            x,y,ids=make_pool(data[n],'C',seed+n,config['warmup_cases_per_family_per_grid'],config['warmup_max_stencils'])
            pools.append(np.asarray(x));labels.append(np.asarray(y));warm_ids.extend(ids)
        x=np.concatenate(pools);y=np.concatenate(labels);rng=np.random.default_rng(seed)
        if len(x)>config['warmup_max_stencils']:
            ix=rng.choice(len(x),config['warmup_max_stencils'],replace=False);x=x[ix];y=y[ix]
        x=jnp.asarray(x);y=jnp.asarray(y)
        log=RUN/f'results/seed{seed}_history.jsonl'
        def append(row):
            with log.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        for it in range(config['warmup_steps']):
            ix=rng.integers(0,len(x),size=config['warmup_batch_size']);params,warm_state,value,norm=warm_step(params,warm_state,x[ix],y[ix])
            if not np.isfinite(float(value)) or not np.isfinite(float(norm)):raise FloatingPointError('Nonfinite warm-up')
            append(dict(phase='warmup',step=it+1,loss=float(value),gradient_norm=float(norm)))
            if (it+1)%50==0:print('seed',seed,'warmup',it+1,'BCE',float(value),flush=True)
        metadata=dict(architecture=config['architecture'],parameter_count=3155,seed=seed,config=config,
            training_inputs='actual bidirectional characteristic state stencils from state_stencils',
            formulation='fv_state_flux',stage_integrator='safeguarded_step at every SSPRK3 stage; no average repairs',
            optimizer='Adam with global-norm clipping',eligible_training_ids=sorted(trainids),warmup_ids=warm_ids,
            validation_ids=sorted(validids),dataset_sha256={name:sha256_file(ROOT/'data/nnteno_v2/data'/name)
                for name in ['euler_train.h5','euler_validation.h5']},
            holdout_sha256=sha256_file(RUN/'configs/fresh_holdout.json'))
        warm_path=RUN/f'checkpoints/fv_conservative_seed{seed}_step0000.pkl'
        save_model(warm_path,params,dict(**metadata,physics_step=0))
        before=predict_cases(params,val,config,neural_predict);score=validation_summary(before,baseline)
        write(RUN/f'results/seed{seed}_validation_step0000.json',dict(summary=score,cases=before))
        best_score=score['macro_l1_ratio'] if score['failures']==0 else float('inf');best_path=warm_path;best_step=0
        print('seed',seed,'warmup validation ratio',best_score,flush=True)
        state=optimizer.init(params);prng=np.random.default_rng(seed+50000);families=list(FAMILY_PROBS['euler'])
        weights=np.asarray(list(FAMILY_PROBS['euler'].values()));weights/=weights.sum()
        nonfinite=0;invalid_updates=0;consecutive_rejected=0;compile_times={};update_times=[];validation_records=[]
        for it in range(1,config['physics_updates']+1):
            n=int(prng.choice(config['train_grids']));family=str(prng.choice(families,p=weights));group=data[n]
            ix=prng.choice(group['indices'][family],config['physics_batch_size'],replace=len(group['indices'][family])<config['physics_batch_size'])
            inputs=(jnp.asarray(group['initial'][ix]),jnp.asarray(group['target'][ix]),jnp.asarray(group['dt'][ix]))
            tick=time.perf_counter()
            params,state,value,norm,accepted,diag=update(params,state,*inputs);jax.block_until_ready(params)
            elapsed=time.perf_counter()-tick
            if n not in compile_times:compile_times[n]=elapsed
            else:update_times.append(elapsed)
            finite=np.isfinite(float(value)) and np.isfinite(float(norm));nonfinite+=int(not finite)
            valid=bool(jnp.all(diag['valid']));invalid_updates+=int(not valid)
            consecutive_rejected=0 if bool(accepted) else consecutive_rejected+1
            append(dict(phase='physics',step=it,cells=n,family=family,
                case_ids=[str(group['ids'][i]) for i in ix],loss=float(value) if finite else None,
                gradient_norm=float(norm) if finite else None,accepted=bool(accepted),
                flux_limited=int(jnp.sum(diag['flux_limited'])),face_limited=int(jnp.sum(diag['face_limited'])),
                raw_invalid_rows=int(jnp.sum(diag['raw_invalid_rows'])),
                minimum_density=float(jnp.min(diag['min_density'])),minimum_pressure=float(jnp.min(diag['min_pressure'])),
                max_cfl=float(jnp.max(diag['max_cfl'])),seconds=elapsed))
            if it%10==0:print('seed',seed,'update',it,'N',n,'loss',float(value),'norm',float(norm),'accepted',bool(accepted),flush=True)
            if consecutive_rejected>=3:raise FloatingPointError('Three consecutive rejected updates; not resampling away failures')
            if it%config['validation_every']==0 or it==config['physics_updates']:
                rows=predict_cases(params,val,config,neural_predict);score=validation_summary(rows,baseline)
                write(RUN/f'results/seed{seed}_validation_step{it:04d}.json',dict(summary=score,cases=rows))
                validation_records.append(dict(step=it,**score))
                path=RUN/f'checkpoints/fv_conservative_seed{seed}_step{it:04d}.pkl'
                save_model(path,params,dict(**metadata,physics_step=it,validation=score))
                if score['failures']==0 and score['macro_l1_ratio']<best_score:
                    best_score=score['macro_l1_ratio'];best_path=path;best_step=it
                print('seed',seed,'validation',it,'ratio',score['macro_l1_ratio'],'best',best_score,'failures',score['failures'],flush=True)
        with (RUN/f'checkpoints/seed{seed}_resume_final.pkl').open('xb') as f:
            pickle.dump(dict(params=jax.tree.map(np.asarray,params),optimizer_state=jax.tree.map(np.asarray,state),
                numpy_rng_state=prng.bit_generator.state,step=config['physics_updates'],config=config),f)
        result=dict(seed=seed,best_checkpoint=str(best_path.relative_to(ROOT)),best_sha256=sha256_file(best_path),
            best_step=best_step,best_validation_ratio=best_score,warmup_validation=validation_summary(before,baseline),
            validation_records=validation_records,nonfinite_gradient_updates=nonfinite,
            invalid_rollout_updates=invalid_updates,compile_and_first_update_s=compile_times,
            warmed_update_median_s=float(np.median(update_times)),wall_seconds=time.perf_counter()-run_start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            architecture=config['architecture'],config=config,holdout_sha256=metadata['holdout_sha256'])
        write(RUN/f'results/seed{seed}_training_summary.json',result);all_runs.append(result)
        print('seed',seed,'TRAINING COMPLETE best step',best_step,'ratio',best_score,flush=True)
    write(RUN/'results/training_summary.json',all_runs)
    write(RUN/'configs/frozen_candidates.json',dict(selection='validation only; no benchmark/holdout tuning',
        candidates=[dict(seed=r['seed'],checkpoint=r['best_checkpoint'],sha256=r['best_sha256'],step=r['best_step']) for r in all_runs],
        holdout_sha256=sha256_file(RUN/'configs/fresh_holdout.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','smoke','train'])
    parser.add_argument('--run-dir',type=Path,default=RUN)
    args=parser.parse_args();configure_run(args.run_dir);config=read_config()
    {'prepare':prepare,'smoke':smoke,'train':train}[args.action](config)
