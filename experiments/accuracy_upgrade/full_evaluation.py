"""Evaluate frozen conservative-training checkpoints on matched numerical problems."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from experiments.accuracy_upgrade import study
from experiments.accuracy_upgrade.full_training import RUN,read_config,configure_run
from experiments.accuracy_upgrade.study import ROOT,OUT,write,evolve
from src.differentiable.model_io import load_checkpoint,sha256_file
from src.differentiable.reconstruction import make_reconstruction
from src.differentiable.v2_benchmark import state as case_state,exact_module
from src.differentiable.euler_jax import spds_jax
from src.differentiable.accuracy_upgrade import pressure,flux_variant
from src.differentiable.evaluation import errors


def register_models():
    config=read_config();frozen=json.loads((RUN/'configs/frozen_candidates.json').read_text())
    if sha256_file(RUN/'configs/fresh_holdout.json')!=frozen['holdout_sha256']:raise ValueError('Holdout identity changed')
    names=[]
    for candidate in frozen['candidates']:
        model=load_checkpoint(candidate['checkpoint'],candidate['sha256'])
        name=f'safe_fv_new_seed{candidate["seed"]}'
        study.EXTRA_RECONSTRUCTIONS[name]=make_reconstruction(model,'gated-soft',config['strength']);names.append(name)
    return names,frozen


def declare():
    config=read_config();cases=study.cases()
    cases.append(dict(id='shu_osher_full',kind='shu-osher',system='euler',time=1.8,length=10.,boundary='transmissive'))
    cases.append(dict(**{k:v for k,v in cases[0].items() if k not in ['id','time']},id='sod_long',time=.2))
    write(RUN/'configs/benchmark_spec.json',dict(cases=cases,grids=config['benchmark_grids'],
        generalization=dict(cases=['sod','lax'],cells=config['generalization_grid']),
        methods=list(study.VARIANTS)+[f'safe_fv_new_seed{s}' for s in config['seeds']],
        inference='matched proposed dt from initial speed, .25 CFL with 1.5 margin; bounded retries advance accepted dt only',
        reference_note='Full-time Shu reuses existing cell-average reference ladder with its uncertainty; no unsupported ranking'))


def reference(case,n):
    if case['kind']=='riemann':return case_state(case,n,case['time']),dict(kind='analytic Riemann cell averages',uncertainty_proxy=0.)
    if case['id']=='shu_osher':
        root=OUT/'results/shu_reference'
        files=[root/'N512_dt1.npz',root/'N1024_dt1.npz',root/'N1024_dt2.npz']
        arrays=[np.load(p)['final'].reshape(n,-1,3).mean(axis=1) for p in files]
        proxy=2*(np.mean(abs(arrays[1]-arrays[0]))+np.mean(abs(arrays[2]-arrays[1])))
        return arrays[2],dict(kind='earlier FV state TENO5 256/512/1024 plus finest dt/2',
            uncertainty_proxy=float(proxy),source_hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in files},
            ranking_certified=False)
    if case['id']=='shu_osher_full':
        root=ROOT/'docs/research_release/results/independent/independent-shu-osher-256'
        metadata=json.loads((root/'reference.json').read_text());data=np.load(root/'reference.npz')
        # The archived state is the dt/2 finest reference restricted to 256.
        # The full fine temporal array was not exported; it is not fabricated.
        if n>256:raise ValueError('Archived full-time Shu reference supports N<=256')
        target=data['state'].reshape(n,-1,3).mean(axis=1)
        a=data['reference_1024'].reshape(n,-1,3).mean(axis=1)
        b=data['reference_2048'].reshape(n,-1,3).mean(axis=1)
        proxy=2*(np.mean(abs(a-b))+np.mean(abs(target-b)))
        return target,dict(kind='archived historical TENO5 cell-average ladder 512/1024/2048 plus finest dt/2',
            time=1.8,length=10.,uncertainty_proxy=float(proxy),source_metadata=metadata,
            source_hashes={str((root/name).relative_to(ROOT)):sha256_file(root/name) for name in ['reference.json','reference.npz']},
            ranking_certified=False,limitation='empirical Cauchy proxy, not a rigorous reference error bound')
    return None,dict(kind='stability-only; no trustworthy blast accuracy reference')


def execute(case,n,name,output,reference_state=None,reference_meta=None):
    initial=case_state(case,n);dx=case.get('length',1.)/n
    speed=float(jnp.max(jnp.abs(spds_jax(jnp.asarray(initial)))))
    count=max(1,int(np.ceil(case['time']*speed*1.5/(.25*dx))));dt=case['time']/count
    final,diag,history,times,states=evolve(initial,name,case['boundary'],case['time'],dx,dt,20)
    row=dict(case=case['id'],definition=case,cells=n,variant=name,proposed_dt=dt,dx=dx,
        reference=reference_meta,raw_invalid_count_scope='accepted attempts; retries reported separately',**diag)
    if diag['success'] and reference_state is not None:
        row.update(errors(final,reference_state))
        row['density_tv']=float(np.sum(abs(np.diff(final[:,0]))))
        pressures=np.asarray(pressure(jnp.asarray(final)))
        row['pressure_tv']=float(np.sum(abs(np.diff(pressures))))
        reference_pressure=np.asarray(pressure(jnp.asarray(reference_state)))
        row['pressure_l1']=float(np.mean(abs(pressures-reference_pressure)))
        row['density_overshoot']=max(0.,float(final[:,0].max()-reference_state[:,0].max()))
        row['density_undershoot']=max(0.,float(reference_state[:,0].min()-final[:,0].min()))
        if case['kind']=='riemann':
            row['error_over_time']=[dict(time=float(t),**errors(u,case_state(case,n,float(t)))) for t,u in zip(times,states)]
            exact=exact_module().ExactEuler(case['left'],case['right'],case['jump'])
            centers=(np.arange(n)+.5)/n
            contact=case['jump']+exact.vel*case['time']
            row['exact_contact_position']=float(contact)
            if abs(exact.rsl-exact.rsr)>1e-10:
                fraction=(final[:,0]-exact.rsr)/(exact.rsl-exact.rsr)
                row['contact_10_90_width_cells']=int(np.sum((abs(centers-contact)<.06)&(fraction>.1)&(fraction<.9)))
            shock_metrics=[]
            for side,exists,speed in [('left',exact.left_shock,exact.L[0]),('right',exact.right_shock,exact.R[1])]:
                if not exists:continue
                position=case['jump']+speed*case['time']
                faces=np.arange(1,n)/n;window=np.flatnonzero(abs(faces-position)<.08)
                if not len(window):continue
                jump=abs(np.diff(final[:,0]));j=window[np.argmax(jump[window])]
                measured=float(faces[j])
                shock_metrics.append(dict(side=side,exact_position=float(position),measured_position=measured,
                    position_error=abs(measured-position),exact_speed=float(speed),
                    measured_average_speed=(measured-case['jump'])/case['time'],
                    definition='maximum density jump in a fixed +/- .08 window around the exact shock'))
            row['shock_metrics']=shock_metrics
        if case['kind']=='shu-osher':
            band=np.arange(n)>=int(.4*n)
            row['density_oscillation_range']=float(np.ptp(final[band,0]))
            row['reference_density_oscillation_range']=float(np.ptp(reference_state[band,0]))
    if not diag['success']:
        last=history[-1]
        if 'diagnostics' in last:
            failed=np.flatnonzero(~last['diagnostics']['stages']['ok'])
            row['first_invalid_stage']=None if not len(failed) else int(failed[0]+1)
    filename=f'{case["id"]}_N{n}_{name}'
    write(output/f'{filename}.json',dict(**row,history=history))
    np.savez_compressed(output/f'{filename}.npz',initial=initial,final=final,
        reference=np.array([]) if reference_state is None else reference_state,times=times,states=states)
    print('benchmark',case['id'],n,name,'valid',diag['success'],'L1',row.get('l1'),flush=True)
    return row


def cached_or_execute(case,n,name,output,ref,meta):
    path=output/f'{case["id"]}_N{n}_{name}.json'
    if path.exists():
        row=json.loads(path.read_text())
        if row['definition']!=case or row['variant']!=name or row['cells']!=n:
            raise ValueError('Cached case identity mismatch')
        arrays=np.load(path.with_suffix('.npz'))
        if arrays['initial'].shape!=(n,3):raise ValueError('Cached state shape mismatch')
        row.pop('history',None)
        return row
    return execute(case,n,name,output,ref,meta)


def baseline_problems():
    """Known canonical controls may run while fixed-policy training continues.

    Fresh holdout and new-model evaluation still require frozen candidates.
    Cached controls retain failures as well as successful trajectories.
    """
    spec=json.loads((RUN/'configs/benchmark_spec.json').read_text())
    output=RUN/'results/problems';output.mkdir(exist_ok=False)
    for n in spec['grids']:
        for case in spec['cases']:
            ref,meta=reference(case,n)
            for name in study.VARIANTS:cached_or_execute(case,n,name,output,ref,meta)
    n=spec['generalization']['cells']
    for case in spec['cases']:
        if case['id'] not in spec['generalization']['cases']:continue
        ref,meta=reference(case,n)
        for name in ['historical_teno','historical_v2_soft25','safe_fv_teno','safe_fv_v2_soft25']:
            cached_or_execute(case,n,name,output,ref,meta)
    write(output/'baseline_complete.json',dict(complete=True,selection_use=False,
          scope='previously examined canonical controls; fresh holdout not accessed'))


def benchmarks():
    names,frozen=register_models();spec=json.loads((RUN/'configs/benchmark_spec.json').read_text())
    output=RUN/'results/problems';output.mkdir(exist_ok=True);rows=[]
    for n in spec['grids']:
        for case in spec['cases']:
            ref,meta=reference(case,n)
            for name in spec['methods']:rows.append(cached_or_execute(case,n,name,output,ref,meta))
    # Independent grid beyond the entire training set; canonical conditions were
    # already examined previously, so this is resolution generalization only.
    n=spec['generalization']['cells']
    for case in spec['cases']:
        if case['id'] not in spec['generalization']['cases']:continue
        ref,meta=reference(case,n)
        for name in ['historical_teno','historical_v2_soft25','safe_fv_teno','safe_fv_v2_soft25']+names:
            rows.append(cached_or_execute(case,n,name,output,ref,meta))
    for row in rows:
        for base in ['historical_teno','safe_fv_teno','safe_fv_v2_soft25']:
            b=next((r for r in rows if r['case']==row['case'] and r['cells']==row['cells'] and r['variant']==base),None)
            if b and b.get('l1',0)>1e-12 and row.get('l1') is not None:
                row['ratio_to_'+base]=row['l1']/b['l1']
                if row['definition']['kind']=='shu-osher':
                    row['ranking_resolved_to_'+base]=row['reference']['uncertainty_proxy']<.25*abs(row['l1']-b['l1'])
    write(output/'summary.json',rows)


def holdout():
    names,frozen=register_models();config=read_config();cases=json.loads((RUN/'configs/fresh_holdout.json').read_text())['cases']
    output=RUN/'results/fresh_holdout';output.mkdir(exist_ok=False)
    methods=['historical_teno','historical_v2_soft25','safe_fv_teno','safe_fv_v2_soft25']+names;rows=[]
    write(output/'frozen_candidates.json',frozen)
    for case in cases:
        for n in case['grids']:
            ref,meta=reference(case,n)
            for name in methods:rows.append(execute(case,n,name,output,ref,meta))
    write(output/'per_case.json',rows);stats=[]
    for n in config['holdout_grids']:
        for name in methods:
            for baseline in ['historical_teno','safe_fv_teno','safe_fv_v2_soft25']:
                group=[r for r in rows if r['variant']==name and r['cells']==n]
                base=[next(r for r in rows if r['variant']==baseline and r['case']==a['case'] and r['cells']==n) for a in group]
                failures=sum(not a['success'] for a in group)
                record=dict(cells=n,variant=name,baseline=baseline,failures=failures)
                if failures==0 and all(a['success'] for a in base):
                    a=np.array([r['l1'] for r in group]);b=np.array([r['l1'] for r in base]);ratio=a/b
                    rng=np.random.default_rng(config['bootstrap_seed']);ix=rng.integers(0,len(a),(config['bootstrap_samples'],len(a)))
                    record.update(mean_l1=float(a.mean()),median_l1=float(np.median(a)),max_l1=float(a.max()),
                        mean_ratio=float(ratio.mean()),median_ratio=float(np.median(ratio)),worst_ratio=float(ratio.max()),
                        ci95=np.quantile(np.mean(ratio[ix],axis=1),[.025,.975]),trajectories=len(a),
                        percent_change=float(100*(ratio.mean()-1)))
                stats.append(record)
    write(output/'statistics.json',stats)


def spatial():
    names,_=register_models();output=RUN/'results/spatial.json';rows=[]
    for profile in ['coupled','momentum','critical']:
        for name in ['historical_teno','fv_teno']+names:
            previous=None;rec=study.reconstruction(name)
            f=jax.jit(flux_variant(rec,'periodic',study.formulation(name),face_guard=name.startswith('safe')))
            from src.differentiable.euler_jax import flux_jax
            for n in [40,80,160,320]:
                u=jnp.asarray(study.averages(n,profile))
                exact=np.diff(np.asarray(flux_jax(jnp.asarray(study.profile(np.arange(n+1)/n,profile)))),axis=0)*n
                err=errors(np.asarray(f(u))*n,exact)
                rows.append(dict(profile=profile,variant=name,cells=n,**err,order=None if previous is None else np.log2(previous/err['l1'])))
                previous=err['l1']
    write(output,rows)


def scalar_transfer():
    """Euler-trained backbone on scalar problems: explicitly zero-shot transfer."""
    names,frozen=register_models();config=read_config();models={}
    for name,entry in zip(names,frozen['candidates']):models[name]=load_checkpoint(entry['checkpoint'],entry['sha256'])
    output=RUN/'results/scalar_transfer';output.mkdir(exist_ok=False);rows=[]
    old_root=OUT/'results/scalar'
    from src.differentiable.diagnostics import diagnostic_rollout
    for source in sorted(old_root.glob('*.json')):
        if source.name=='summary.json' or not source.name.endswith('_teno5.json'):continue
        base=json.loads(source.read_text());arrays=np.load(source.with_suffix('.npz'))
        n=base['cells'];system=base['system'];dt=base['dt'];steps=base['steps'];initial=arrays['initial'];target=arrays['reference']
        for name,model in models.items():
            rec=make_reconstruction(model,'gated-soft',config['strength'])
            fn=jax.jit(lambda u:diagnostic_rollout(u,rec,dt,1/n,steps,system,'periodic',wave_speed=.83,
                alpha='dynamic' if system=='burgers' else .83,return_device=True))
            tick=time.perf_counter();final,history=fn(jnp.asarray(initial));final.block_until_ready();final=np.asarray(final)
            row=dict(case=base['case'],system=system,cells=n,variant=name,zero_shot_euler_to_scalar=True,
                solver='unchanged scalar state LF + SSPRK3',dt=dt,time=.2,steps=steps,**errors(final,target),
                valid=bool(np.isfinite(final).all()),l1_ratio_to_classical=float(np.mean(abs(final-target))/base['l1']),
                tv=float(np.sum(abs(final-np.roll(final,1)))),overshoot=max(0.,float(final.max()-target.max())),
                undershoot=max(0.,float(target.min()-final.min())),conservation_residual=float(np.sum(final-initial)/n),
                wall_seconds_including_compilation=time.perf_counter()-tick,
                classical_source=str(source.relative_to(ROOT)),classical_source_sha256=sha256_file(source))
            write(output/f'{base["case"]}_N{n}_{name}.json',row)
            np.savez_compressed(output/f'{base["case"]}_N{n}_{name}.npz',initial=initial,final=final,reference=target)
            rows.append(row);print('zero-shot scalar',base['case'],n,name,row['l1'],flush=True)
    write(output/'summary.json',rows)


def performance():
    names,_=register_models()
    # Reuse the existing synchronized production timing code unchanged.
    from experiments.accuracy_upgrade import performance as timing
    timing.VARIANTS=('historical_teno','safe_fv_teno','safe_fv_v2_soft25',*names)
    timing.run(RUN/'results/performance.json')


def classification():
    """Training-only matched-state probe; no classifier-based selection."""
    names,frozen=register_models();config=read_config()
    from experiments.accuracy_upgrade.full_training import initial_params
    from experiments.accuracy_upgrade.train import make_pool
    from scripts.unrolled_physics_loss.Train_Gated_NNTENO5_Dataset_V2 import load_split
    from src.differentiable.teno5_jax_toy import toy_keep_weight
    data,_=load_split(ROOT/'data/nnteno_v2/data','euler','train');rows=[]
    @jax.jit
    def measure(params,x,y):
        keep=jnp.clip(toy_keep_weight(params,x),1e-7,1.-1e-7)
        loss=jnp.mean((-y*jnp.log(keep)-(1-y)*jnp.log1p(-keep))*jnp.where(y<.5,2.,1.))
        return loss,jnp.mean((keep>=.5)==(y>=.5))
    for n in config['train_grids']:
        x,y,ids=make_pool(data[n],'C',1337+n,8,15000)
        for entry in frozen['candidates']:
            seed=entry['seed']
            models=dict(initial=initial_params(seed),
                warmup=load_checkpoint(RUN/f'checkpoints/fv_conservative_seed{seed}_step0000.pkl').backbone,
                selected=load_checkpoint(entry['checkpoint'],entry['sha256']).backbone)
            for phase,params in models.items():
                bce,accuracy=measure(params,x,y)
                rows.append(dict(seed=seed,cells=n,phase=phase,weighted_bce=float(bce),accuracy=float(accuracy),
                    trajectory_ids=ids,stencils=len(x),selection_use=False,split='train only'))
    write(RUN/'results/classification_probe.json',rows)


def smoke():
    output=RUN/'results/evaluation_smoke';output.mkdir(exist_ok=False)
    case=study.cases()[0];ref,meta=reference(case,16)
    model=load_checkpoint(RUN/'checkpoints/fv_conservative_seed4101_step0000.pkl')
    name='safe_fv_warmup_smoke'
    study.EXTRA_RECONSTRUCTIONS[name]=make_reconstruction(model,'gated-soft',read_config()['strength'])
    rows=[execute(case,16,v,output,ref,meta) for v in ['safe_fv_teno',name]]
    assert all(r['success'] and np.isfinite(r['l1']) for r in rows)
    assert all(max(abs(x) for x in r['conservation_residual'])<1e-12 for r in rows)
    write(output/'summary.json',rows)


def diagnose():
    """Diagnose rejected validation checkpoints; never select them from tests."""
    config=read_config();output=RUN/'results/rejected_validation_diagnostics';output.mkdir(exist_ok=False)
    from experiments.accuracy_upgrade.full_training import build_predict
    from scripts.unrolled_physics_loss.Train_Gated_NNTENO5_Dataset_V2 import load_split
    data,_=load_split(ROOT/'data/nnteno_v2/data','euler','validation');rows=[]
    predictor=build_predict(config)
    for summary in json.loads((RUN/'results/training_summary.json').read_text()):
        seed=summary['seed']
        for event in summary['validation_records']:
            if event['failures']==0:continue
            step=event['step'];model=load_checkpoint(RUN/f'checkpoints/fv_conservative_seed{seed}_step{step:04d}.pkl')
            params=[(jnp.asarray(w),jnp.asarray(b)) for w,b in model.backbone]
            records=json.loads((RUN/f'results/seed{seed}_validation_step{step:04d}.json').read_text())['cases']
            name=f'safe_fv_rejected_seed{seed}_step{step}'
            study.EXTRA_RECONSTRUCTIONS[name]=make_reconstruction(model,'gated-soft',config['strength'])
            for old in records:
                if old['valid']:continue
                n=old['cells'];g=data[n];index=list(g['ids']).index(old['case_id'])
                lo=(index//config['validation_batch_size'])*config['validation_batch_size'];hi=min(lo+config['validation_batch_size'],len(g['ids']))
                _,d=predictor(params,jnp.asarray(g['initial'][lo:hi]),jnp.asarray(g['dt'][lo:hi]))
                d=jax.tree.map(np.asarray,d);i=index-lo
                mapped={k:np.asarray(v[i]).tolist() for k,v in d.items()}
                initial=g['initial'][index];dt=float(g['dt'][index]);target_time=dt*config['rollout_steps']
                final,diag,history,times,states=evolve(initial,name,'transmissive',target_time,1/n,dt)
                row=dict(seed=seed,step=step,case_id=old['case_id'],family=old['family'],cells=n,
                    mapped_failure_reproduced=not mapped['valid'],mapped_diagnostics=mapped,
                    bounded_retry_inference=diag,selection_use=False)
                if diag['success']:row['retry_solution_error']=errors(final,g['target'][index])
                rows.append(row);print('rejected validation diagnostic',seed,step,old['case_id'],mapped['max_cfl'],diag['success'],diag['retries'],flush=True)
    write(output/'summary.json',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['declare','baselines','problems','holdout','spatial','scalar','performance','classification','smoke','diagnose'])
    p.add_argument('--run-dir',type=Path,default=RUN);a=p.parse_args();RUN=configure_run(a.run_dir)
    {'declare':declare,'baselines':baseline_problems,'problems':benchmarks,'holdout':holdout,'spatial':spatial,'scalar':scalar_transfer,'performance':performance,'classification':classification,'smoke':smoke,'diagnose':diagnose}[a.action]()
