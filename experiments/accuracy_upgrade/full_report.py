"""Measured reports, figures and independent checks for conservative retraining."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.accuracy_upgrade.full_training import RUN,read_config,configure_run
from experiments.accuracy_upgrade.study import ROOT,OUT,write
from src.differentiable.model_io import sha256_file

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':240,'pdf.fonttype':42})
COLORS={'historical_teno':'#888888','historical_v2_soft25':'#cc8800','safe_fv_teno':'#222222',
        'safe_fv_v2_soft25':'#a65628','safe_fv_new_seed4101':'#0072b2','safe_fv_new_seed4102':'#009e73'}
LABELS={'historical_teno':'Historical TENO5','historical_v2_soft25':'Historical saved V2',
        'safe_fv_teno':'Conservative FV TENO5','safe_fv_v2_soft25':'Conservative FV + saved V2',
        'safe_fv_new_seed4101':'New FV model, seed 4101','safe_fv_new_seed4102':'New FV model, seed 4102'}


def load(p):return json.loads((RUN/p).read_text())
def number(x):return '—' if x is None else f'{x:.6g}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)+'\n'
def finish(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(RUN/f'plots/{name}.{ext}',bbox_inches='tight')
    plt.close(fig)
def history(seed):return [json.loads(x) for x in (RUN/f'results/seed{seed}_history.jsonl').read_text().splitlines()]


def figures():
    config=read_config();training=load('results/training_summary.json');problems=load('results/problems/summary.json')
    old=load('results/validation_saved_v2_safe_fv.json')['summary']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for run in training:
        seed=run['seed'];h=[r for r in history(seed) if r['phase']=='physics' and r['family']!='constant' and r['loss'] is not None]
        axes[0].semilogy([r['step'] for r in h],[r['loss'] for r in h],'.',alpha=.4,label=f'seed {seed}',ms=3)
        records=[dict(step=0,**run['warmup_validation'])]+run['validation_records']
        axes[1].plot([r['step'] for r in records],[r['macro_l1_ratio'] if r['failures']==0 else np.nan for r in records],'o-',label=f'seed {seed}')
        for r in records:
            if r['failures']:
                axes[1].text(.98,.96,f'seed {seed}, update {r["step"]}:\n{r["failures"]} failed; no accuracy score',
                             transform=axes[1].transAxes,ha='right',va='top',fontsize=7,color='red')
    axes[0].set(xlabel='PDE optimizer update',ylabel='Sampled normalized MSE',title='Nonconstant batches; all batches logged')
    axes[1].axhline(old['macro_l1_ratio'],color='#a65628',ls='--',label='saved V2 in same FV solver')
    axes[1].axhline(1.,color='k',ls=':',label='conservative FV TENO5')
    axes[1].set(xlabel='PDE optimizer update',ylabel='Validation macro paired L1 ratio',title='All 307 validation trajectories, N=64/128/256')
    for ax in axes:ax.legend(fontsize=7)
    fig.suptitle('3155-parameter backbone; canonical state warm-up; K=15 full conservative SSPRK3, fixed soft λ=.25')
    finish(fig,'training_and_validation')
    names=['safe_fv_teno','safe_fv_v2_soft25']+[f'safe_fv_new_seed{s}' for s in config['seeds']]
    for case in ['sod','lax','two_rarefactions','shu_osher_full']:
        fig,axes=plt.subplots(2,2,figsize=(11,7));n=256
        baseline=next(r for r in problems if r['case']==case and r['cells']==n and r['variant']=='safe_fv_teno')
        source=RUN/f'results/problems/{case}_N{n}_safe_fv_teno.npz';ref=np.load(source)['reference']
        x=(np.arange(n)+.5)*baseline['dx'];x=x-5 if case=='shu_osher_full' else x
        def pressure(u):return .4*(u[:,2]-.5*u[:,1]**2/u[:,0])
        axes[0,0].plot(x,ref[:,0],'k:',label='reference');axes[0,1].plot(x,pressure(ref),'k:')
        for name in names:
            row=next(r for r in problems if r['case']==case and r['cells']==n and r['variant']==name)
            if not row['success']:continue
            u=np.load(RUN/f'results/problems/{case}_N{n}_{name}.npz')['final']
            for ax,val in zip(axes[0],[u[:,0],pressure(u)]):ax.plot(x,val,color=COLORS[name],lw=1,label=LABELS[name])
            axes[1,0].plot(x,u[:,0]-ref[:,0],color=COLORS[name],lw=1)
            axes[1,1].plot(x,pressure(u)-pressure(ref),color=COLORS[name],lw=1)
        axes[0,0].set_ylabel('Density');axes[0,1].set_ylabel('Pressure');axes[1,0].set_ylabel('Density error');axes[1,1].set_ylabel('Pressure error')
        for ax in axes.ravel():ax.set_xlabel('x')
        axes[0,0].legend(fontsize=7)
        note='analytic Riemann cell averages' if case!='shu_osher_full' else 'archived 2048-cell dt/2 numerical reference; uncertainty retained'
        fig.suptitle(f'{case}, N=256, t={baseline["definition"]["time"]}; transmissive, SSPRK3, soft λ=.25\n{note}; all shown solvers use conservative safeguards')
        finish(fig,case+'_profiles')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,case in zip(axes,['sod','lax']):
        for name in ['historical_teno','safe_fv_teno','safe_fv_v2_soft25']+names[2:]:
            rows=sorted([r for r in problems if r['case']==case and r['variant']==name and r['success']],key=lambda r:r['cells'])
            ax.loglog([r['cells'] for r in rows],[r['l1'] for r in rows],'o-',label=LABELS[name],color=COLORS[name])
        ax.set(xlabel='N',ylabel='Conserved-component L1 error',title=case)
    axes[0].legend(fontsize=7);fig.suptitle('Resolution tests through unseen N=512; exact cell averages; shock-containing solutions')
    finish(fig,'resolution_generalization')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,case in zip(axes,['sod_long','lax']):
        for name in names:
            row=next(r for r in problems if r['case']==case and r['cells']==256 and r['variant']==name)
            if not row['success']:continue
            ax.plot([r['time'] for r in row['error_over_time']],[r['l1'] for r in row['error_over_time']],label=LABELS[name],color=COLORS[name])
        ax.set(xlabel='Physical time',ylabel='L1 error to analytic cell averages',title=case)
    axes[0].legend(fontsize=7);fig.suptitle('Frozen validation-selected models; N=256, conservative FV SSPRK3; no test-time tuning')
    finish(fig,'error_over_time')
    stats=load('results/fresh_holdout/statistics.json');fig,axes=plt.subplots(1,2,figsize=(10,4))
    methods=['safe_fv_v2_soft25']+names[2:]
    for ax,n in zip(axes,config['holdout_grids']):
        for i,name in enumerate(methods):
            row=next(r for r in stats if r['cells']==n and r['variant']==name and r['baseline']=='safe_fv_teno')
            if 'mean_ratio' not in row:continue
            lo,hi=row['ci95'];mean=row['mean_ratio']
            ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color=COLORS[name])
        ax.axhline(1.,color='k',ls=':');ax.set_xticks(range(len(methods)),['Saved V2','New 4101','New 4102'],fontsize=9)
        ax.set(ylabel='Mean paired L1 ratio\n(classical conservative FV = 1)',title=f'N={n}; eight independent IVPs')
    fig.suptitle('Fresh holdout; same conservative FV solver, soft λ=.25; paired 95% case-bootstrap intervals')
    finish(fig,'fresh_holdout')
    perf=load('results/performance.json');fig,ax=plt.subplots(figsize=(7.5,4.5))
    for name in ['historical_teno']+names:
        rows=[r for r in perf['rows'] if r['variant']==name]
        ax.loglog([r['warm_median_s'] for r in rows],[r['l1'] for r in rows],'o-',label=LABELS[name],color=COLORS[name])
    ax.set(xlabel='Synchronized warmed CPU simulation seconds',ylabel='L1 error',title='Sod t=.15; N=64/128/256; nine production repeats')
    ax.legend(fontsize=8);finish(fig,'accuracy_and_cost')
    spatial=load('results/spatial.json');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,profile in zip(axes,['coupled','critical']):
        for name in ['historical_teno','fv_teno']+names[2:]:
            rows=[r for r in spatial if r['profile']==profile and r['variant']==name]
            ax.loglog([r['cells'] for r in rows],[r['l1'] for r in rows],'o-',label=name+f' (p={rows[-1]["order"]:.2f})')
        ax.set(xlabel='N',ylabel='Spatial flux-divergence L1 error',title=profile)
    axes[0].legend(fontsize=6);fig.suptitle('Exact smooth Euler cell averages; periodic, instantaneous spatial diagnostic; no temporal error')
    finish(fig,'smooth_spatial_accuracy')
    fig,axes=plt.subplots(2,2,figsize=(10,6));n=128;x=(np.arange(n)+.5)/n
    for i,case in enumerate(['square','burgers_hat']):
        base=np.load(OUT/f'results/scalar/{case}_{n}_teno5.npz');target=base['reference']
        axes[i,0].plot(x,target,'k:',label='analytic cell averages')
        old=np.load(OUT/f'results/scalar/{case}_{n}_v2_soft25.npz')['final']
        axes[i,0].plot(x,old,color='#a65628',lw=1,label='saved scalar V2')
        axes[i,1].plot(x,old-target,color='#a65628',lw=1)
        for name in names[2:]:
            u=np.load(RUN/f'results/scalar_transfer/{case}_N{n}_{name}.npz')['final']
            axes[i,0].plot(x,u,color=COLORS[name],lw=1,label=LABELS[name])
            axes[i,1].plot(x,u-target,color=COLORS[name],lw=1)
        axes[i,0].set_ylabel(case+' solution');axes[i,1].set_ylabel('Local scalar error')
        for ax in axes[i]:ax.set_xlabel('x')
    axes[0,0].legend(fontsize=7);fig.suptitle('Zero-shot Euler-to-scalar transfer; unchanged scalar LF/SSPRK3, N=128, t=.2, periodic, λ=.25')
    finish(fig,'scalar_zero_shot_transfer')


def report():
    config=read_config();training=load('results/training_summary.json');problems=load('results/problems/summary.json')
    holdout=load('results/fresh_holdout/statistics.json');scalar=load('results/scalar_transfer/summary.json');perf=load('results/performance.json')
    old=load('results/validation_saved_v2_safe_fv.json')['summary']
    primary=min(training,key=lambda r:r['best_validation_ratio'])
    primary_name=f'safe_fv_new_seed{primary["seed"]}'
    analytic=[r for r in problems if r['variant']==primary_name and r['definition']['kind']=='riemann' and r['success']]
    compared=[r for r in analytic if 'ratio_to_safe_fv_v2_soft25' in r]
    wins=sum(r['ratio_to_safe_fv_v2_soft25']<1.-1e-8 for r in compared)
    regressions=sum(r['ratio_to_safe_fv_v2_soft25']>1.+1e-8 for r in compared)
    worst_old=max(compared,key=lambda r:r['ratio_to_safe_fv_v2_soft25'])
    worst_classical=max(analytic,key=lambda r:r.get('ratio_to_safe_fv_teno',0))
    validation_change=100*(primary['best_validation_ratio']/old['macro_l1_ratio']-1.)
    confirmation=[r for r in holdout if r['variant'].startswith('safe_fv_new') and r['baseline']=='safe_fv_v2_soft25']
    confirmation_text='; '.join(f'{r["variant"]}, N={r["cells"]}: {r["percent_change"]:+.2f}%' for r in confirmation if 'percent_change' in r)
    no_limiting=all(r['limited_fe_stages']==0 and r['limited_face_stages']==0 and r['retries']==0 for r in load('results/fresh_holdout/per_case.json'))
    methods=['historical_teno','historical_v2_soft25','safe_fv_teno','safe_fv_v2_soft25']+[f'safe_fv_new_seed{s}' for s in config['seeds']]
    train_table=table(['Seed','Selected attempt','Applied / attempted PDE updates','Validation L1 ratio / conservative classical','Relative to old V2, same solver','Nonfinite gradient updates','Rejected training updates','Wall seconds'],[
        [r['seed'],r['best_step'],f'{sum(x["accepted"] for x in history(r["seed"]) if x["phase"]=="physics")}/{config["physics_updates"]}',number(r['best_validation_ratio']),number(r['best_validation_ratio']/old['macro_l1_ratio']),r['nonfinite_gradient_updates'],r['invalid_rollout_updates'],number(r['wall_seconds'])] for r in training])
    active_rows=[]
    for seed in config['seeds']:
        h=[r for r in history(seed) if r['phase']=='physics']
        active_rows.append([seed,sum(r['raw_invalid_rows'] for r in h),sum(r['flux_limited'] for r in h),
            sum(r['face_limited'] for r in h),number(min(r['minimum_density'] for r in h)),
            number(min(r['minimum_pressure'] for r in h)),number(max(r['max_cfl'] for r in h))])
    active_table=table(['Seed','Pre-flux-correction invalid cell rows','Global flux-stage activations','Global face-stage activations','Min density','Min pressure','Max low-order CFL'],active_rows)
    diagnostics=load('results/rejected_validation_diagnostics/summary.json')
    diagnostic_table=table(['Rejected checkpoint case','Mapped failure reproduced','Max low-order CFL','Adaptive inference completed','Retry count','Max conservation residual'],[
        [r['case_id'],r['mapped_failure_reproduced'],number(r['mapped_diagnostics']['max_cfl']),
         r['bounded_retry_inference']['success'],r['bounded_retry_inference']['retries'],
         number(max(abs(x) for x in r['bounded_retry_inference']['conservation_residual']))] for r in diagnostics])
    failed_validations=[(run['seed'],r['step'],r['failures']) for run in training for r in run['validation_records'] if r['failures']]
    failure_note=('No validation checkpoint failed.' if not failed_validations else
        'Rejected validation checkpoints: '+', '.join(f'seed {s}, update {step}: {count} failed fixed-dt trajectories' for s,step,count in failed_validations)+
        '. Their partial-rollout errors are diagnostic only and are not valid accuracy scores. They are excluded from the validation accuracy curve and checkpoint selection. Inference can retry smaller timesteps, but fixed-time training/validation does not silently change its 15-step budget.')
    err_table=table(['Case','N','Conservative FV TENO5 L1','Same solver + old V2 L1','New seed 4101 L1','New seed 4102 L1'],[
        [case,n]+[number(next(r for r in problems if r['case']==case and r['cells']==n and r['variant']==name).get('l1')) for name in methods[2:]]
        for case in ['sod','lax','two_rarefactions','contrast','near_vacuum','shu_osher_full'] for n in [128,256]])
    stability=table(['Variant','Completed / attempted','Minimum accepted density','Minimum accepted pressure','Max mass residual','Max momentum residual','Max energy residual'],[
        [name,f'{sum(r["success"] for r in problems if r["variant"]==name)}/{sum(r["variant"]==name for r in problems)}']+
        [number(min(r[k] for r in problems if r['variant']==name and r['success'])) for k in ['minimum_density','minimum_pressure']]+
        [number(max(abs(r['conservation_residual'][k]) for r in problems if r['variant']==name and r['success'])) for k in range(3)]
        for name in sorted(set(r['variant'] for r in problems))])
    hold_table=table(['N','Model','Baseline','Mean paired ratio','95% paired interval','Worst trajectory ratio'],[
        [r['cells'],r['variant'],r['baseline'],number(r.get('mean_ratio')),
         '—' if 'ci95' not in r else f'{r["ci95"][0]:.5f}–{r["ci95"][1]:.5f}',number(r.get('worst_ratio'))]
        for r in holdout if r['variant'].startswith('safe_fv_new') and r['baseline'] in ['safe_fv_teno','safe_fv_v2_soft25']])
    gen_table=table(['Case','Variant','N=512 L1','Ratio to same-solver classical','Ratio to old V2, same solver'],[
        [r['case'],r['variant'],number(r.get('l1')),number(r.get('ratio_to_safe_fv_teno')),number(r.get('ratio_to_safe_fv_v2_soft25'))]
        for r in problems if r['cells']==512])
    scalar_table=table(['Problem','N','Seed 4101 / classical L1','Seed 4102 / classical L1'],[
        [case,n]+[number(next(r for r in scalar if r['case']==case and r['cells']==n and r['variant']==name)['l1_ratio_to_classical']) for name in methods[-2:]]
        for case in sorted(set(r['case'] for r in scalar)) for n in [64,128,256]])
    performance_table=table(['Variant, N=128','Warmed simulation seconds','Compilation seconds','L1 error'],[
        [r['variant'],number(r['warm_median_s']),number(r['compile_s']),number(r['l1'])] for r in perf['rows'] if r['cells']==128])
    classification=load('results/classification_probe.json')
    class_table=table(['Seed','N','Initial BCE / accuracy','Warm-up BCE / accuracy','Selected BCE / accuracy'],[
        [seed,n]+[f'{number(next(r for r in classification if r["seed"]==seed and r["cells"]==n and r["phase"]==phase)["weighted_bce"])} / {number(next(r for r in classification if r["seed"]==seed and r["cells"]==n and r["phase"]==phase)["accuracy"])}' for phase in ['initial','warmup','selected']]
        for seed in config['seeds'] for n in config['train_grids']])
    regression=(RUN/'results/cpu_regression.log').read_text();passed=re.search(r'(\d+) passed',regression).group(1)
    legacy=(RUN/'results/legacy_tests.log').read_text();legacy_passed=re.search(r'(\d+) passed',legacy).group(1)
    text=f'''# Full conservative FV NN-TENO5 training and problem tests

Two new models were trained from scratch, with the existing 3155-parameter backbone, through the actual finite-volume-consistent Euler solver and its conservative SSPRK3 safeguard. The primary candidate is **seed {primary['seed']}, PDE update {primary['best_step']}**, chosen solely by validation error. Its checkpoint is:

`{primary['best_checkpoint']}`

SHA-256: `{primary['best_sha256']}`. Both seeds, all evaluated checkpoints, final optimizer/sampler states and complete histories are retained. No old checkpoint, dataset or result is overwritten. This report presents all requested completed comparisons; lower validation loss alone is not a general accuracy or stability claim.

The validation-selected primary changes validation macro L1 by **{validation_change:+.3f}% relative to old V2 in the same conservative FV solver**. On the tested analytic canonical configurations it has lower error than old V2 in {wins}/{len(compared)} and higher error in {regressions}/{len(compared)}. These are correlated configuration counts, not independent statistical samples. Its worst ratio to old V2 is **{worst_old['ratio_to_safe_fv_v2_soft25']:.4f}** on {worst_old['case']} at N={worst_old['cells']}; its worst ratio to conservative classical FV TENO5 is **{worst_classical['ratio_to_safe_fv_teno']:.4f}** on {worst_classical['case']} at N={worst_classical['cells']}. These regressions must accompany favorable means. Fresh-holdout uncertainty and cost are reported separately below.

**Recommendation: retain the old saved V2 weights as the preferred existing neural checkpoint; do not promote these new weights as a general replacement.** Fresh-holdout mean L1 changes relative to old V2 in the same solver are: {confirmation_text}. The improved short-horizon validation score did not generalize to this predeclared set. No post-holdout retuning is performed. This is a measured negative generalization result, not a failed training job.

## Training protocol and what changed

- Architecture: 5→64→32→16→8→3, unchanged; deterministic TENO gate, soft neural weights and fixed λ=.25. No richer mixing head or architecture search.
- Canonical warm-up: both directions of the actual characteristic state stencils, 150 teacher steps, batch 256, at most 15000 balanced training stencils, sampled from training trajectories at N=64/128/256.
- Physics training: 300 attempted Adam updates per seed, batch 8, learning rate .0002, global-norm clipping at 1; K=15 actual SSPRK3 steps at each trajectory's own dt. All three grids participate. The objective is per-trajectory component-normalized solution MSE; no TV/BVD terms are added. Rejected batches consume an attempt and are not resampled away; applied-update counts are reported separately.
- Every SSPRK3 FE-like stage uses physical face scaling and shared high/low flux correction. Conservative averages are never repaired. Limiter coefficients and switching decisions are stop-gradient. Failed fixed-time training rollouts or nonfinite gradients reject the optimizer update; three consecutive rejections stop training.
- `lax.map` preserves lazy safety branches across independent training trajectories. The gradient smoke and invalid-batch rejection tests pass.
- Validation uses all 307 Euler validation trajectories at N=64/128/256. Select the smallest zero-failure macro family×grid mean paired L1 ratio against conservative classical FV TENO5, including warm-up. No original test data are read. Both seeds have identical declared budgets; exact sampling histories are logged.

This is a larger experiment than the earlier N=64/60-update pilots. It matches the saved V2 architecture, declared teacher/PDE-attempt budgets, batch size and grid set, but changes the solver, safeguard, normalization and sampling. It is not an input-only causal ablation. The old V2 checkpoint is explicitly reevaluated in the **same** conservative solver to separate training effects from numerical-formulation effects.

In the FV path, λ mixes characteristic **state reconstructions before** evaluating the nonlinear Rusanov flux. This is not a convex blend of two already-computed physical Euler fluxes. The independent admissibility coefficient θ limits shared interface fluxes afterwards. Neither a bounded λ nor neural candidate probabilities alone establish positivity, TVD or entropy stability.

## Validation results and checkpoint selection

Old V2 in the same conservative solver has validation macro L1 ratio **{old['macro_l1_ratio']:.6f}** to classical FV TENO5.

{train_table}

Validation curves include every 50-update evaluation, including unfavorable later evaluations. Final training checkpoints are not automatically preferred. Training times include compilation, validation, logging and concurrent regression/control work; do not compare seed timings as pure optimizer throughput. Per-grid first compilations, warmed update medians, loss, gradient norm, limiter activation, stage minima and actual batch case IDs are saved in the histories.

{failure_note}

Diagnostic replays of rejected validation trajectories, without using them to select or retune a model:

{diagnostic_table}

The configured low-order CFL ceiling is .9. The mapped replay retains the original batch shape and parameter structure. Bounded-retry inference keeps the physical target time and records accepted boundary exchanges; it is not a replacement for the failed fixed-step validation score or a reason to promote that rejected checkpoint.

Training-stage diagnostics, across all sampled batches:

{active_table}

The safeguard is part of the actual rollout even where it is inactive. Inactive training corrections cannot establish that the network learned positivity or learned how to handle blast/near-vacuum conditions. Any stability provided by active test-time limiting belongs to the numerical safeguard, rather than being credited to the neural network.

## Canonical Euler problems

All methods use identical initial cell averages, proposed timesteps, final physical times and references. The FV controls use the same safeguard as the newly trained networks. Historical TENO5, historical V2, the historical P0 learned model and unlimited FV controls are also rerun. Bounded inference retries advance only accepted physical time. A failed trajectory has no accuracy score.

Aggregate L1 averages unscaled conserved components with different units; density/momentum/energy L1/L2/Linf and pressure error are also exported. Dashes indicate failed accuracy trajectories.

{err_table}

Sod is evaluated at t=.15 and again at t=.2. Lax uses t=.1, two-rarefactions t=.08, contrast t=.012 and near-vacuum t=.1. Short-time and full-time Shu–Osher use t=.15 and t=1.8 on length 10. Shock positions/speeds, contact 10–90% widths, density/pressure TV and extrema are exported for applicable analytic cases. Contact width is a diagnostic in a declared fixed neighborhood, not a universal sharpness metric.

## Stability and conservation

{stability}

The suite includes reflecting-wall blast smoke t=.001 and full-time t=.038. These are **stability tests only** because no trustworthy blast accuracy reference is supplied. Wall mass/energy fluxes are zero; momentum budgets include wall pressure force. Reported conservation residuals include accepted boundary exchanges with SSPRK weights. Large blast energy scales and accumulated roundoff must be considered alongside absolute residuals.

Every failed historical/unlimited trajectory remains in the machine-readable records with partial physical time and first invalid step/stage. Accepted-attempt pre-correction invalid counts, limiter activations and total retries are recorded; detailed per-retry invalid-row counts are not aggregated by the reused trajectory harness. No hidden state clipping is used to conceal failures.

## Fresh confirmation and resolution generalization

Eight independent Riemann/contact/rarefaction IVPs were seeded with 2026111203 and hashed before training. Weights were frozen after validation selection, before any new-model holdout evaluation. These differ from the previously examined holdout. No strength, optimizer, checkpoint or architecture is selected using them. Intervals use 2000 paired trajectory resamples separately at each grid, seed 2026111204. They condition on a trained model and do not quantify training-seed uncertainty. Two resolutions of an IVP and multiple times are not independent trajectories.

{hold_table}

No safeguard activation or retry occurred anywhere in this confirmation comparison: **{no_limiting}**. Thus an active limiter is not the explanation for its observed model-to-model accuracy differences. Contact cases have particularly unfavorable new-model ratios; the per-case files retain the full family breakdown. Short-horizon fitting or rollout-distribution shift are possible mechanisms, not isolated causal findings.

N=512 was absent from the entire training/validation grid set. Canonical Sod/Lax conditions themselves were previously examined, so this is a grid-transfer test, not a fresh-physics holdout:

{gen_table}

## Reference uncertainty and smooth accuracy

Sod/Lax/Riemann references are analytic cell averages. Short-time Shu uses the earlier FV state reference ladder 256/512/1024 plus finest dt/2. Full-time Shu reuses the existing matched canonical cell-average reference ladder 512/1024/2048 plus finest dt/2. Source hashes and the original scheme identity are retained. The archived finest temporal array was not exported in full; its 256-cell projection is used only at N≤256. It is never fabricated or interpolated to a finer target grid.

For numerical references, twice the final spatial plus temporal Cauchy change is an empirical uncertainty proxy, not an error bound. A method ranking is marked unresolved unless the proxy is below one quarter of the error margin. Full-time Shu profiles and numerical-reference errors are reported even when a ranking is unresolved; these values alone do not establish superiority.

Smooth spatial tests use exact coupled/momentum/critical-point cell averages, periodic boundaries, N=40/80/160/320 and exact instantaneous flux divergences. All absolute norms/orders are in `results/spatial.json`. They do not establish fifth-order time integration: SSPRK3 is third order in time. Existing critical-point limitations remain relevant.

## Classifier diagnostics

Training-only matched-state probes, independent of checkpoint selection:

{class_table}

BCE/accuracy measure teacher imitation, not PDE solution quality. Accuracy is mean per-candidate binary keep/discard agreement, not exact whole-mask accuracy. Numerical validation remains the selection criterion.

## Scalar transfer tests

The new networks were trained on Euler data. The following are explicitly **zero-shot Euler-to-scalar transfer** tests through the unchanged scalar solver, with classical analytic references and matched settings from the archived scalar controls. They are not presented as newly scalar-trained models or as evidence of post-interaction Burgers accuracy.

{scalar_table}

Advection includes sine, Fourier, Gaussian and square profiles; Burgers includes a pre-shock smooth wave and a moving top-hat entropy shock before wave interaction. L1/L2/Linf, TV, extrema and conservative residuals are saved. A ratio above one is a regression, including for tiny smooth errors.

## Isolated CPU cost

After all other experiment workers finish: nine synchronized production repeats, separate compilation/first execution, N=64/128/256, Sod t=.15. The same existing timing implementation is reused with the new frozen models.

{performance_table}

`results/performance.json` also contains measured discrete-grid equal-budget comparisons, per-step costs, executable temporary-memory estimates and lifetime process RSS. Lower same-grid error does not imply better accuracy per unit time. Process RSS is not per-model allocation; compiler estimates are not GPU peak measurements. No GPU claim is made.

## Tests, artifacts and limits

The full CPU suite passes **{passed} tests and 37 subtests**; the separate legacy environment passes **{legacy_passed} tests**. Four new tests cover full-safeguard gradients and conservation, zero-strength identity, rejection of invalid batched updates, and checkpoint compatibility/determinism. The original experiment's artifact verifier passes before and after training. Its seven already-missing ignored historical checkpoints remain an explicitly recorded baseline limitation.

The original method, original checkpoints and earlier results remain intact. Reproduction and inference instructions are in `../README.md`; exact configuration and candidate hashes are under `../configs/`. Failed/unfavorable runs are retained. No Git push is performed.

Remaining limits: two training seeds, 300-update short-horizon supervision, Euler Riemann-style training data, a small eight-IVP fresh holdout, numerical-reference uncertainty for Shu–Osher, no blast accuracy reference, and no scalar-specific retraining. The global safeguard can be dissipative. Any preferred-model decision must consider problem-wise regressions and CPU cost, rather than validation mean alone.
'''
    with (RUN/'reports/TRAINING_RESULTS.md').open('x') as f:f.write(text)
    write(RUN/'results/primary_candidate.json',dict(selection='minimum validation macro ratio across seeds; no problem/holdout tuning',
        seed=primary['seed'],checkpoint=primary['best_checkpoint'],sha256=primary['best_sha256'],step=primary['best_step']))
    # Flat CSV accompanies the full JSON; failures have blank error fields.
    columns=['case','cells','variant','success','l1','l2','linf','pressure_l1','minimum_density','minimum_pressure',
        'limited_fe_stages','limited_face_stages','retries','ratio_to_safe_fv_teno','ratio_to_safe_fv_v2_soft25']
    with (RUN/'results/problem_comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(problems)


def verify():
    frozen=load('configs/frozen_candidates.json');assert sha256_file(RUN/'configs/fresh_holdout.json')==frozen['holdout_sha256']
    provenance=load('configs/provenance.json')
    assert sha256_file(RUN/'configs/training.json')==provenance['config_sha256']
    assert frozen['holdout_sha256']==provenance['holdout_sha256']
    for c in frozen['candidates']:assert sha256_file(ROOT/c['checkpoint'])==c['sha256']
    metric_profiles=0
    for group,name in [('problems','summary.json'),('fresh_holdout','per_case.json')]:
        for row in load(f'results/{group}/{name}'):
            source=RUN/f'results/{group}/{row["case"]}_N{row["cells"]}_{row["variant"]}.npz'
            data=np.load(source)
            if not row['success']:
                assert 'l1' not in row
                continue
            assert abs(row['final_time']-row['definition']['time'])<1e-12
            final=data['final'];rho=final[:,0];p=.4*(final[:,2]-.5*final[:,1]**2/rho)
            assert np.isfinite(final).all() and np.all(rho>0) and np.all(p>0)
            balance=row['dx']*np.sum(final-data['initial'],axis=0)+row['boundary_exchange']
            np.testing.assert_allclose(balance,row['conservation_residual'],rtol=1e-12,atol=1e-14)
            if data['reference'].size:
                diff=final-data['reference']
                for key,val in dict(l1=np.mean(abs(diff)),l2=np.sqrt(np.mean(diff*diff)),linf=np.max(abs(diff))).items():
                    np.testing.assert_allclose(val,row[key],rtol=1e-12,atol=1e-14)
                metric_profiles+=1
    for entry in frozen['candidates']:
        meta=json.loads((ROOT/entry['checkpoint']).with_suffix('.json').read_text())
        assert meta['config']==read_config()
        assert not set(meta['eligible_training_ids']).intersection(meta['validation_ids'])
        assert set(meta['warmup_ids']).issubset(meta['eligible_training_ids'])
        assert meta['holdout_sha256']==frozen['holdout_sha256']
    from experiments.accuracy_upgrade.verify import verify as verify_previous
    previous=verify_previous()
    manifest=RUN/'artifact_hashes.json'
    if manifest.exists():
        hashes=json.loads(manifest.read_text())
    else:
        paths=[p for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['artifact_hashes.json','verification.json']]
        paths.extend(OUT/name for name in ['full_training.py','full_evaluation.py','full_report.py'])
        paths.append(ROOT/'tests/test_accuracy_upgrade_training.py')
        hashes={os.path.relpath(p,RUN):sha256_file(p) for p in sorted(paths)}
        write(manifest,hashes)
    for name,expected in hashes.items():
        path=(RUN/name).resolve()
        assert path.is_relative_to(ROOT),name
        assert sha256_file(path)==expected,name
    result=dict(passed=True,metric_profiles_recomputed=metric_profiles,new_artifact_hashes_checked=len(hashes),
        frozen_candidates=len(frozen['candidates']),previous_experiment_preservation=previous)
    if not (RUN/'results/verification.json').exists():write(RUN/'results/verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['figures','report','verify'])
    p.add_argument('--run-dir',type=Path,default=RUN);a=p.parse_args();RUN=configure_run(a.run_dir)
    {'figures':figures,'report':report,'verify':verify}[a.action]()
