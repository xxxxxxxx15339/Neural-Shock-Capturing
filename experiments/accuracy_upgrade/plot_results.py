"""Figures from measured artifacts only. Each figure labels its configuration."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.accuracy_upgrade.study import OUT,VARIANTS

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.dpi':240,'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.2})
LABELS={'historical_teno':'Historical TENO5','historical_p0_learned':'Historical P0 learned mix',
        'historical_v2_soft25':'Historical V2, soft λ=.25','fv_teno':'FV state TENO5',
        'fv_v2_soft25':'FV state V2, soft λ=.25','safe_fv_teno':'Safeguarded FV TENO5',
        'safe_fv_v2_soft25':'Safeguarded FV V2, soft λ=.25'}
COLORS=dict(zip(VARIANTS,['#555555','#888888','#bf6c00','#0072b2','#009e73','#7b3294','#d55e00']))
def load(p):return json.loads((OUT/p).read_text())
def finish(fig,name):
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/f'plots/{name}.{suffix}',bbox_inches='tight')
    plt.close(fig)
def p(u):return .4*(u[:,2]-.5*u[:,1]**2/u[:,0])

def spatial():
    data=load('results/spatial.json');fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,kind in zip(axes,['coupled','momentum','critical']):
        for v in ['historical_teno','historical_v2_soft25','fv_teno','fv_v2_soft25']:
            rows=[r for r in data if r['profile']==kind and r['variant']==v]
            ax.loglog([r['cells'] for r in rows],[r['l1'] for r in rows],'o-',color=COLORS[v],
                label=LABELS[v]+f' (last p={rows[-1]["order"]:.2f})',lw=1.5,ms=4)
        ax.set(title=kind,xlabel='Cell count N',ylabel='Mean conserved-component spatial L1 error')
        ax.legend(fontsize=7)
    fig.suptitle('Exact cell averages → instantaneous Euler flux divergence; periodic, float64; no time error')
    finish(fig,'smooth_spatial_convergence')
    data=load('results/temporal.json');fig,ax=plt.subplots(figsize=(7,4))
    for v in ['historical_teno','fv_teno','fv_v2_soft25','safe_fv_teno']:
        r=[r for r in data if r['variant']==v and r['refinement']==1]
        ax.loglog([x['cells'] for x in r],[x['l1'] for x in r],'o-',color=COLORS[v],label=LABELS[v])
    ax.set(xlabel='N',ylabel='L1 error at t=.02',title='Euler entropy wave; SSPRK3, dt≈0.2 dx^(5/3), periodic')
    ax.legend(fontsize=8);finish(fig,'smooth_time_dependent')

def profiles():
    root=OUT/'results/euler_actual_stages'
    for case in ['sod','two_rarefactions','blast_full']:
        fig,axes=plt.subplots(2,2,figsize=(11,7));n=128;x=(np.arange(n)+.5)/n
        reference=np.load(root/f'{case}_{n}_historical_teno.npz')['reference']
        if reference.size:
            axes[0,0].plot(x,reference[:,0],'k:',label='Exact cell averages');axes[0,1].plot(x,p(reference),'k:')
        for v in ['historical_teno','historical_v2_soft25','fv_teno','fv_v2_soft25','safe_fv_teno','safe_fv_v2_soft25']:
            info=json.loads((root/f'{case}_{n}_{v}.json').read_text())
            if not info['success']:continue
            u=np.load(root/f'{case}_{n}_{v}.npz')['final'];color=COLORS[v]
            axes[0,0].plot(x,u[:,0],color=color,label=LABELS[v],lw=1)
            axes[0,1].plot(x,p(u),color=color,lw=1)
            if reference.size:
                axes[1,0].plot(x,u[:,0]-reference[:,0],color=color,lw=1)
                axes[1,1].plot(x,p(u)-p(reference),color=color,lw=1)
            else:
                h=info['history'];axes[1,0].plot([r['time'] for r in h],[r['min_pressure'] for r in h],color=color)
                axes[1,1].plot([r['time'] for r in h],[r['face_theta_min'] for r in h],color=color)
        axes[0,0].set_ylabel('Density');axes[0,1].set_ylabel('Pressure')
        axes[1,0].set_ylabel('Density error' if reference.size else 'Minimum pressure through time')
        axes[1,1].set_ylabel('Pressure error' if reference.size else 'Global face scaling coefficient')
        for ax in axes[0]:ax.set_xlabel('x')
        for ax in axes[1]:ax.set_xlabel('x' if reference.size else 'time')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'{case}, N=128; SSPRK3, matched proposed dt; retries reported separately\n'
                     +('Reflecting walls; valid completed trajectories only; no accuracy reference' if case=='blast_full' else 'Transmissive; analytic Riemann reference; valid completed trajectories only'))
        finish(fig,f'euler_{case}_profiles')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,case in zip(axes,['sod','lax','two_rarefactions']):
        for v in VARIANTS:
            info=json.loads((root/f'{case}_128_{v}.json').read_text())
            if not info['success']:continue
            h=info['error_over_time'];ax.plot([r['time'] for r in h],[r['l1'] for r in h],color=COLORS[v],label=LABELS[v])
        ax.set(title=case,xlabel='Physical time',ylabel='Mean conserved-component L1 error')
    axes[0].legend(fontsize=7);fig.suptitle('N=128; SSPRK3; transmissive; exact cell-average references; failed variants excluded from curves')
    finish(fig,'euler_error_over_time')
    refs=load('results/shu_reference/comparisons.json');fig,axes=plt.subplots(1,2,figsize=(11,4));n=128;x=10*(np.arange(n)+.5)/n-5
    ref=np.load(OUT/f'results/shu_reference/reference_N{n}.npz')['final']
    axes[0].plot(x,ref[:,0],'k:',label='FV TENO5 1024, dt/2, restricted')
    for v in ['historical_teno','historical_v2_soft25','fv_teno','fv_v2_soft25']:
        u=np.load(root/f'shu_osher_{n}_{v}.npz')['final'];axes[0].plot(x,u[:,0],label=LABELS[v],color=COLORS[v],lw=1)
        axes[1].plot(x,u[:,0]-ref[:,0],color=COLORS[v],lw=1)
    axes[0].legend(fontsize=7);axes[0].set_ylabel('Density');axes[1].set_ylabel('Density error to numerical reference')
    for ax in axes:ax.set_xlabel('x')
    fig.suptitle('Short-time Shu–Osher, t=.15, N=128, transmissive; reference uncertainty reported separately')
    finish(fig,'shu_osher_short_time')

def scalar():
    fig,axes=plt.subplots(2,2,figsize=(10,6));root=OUT/'results/scalar';n=128;x=(np.arange(n)+.5)/n
    for i,kind in enumerate(['square','burgers_hat']):
        ref=np.load(root/f'{kind}_{n}_teno5.npz')['reference'];axes[i,0].plot(x,ref,'k:',label='Exact cell averages')
        for name in ['teno5','v2_soft25','p0_learned']:
            u=np.load(root/f'{kind}_{n}_{name}.npz')['final'];axes[i,0].plot(x,u,label=name,lw=1);axes[i,1].plot(x,u-ref,label=name,lw=1)
        axes[i,0].set_ylabel(kind+' solution');axes[i,1].set_ylabel('Local error')
        for ax in axes[i]:ax.set_xlabel('x')
    axes[0,0].legend(fontsize=8);fig.suptitle('Unchanged scalar solvers; N=128, t=.2, periodic, SSPRK3; V2 soft λ=.25')
    finish(fig,'scalar_profiles')

def training():
    data=load('checkpoints/paired_pilot/summary.json');fig,axes=plt.subplots(1,2,figsize=(10,4))
    for seed in [3101,3102]:
        rows=[r for r in data if r['seed']==seed]
        axes[0].plot([r['variant'] for r in rows],[r['warm_classifier_bce'] for r in rows],'o-',label=f'seed {seed}')
        axes[1].plot([r['variant'] for r in rows],[r['final_validation']['mean_l1'] for r in rows],'o-',label=f'seed {seed}')
    axes[0].set_ylabel('Teacher weighted BCE on matched training probes');axes[1].set_ylabel('Validation mean L1 (nonconstant, N=64/128)')
    for ax in axes:ax.set_xlabel('A: mismatched / B: split matched / C: FV matched');ax.legend()
    fig.suptitle('Paired pilot: 150 teacher steps + 60 × 15-step PDE updates; 3155-parameter backbone')
    finish(fig,'paired_training')

def gate():
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from src.differentiable.model_io import load_checkpoint
    from src.differentiable.accuracy_upgrade import state_stencils
    from src.differentiable.gated_teno5_jax import gated_reconstruction_diagnostics
    u=np.load(OUT/'results/euler_actual_stages/sod_128_fv_v2_soft25.npz')['final'];plus,minus=state_stencils(jnp.asarray(u))
    model=load_checkpoint('models/v2/euler_gated_dataset_v2_seed2026.pkl')
    d=gated_reconstruction_diagnostics(model.backbone,plus,'soft',.25)
    strength=.25*np.asarray(d['gate']);x=np.arange(len(u)+1)/len(u)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for k in range(3):axes[0].step(x,strength[:,k],where='mid',label=f'Characteristic field {k}')
    axes[0].set(xlabel='Face x',ylabel='Effective gated λ',ylim=(-.02,.3))
    axes[1].hist(np.abs(np.asarray(d['correction'])).ravel(),bins=40,log=True)
    axes[1].set(xlabel='Absolute characteristic reconstruction correction',ylabel='Count (log)')
    axes[0].legend(fontsize=8);fig.suptitle('Saved V2, FV state inputs, Sod N=128 at t=.15; fixed soft λ=.25, no new learned head')
    finish(fig,'mixing_gate_behavior')

if __name__=='__main__':
    spatial();profiles();scalar();training();gate()
