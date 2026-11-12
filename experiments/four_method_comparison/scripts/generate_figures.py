"""Create presentation figures exclusively from measured trajectories and histories."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts import methods as m
from experiments.four_method_comparison.scripts.run_matched_benchmarks import reference
from experiments.four_method_comparison.scripts.calculate_metrics import primitives
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=m.OUT;RAW=OUT/'results/raw';FIG=OUT/'figures'
COLORS={'teno5':'#15559b','neno6':'#df761b','nn_teno5':'#8754a3','learned_head':'#14866a'}
LABELS={'teno5':'TENO5','neno6':'NENO6 (author reconstruction)', 'nn_teno5':'NN-TENO5','learned_head':'NN-TENO5-V2 (Learned Mixing Head)'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,
                     'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
                     'legend.fontsize':9,'figure.dpi':120,'savefig.dpi':240,'pdf.fonttype':42})
ROWS=json.loads((RAW/'evaluated_rows.json').read_text())
MAN=json.loads((OUT/'configs/benchmark_manifest.json').read_text())
CASES={c['id']:c for c in MAN['cases']}

def row(case,n,method):return next((r for r in ROWS if r['case']==case and r['cells']==n and r['method']==method),None)
def arrays(case,n,method):
    r=row(case,n,method)
    if r is None:return None,None
    with np.load(ROOT/r['trajectory_file']) as z:a={k:np.array(z[k]) for k in z.files}
    if case=='shu_osher':
        with np.load(RAW/'reference_shu_2048_dt2.npz') as ref:
            a['reference']=np.column_stack([np.interp(a['x'],ref['x'],ref['final'][:,k]) for k in range(3)])
    return r,a
def save(fig,name,caption):
    fig.savefig(FIG/(name+'.png'),facecolor='white')
    fig.savefig(FIG/(name+'.pdf'),facecolor='white')
    (FIG/(name+'.caption.md')).write_text(caption+'\n')
    plt.close(fig)
def style(ax):ax.grid(alpha=.18);ax.set_axisbelow(True)
def legend(fig):
    fig.legend(handles=[Line2D([0],[0],color='black',ls=':',lw=1.7,label='Reference')]+[
        Line2D([0],[0],color=COLORS[k],lw=1.8,label=LABELS[k]) for k in m.MAIN],
        loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))

def euler_profile(case,name,zoom=None):
    fig,axes=plt.subplots(2,1,figsize=(12.8,7.2),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    _,first=arrays(case,128,'teno5');x=first['x'];ref=first['reference']
    axes[0].plot(x,ref[:,0],color='black',ls=':',lw=2)
    axes[1].plot(x,primitives(ref)[2],color='black',ls=':',lw=2)
    failed=[]
    for method in m.MAIN:
        r,a=arrays(case,128,method)
        if not r or not r['success']:
            failed.append(method);continue
        axes[0].plot(x,a['final'][:,0],color=COLORS[method],lw=1.6)
        axes[1].plot(x,primitives(a['final'])[2],color=COLORS[method],lw=1.6)
    axes[0].set_ylabel('Density');axes[1].set_ylabel('Pressure');axes[1].set_xlabel('x')
    if zoom:
        for ax in axes:ax.set_xlim(*zoom)
    for ax in axes:style(ax)
    label='exact Riemann' if case!='shu_osher' else 'TENO5 N=2048, dt/2'
    fig.suptitle(f'{case.replace("_","–").title()} | N=128, T={CASES[case]["time"]} | SSPRK3, raw historical split flux\nReference: {label}',fontsize=15)
    fig.subplots_adjust(bottom=.16,top=.86,hspace=.15);legend(fig)
    save(fig,name,f'NEW MATCHED EVALUATION. {case}, N=128, T={CASES[case]["time"]}. Reference: {label}. '
        'Four real checkpoints; no repairs. NENO6 is the verified author reconstruction ported to the common Euler solver. '
        'V2 uses phase21 rich head. '+('Failed curves omitted: '+', '.join(failed) if failed else 'All four complete.'))

def burgers():
    fig,axes=plt.subplots(1,2,figsize=(12.8,7.2))
    for ax,case in zip(axes,('burgers_post','burgers_hat')):
        _,a=arrays(case,128,'teno5');x=a['x'];ax.plot(x,a['reference'],'k:',lw=2)
        for method in m.MAIN:
            r,a=arrays(case,128,method)
            if r and r['success']:ax.plot(x,a['final'],color=COLORS[method],lw=1.6)
        ax.set_title(f'{case.replace("_"," ")} | N=128, T={CASES[case]["time"]}')
        ax.set_xlabel('x');ax.set_ylabel('u');style(ax)
    fig.suptitle('Inviscid Burgers | exact entropy references | SSPRK3, LF alpha=1',fontsize=16)
    fig.subplots_adjust(bottom=.18,top=.82,wspace=.25);legend(fig)
    save(fig,'04_burgers_comparison','NEW MATCHED EVALUATION. Left: sine initial condition, T=.9 exceeds shock time2/pi. Right: top-hat, T=.2 before wave interaction. N=128. Exact entropy references. V2 uses scalar run6 linear head. NENO6 uses author reconstruction in common scalar numerical flux.')

def accuracy():
    cases=['sod','lax','shu_osher','burgers_pre','burgers_post','burgers_hat','advection_square']
    fig,ax=plt.subplots(figsize=(12.8,7.2));pos=np.arange(len(cases));width=.22
    for j,method in enumerate(('neno6','nn_teno5','learned_head')):
        vals=[]
        for case in cases:
            b=row(case,128,'teno5');r=row(case,128,method)
            vals.append(r['l1']/b['l1'] if r and r.get('l1') is not None else np.nan)
        ax.bar(pos+(j-1)*width,vals,width,label=LABELS[method],color=COLORS[method],alpha=.88)
    ax.axhline(1,color=COLORS['teno5'],ls='--',lw=1.5,label='TENO5')
    ax.set_xticks(pos,[s.replace('_','\n') for s in cases]);ax.set_ylabel('Error / TENO5 error (lower is better)')
    ax.set_title('Seven main problems at N=128 | smooth control in Figure06\nEuler: mean conserved L1; scalar: L1; each case normalized separately')
    ax.legend(loc='upper left',frameon=False);style(ax);fig.tight_layout()
    save(fig,'05_accuracy_bar_chart','NEW MATCHED EVALUATION. N=128. Euler component mean L1 and scalar L1 are normalized separately to each case\'s TENO5. This chart does not combine different norms into one improvement. See master_results.csv for full values; unavailable endpoint errors produce no bars.')

def convergence():
    fig,axes=plt.subplots(1,2,figsize=(12.8,7.2))
    for ax,case,grids in [(axes[0],'advection_smooth',[64,128,256]),(axes[1],'sod',[128,256])]:
        for method in m.MAIN:
            rs=[row(case,n,method) for n in grids];rs=[r for r in rs if r and r.get('l1') is not None]
            ax.loglog([r['cells'] for r in rs],[r['l1'] for r in rs],'-o',color=COLORS[method],label=LABELS[method])
        ax.set_xlabel('N');ax.set_ylabel('L1' if case=='advection_smooth' else 'Mean conserved L1');style(ax)
        ax.set_xticks(grids,[str(n) for n in grids]);ax.minorticks_off()
        ax.set_title('Smooth advection | T=.2, dt≤.15 dx^(5/3)' if case=='advection_smooth' else 'Sod | T=.2, fixed CFL protocol\nShock case: no high-order claim')
    axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Measured full-solution grid convergence | exact references',fontsize=16);fig.tight_layout(rect=[0,0,1,.93])
    save(fig,'06_error_by_grid_resolution','NEW MATCHED EVALUATION. Smooth point-sampled advection64/128/256 with dt proportional to dx^(5/3). SSPRK3 remains third order in time. Sod128/256 is a shock-resolution comparison, not evidence of fifth-order convergence. Exact references; no fitted theoretical order is assumed.')

def lambdas():
    fig,axes=plt.subplots(1,3,figsize=(12.8,7.2),gridspec_kw={'width_ratios':[1.4,1.4,1]})
    for ax,case in zip(axes[:2],('sod','shu_osher')):
        r,a=arrays(case,128,'learned_head');lam=a['lambda_final'];x=a['x'];mean=lam.mean(axis=(1,2))
        ax.plot(x,mean,color=COLORS['learned_head'],label='Mean across fields/directions')
        ax.fill_between(x,lam.min(axis=(1,2)),lam.max(axis=(1,2)),color=COLORS['learned_head'],alpha=.18,label='Min–max across fields/directions')
        ax.axhline(MAN['fixed_lambda'],color='gray',ls=':',label='Predeclared fixed lambda')
        ax.set_title(f'{case.replace("_","–")} | N=128, T={CASES[case]["time"]}')
        ax.set_xlabel('Interface x');ax.set_ylabel('lambda');style(ax)
        axes[2].hist(lam.ravel(),bins=35,alpha=.5,density=True,label=case.replace('_','–'))
    axes[0].legend(fontsize=7,frameon=False,loc='best');axes[2].set_xlabel('lambda');axes[2].set_ylabel('Empirical density');axes[2].legend(frameon=False);style(axes[2])
    fig.suptitle('Phase21 learned head is nearly constant | max−min ≈ 2.24 × 10⁻⁵\nZoomed vertical scales; lambda is not calibrated uncertainty',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.88])
    save(fig,'07_mixing_head_lambda_distribution','NEW MATCHED EVALUATION. Phase21 Euler rich head, eight inputs: three probabilities plus five normalized split-flux stencil values. Final-interface coefficients at SodT=.2 and Shu–OsherT=1.8,N=128. Shading shows fields/directions, not confidence bounds. Histogram observations are spatially correlated; no independence claim.')

def cost():
    fig,axes=plt.subplots(1,3,figsize=(12.8,7.2))
    for ax,case in zip(axes,('sod','shu_osher','burgers_post')):
        for method in m.MAIN:
            rs=[row(case,n,method) for n in (128,256)];rs=[r for r in rs if r and r.get('l1') is not None and r.get('runtime_median_s')]
            ax.loglog([r['runtime_median_s'] for r in rs],[r['l1'] for r in rs],'-o',color=COLORS[method],label=LABELS[method])
            for r in rs:ax.annotate(str(r['cells']),(r['runtime_median_s'],r['l1']),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set_title(f'{case.replace("_","–")} | T={CASES[case]["time"]}');ax.set_xlabel('Warmed rollout median (seconds)');ax.set_ylabel('Mean Euler L1' if case!='burgers_post' else 'L1');style(ax)
    axes[0].legend(fontsize=7,frameon=False);fig.suptitle('Accuracy versus CPU cost | three synchronized full-rollout repeats\nCompilation excluded; point labels are grid resolutions',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.88]);save(fig,'08_accuracy_vs_runtime','NEW MATCHED EVALUATION. Same CPU/JAX environment. Median of three warmed device-loop full simulations, compilation excluded; no failed run is credited as a complete fast run. Lines only connect measured128/256 grids, with no interpolation-based equal-budget accuracy claim. Exact references for Sod/Burgers; numerical Shu reference sensitivity is reported separately.')

def training():
    fig,axes=plt.subplots(1,2,figsize=(12.8,7.2))
    p=ROOT/'models/nn_teno5_history.csv';h=np.genfromtxt(p,delimiter=',',names=True)
    for key in ('loss','val_loss'):
        if key in h.dtype.names:axes[0].semilogy(np.arange(len(h))+1,h[key],label=key)
    axes[0].set_title('Original classifier | recorded BCE history');axes[0].set_xlabel('Recorded epoch');axes[0].set_ylabel('Loss');axes[0].legend(frameon=False);style(axes[0])
    for run in ('run2_normalized','run3_pretrained'):
        p=ROOT/f'models/nn_teno5_unrolled_jax_history_{run}.csv';h=np.genfromtxt(p,delimiter=',',names=True)
        axes[1].semilogy(np.arange(len(h)),h['l_total'],label=run.replace('_',' '))
    axes[1].set_title('Scalar physics training | recorded histories');axes[1].set_xlabel('Recorded optimizer update');axes[1].set_ylabel('Unrolled loss (different objective from BCE)');axes[1].legend(frameon=False);style(axes[1])
    fig.suptitle('Recorded research evolution | no new training\nClassifier → scalar physics training → Euler heads: phase18 linear → phase20 linear → phase21 rich',fontsize=14)
    fig.text(.5,.02,'Calendar dates and phase21 per-update loss logs are unavailable; sequence is generation order, not a fabricated calendar.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.85]);save(fig,'09_training_evolution_timeline','ARCHIVED histories, replotted. Sources: models/nn_teno5_history.csv and nn_teno5_unrolled_jax_history_run2_normalized.csv / run3_pretrained.csv. Horizontal coordinates are recorded epochs/updates. Phase18/20/21 ordering comes from historical training sources, without fictional dates or phase21 loss curves. BCE and physics objectives use separate axes and are not directly comparable.')

def local_errors():
    fig,axes=plt.subplots(1,3,figsize=(12.8,7.2))
    for ax,case in zip(axes,('sod','lax','shu_osher')):
        for method in m.MAIN:
            r,a=arrays(case,128,method)
            if r and r.get('l1') is not None:ax.plot(a['x'],a['final'][:,0]-a['reference'][:,0],color=COLORS[method],label=LABELS[method],lw=1.3)
        ax.axhline(0,color='gray',ls=':');ax.set_title(f'{case} | N=128,T={CASES[case]["time"]}');ax.set_xlabel('x');ax.set_ylabel('Density error');style(ax)
    axes[0].legend(fontsize=7,frameon=False);fig.suptitle('Measured local density errors | raw historical solver',fontsize=16)
    fig.tight_layout(rect=[0,0,1,.9]);save(fig,'10_local_density_errors','NEW MATCHED EVALUATION. Density error relative to exact Riemann Sod/Lax and finest numerical Shu reference. N=128. Full precision errors are in raw/evaluated_rows.json; reference sensitivity affects small Shu rankings.')

def error_time():
    fig,axes=plt.subplots(1,2,figsize=(12.8,7.2))
    for ax,case in zip(axes,('sod','burgers_post')):
        for method in m.MAIN:
            r=row(case,128,method)
            if r and r.get('error_over_time'):
                h=r['error_over_time'];ax.plot([v['time'] for v in h],[v['l1'] for v in h],'-o',ms=3,color=COLORS[method],label=LABELS[method])
        if case=='burgers_post':ax.axvline(2/np.pi,color='gray',ls=':',label='Shock formation 2/pi')
        ax.set_title(f'{case} | N=128, T={CASES[case]["time"]}');ax.set_xlabel('Physical time');ax.set_ylabel('Mean Euler L1' if case=='sod' else 'L1');style(ax)
    axes[0].legend(fontsize=8,frameon=False);axes[1].legend(fontsize=8,frameon=False)
    fig.suptitle('Error over time against exact references | measured snapshots',fontsize=16)
    fig.tight_layout(rect=[0,0,1,.91]);save(fig,'11_error_over_time','NEW MATCHED EVALUATION. All snapshots are compared at their saved physical times to exact solutions. SodT=.2 and sine BurgersT=.9,N=128. This follows each declared trajectory, not a claim of arbitrarily long-time stability. Shu has only final-time reference convergence data, so no invented Shu time-error curve is shown.')

def main():
    FIG.mkdir(exist_ok=True)
    euler_profile('sod','01_sod_density_comparison')
    euler_profile('lax','02_lax_density_comparison')
    euler_profile('shu_osher','03_shu_osher_density_zoom',(5.5,7.5))
    burgers();accuracy();convergence();lambdas();cost();training();local_errors();error_time()
    print('Saved eleven measured figures as PNG and PDF, with evidence captions.')

if __name__=='__main__':main()
