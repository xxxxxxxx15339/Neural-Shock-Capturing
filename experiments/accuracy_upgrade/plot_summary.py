"""Final confirmation, long-rollout and isolated-cost figures; artifacts only."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.accuracy_upgrade.plot_results import OUT,finish,LABELS,COLORS
from experiments.accuracy_upgrade.study import VARIANTS


def run():
    perf=json.loads((OUT/'results/performance.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for v in VARIANTS:
        rows=[r for r in perf['rows'] if r['variant']==v]
        axes[0].loglog([r['warm_median_s'] for r in rows],[r['l1'] for r in rows],'o-',label=LABELS[v],color=COLORS[v])
        axes[1].plot([r['cells'] for r in rows],[r['warm_per_step_s'] for r in rows],'o-',label=LABELS[v],color=COLORS[v])
        if v in ('historical_teno','fv_v2_soft25'):
            for r in rows:
                axes[0].annotate(str(r['cells']),(r['warm_median_s'],r['l1']),
                                 xytext=(5,3),textcoords='offset points',fontsize=7)
    axes[0].set(xlabel='Warmed simulation time [s]',ylabel='Conserved-component L1 error')
    axes[1].set(xlabel='N',ylabel='Warmed time per SSPRK3 step [s]');axes[0].legend(fontsize=7)
    fig.suptitle('Sod t=.15, transmissive, CPU float64; 9 synchronized production repeats; matched dt/grid')
    finish(fig,'accuracy_vs_computational_cost')
    data=json.loads((OUT/'results/long_rollout/summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,n in zip(axes,[64,128]):
        for r in data:
            if r['cells']!=n or not r['success']:continue
            name=r['variant'];h=r['error_over_time'];label=LABELS.get(name,name)
            ax.plot([x['time'] for x in h],[x['l1'] for x in h],label=label,lw=1)
        ax.set(xlabel='Physical time',ylabel='L1 error to exact cell averages',title=f'N={n}')
    axes[0].legend(fontsize=6);fig.suptitle('Frozen paired pilots: Sod t=.2, SSPRK3, transmissive; no checkpoint selection')
    finish(fig,'pilot_long_rollout_errors')
    data=json.loads((OUT/'results/final_holdout/statistics.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
    select=['historical_v2_soft25','fv_teno','fv_v2_soft25','safe_fv_v2_soft25']
    for ax,n in zip(axes,[64,128]):
        rows=[r for r in data if r['baseline']=='historical_teno' and r['cells']==n and r['variant'] in select]
        for i,r in enumerate(rows):
            lo,hi=r['ci95'];ax.errorbar(i,r['mean_ratio'],yerr=[[r['mean_ratio']-lo],[hi-r['mean_ratio']]],fmt='o',capsize=4,color=COLORS[r['variant']])
        ax.axhline(1,color='k',ls=':',lw=1);ax.set_xticks(range(len(rows)),[LABELS[r['variant']] for r in rows],rotation=20,ha='right',fontsize=8)
        ax.set(ylabel='Mean paired L1 ratio to historical TENO5',title=f'N={n}')
    fig.suptitle('Predeclared holdout: six independent moderate Riemann IVPs, t=.08; paired 95% bootstrap intervals')
    finish(fig,'frozen_holdout_comparison')
if __name__=='__main__':run()
