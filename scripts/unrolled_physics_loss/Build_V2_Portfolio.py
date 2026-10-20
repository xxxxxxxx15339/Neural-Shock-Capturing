#!/usr/bin/env python3
"""Build measured figures and a seven-page PDF carousel from release artifacts.

No solver executes here. Missing studies are labeled as incomplete.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch,Patch

from src.differentiable.model_io import sha256_file

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42,
 'ps.fonttype':42,'figure.dpi':110,'savefig.dpi':320})
COLORS=['#52677b','#008b86','#df8537','#b23e48','#805dad','#2a6fba']
SOURCE=ROOT/'docs/research_release/results'
CAPTIONS={}


def read(path):return json.loads(path.read_text())


def save(fig,out,name,caption):
    fig.tight_layout()
    for ext in ['png','pdf','svg']:
        target=out/f'{name}.{ext}';fig.savefig(target,bbox_inches='tight')
        if ext=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig);CAPTIONS[name]=caption


def diagrams(out):
    fig,ax=plt.subplots(figsize=(11,3.8));ax.axis('off')
    boxes=[(.01,.55,'Conserved cell averages\nScalar u / Euler U'),(.34,.55,'Five-point stencils\nScalar states / Euler split fluxes'),(.69,.55,'Classical TENO5\nCandidate polynomials + rejection'),(.34,.02,'MLP: 5 → 64 → 32 → 16 → 8 → 3\nSeparate scalar / Euler parameters'),(.69,.02,'Hybrid reconstruction\nTENO5 + g λ (NN − TENO5)'),(.01,.02,'Conservative numerical flux\nFlux difference + SSPRK3')]
    for x,y,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),.29,.34,boxstyle='round,pad=.01',facecolor='#eef6f6',edgecolor='#008b86'))
        ax.text(x+.145,y+.17,label,ha='center',va='center',fontsize=9)
    for a,b in [((.30,.72),(.34,.72)),((.63,.72),(.69,.72)),((.84,.55),(.84,.36)),((.48,.55),(.48,.36)),((.63,.19),(.69,.19))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#52677b','lw':1.6})
    ax.plot([.84,.84,.16],[.02,-.08,-.08],color='#52677b',lw=1.6)
    ax.annotate('',xy=(.16,.02),xytext=(.16,-.08),arrowprops={'arrowstyle':'->','color':'#52677b','lw':1.6})
    ax.set(xlim=(0,1),ylim=(-.13,.94))
    ax.set_title('NN-TENO5 reconstruction and differentiable solver',loc='left',fontweight='bold')
    save(fig,out,'pipeline','Conceptual architecture diagram. Neural candidate weights, deterministic gating and correction strength are separate; all methods update a conservative numerical-flux difference.')
    fig,ax=plt.subplots(figsize=(9,3));ax.axis('off')
    for x,y,text in [(.02,.3,'Compute TENO5 χ₀, χ₁, χ₂'),(.37,.3,'All χ ≥ 10⁻⁵?'),(.73,.60,'Yes: g = 0\nUse classical reconstruction'),(.73,.02,'No: g = 1\nApply λ-scaled neural correction')]:
        ax.text(x,y,text,va='center',bbox=dict(boxstyle='round,pad=.7',facecolor='#eef6f6',edgecolor='#008b86'))
    for a,b in [((.29,.3),(.36,.3)),((.58,.3),(.72,.6)),((.58,.3),(.72,.02))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#52677b'))
    ax.set(xlim=(0,1.12),ylim=(-.3,.9),title='Deterministic smoothness gate — definition, not an accuracy proof')
    save(fig,out,'smoothness_gate','Illustrative decision diagram. The detector may reject candidates at smooth critical points; measured convergence remains necessary.')


def strengths(out):
    report=read(SOURCE/'full-validation/summary.json')['summary']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,physics in zip(axes,['scalar','euler']):
        for k,mode in enumerate(['soft','hard']):
            rows=[r for r in report if r['physics']==physics and r['mode']==mode]
            x=[r['strength'] for r in rows];y=[r['macro_relative_l1'] for r in rows]
            ax.plot(x,y,'o-',color=COLORS[k+1],label=f'{mode} candidate policy')
            ci=np.array([r['macro_ci95'] for r in rows]);ax.fill_between(x,ci[:,0],ci[:,1],color=COLORS[k+1],alpha=.14)
            for r in rows:
                if r['failures']:ax.annotate(f'{r["failures"]} failure(s)',(r['strength'],r['macro_relative_l1']),fontsize=8,rotation=25)
        ax.axhline(1,color=COLORS[0],ls=':',label='classical TENO5');ax.set(xlabel='Correction strength λ',ylabel='Macro L1 / TENO5',title=physics.capitalize());ax.legend(fontsize=8)
    save(fig,out,'correction_strength','Full validation: 615 scalar and 307 Euler trajectories; equal weight per nonconstant family×resolution. Shading: paired stratified trajectory bootstrap 95% intervals, conditional on valid cases. Failed settings are ineligible for selection. Constants and baseline errors ≤1e-10 are excluded from ratios.')
    for physics in ['scalar','euler']:
        families=sorted({g['family'] for r in report if r['physics']==physics for g in r['groups'] if g['family']!='constant'})
        fig,axes=plt.subplots(2,3,figsize=(12,7));axes=axes.ravel()
        for ax,family in zip(axes,families):
            for k,n in enumerate([64,128,256]):
                for mode,ls in [('soft','-'),('hard','--')]:
                    rows=[r for r in report if r['physics']==physics and r['mode']==mode]
                    yy=[next(g for g in r['groups'] if g['family']==family and g['cells']==n)['relative_l1'] for r in rows]
                    ax.plot([r['strength'] for r in rows],yy,ls,color=COLORS[k],label=f'N={n}, {mode}')
            ax.axhline(1,color='gray',ls=':');ax.set(title=family,xlabel='λ',ylabel='L1 / TENO5');ax.legend(fontsize=7)
        for ax in axes[len(families):]:ax.axis('off')
        save(fig,out,f'strength_families_{physics}',f'{physics.capitalize()} validation strength sensitivity by family and resolution. Solid: soft; dashed: hard. All per-trajectory failures remain in the accompanying JSON report.')


def convergence(out):
    path=SOURCE/'selected-spatial/convergence.json'
    if not path.exists():path=SOURCE/'convergence/convergence.json'
    if path.exists():
        report=read(path);fig,axes=plt.subplots(2,2,figsize=(11,8))
        for i,physics in enumerate(['scalar','euler']):
            for j,profile in enumerate(['regular','critical-sin4']):
                ax=axes[i,j]
                for k,method in enumerate(['teno5','gated-soft','gated-hard','neural-soft']):
                    rows=next(x['rows'] for x in report['spatial'] if x['physics']==physics and x['profile']==profile and x['method']==method)
                    ax.loglog([r['cells'] for r in rows],[r['l1'] for r in rows],'o-',color=COLORS[k],label=method)
                ax.set(xlabel='Cells N',ylabel='Mean absolute derivative error',title=f'{physics}, {profile}');ax.legend(fontsize=8)
        save(fig,out,'smooth_convergence','Semidiscrete cell-average divergence errors: no SSPRK3 temporal error. Euler uses an advected entropy wave; the general nonlinear Euler diagnostic is separate. Critical-point behavior is reported without a fifth-order guarantee.')
    path=SOURCE/'nonlinear-euler/nonlinear_euler.json'
    if path.exists():
        fig,ax=plt.subplots(figsize=(7,4))
        for k,solver in enumerate(['historical-split-flux','experimental-fv-state-flux']):
            rows=next(r['rows'] for r in read(path)['results'] if r['solver']==solver and r['method']=='gated-soft')
            ax.loglog([r['cells'] for r in rows],[r['l1'] for r in rows],'o-',color=COLORS[k],label=solver)
        ax.set(xlabel='Cells N',ylabel='Mean absolute derivative error',title='General smooth Euler: cell-average consistency');ax.legend()
        save(fig,out,'nonlinear_euler_consistency','Instantaneous nonlinear smooth Euler diagnostic. Historical split-flux treatment approaches second order on cell averages; optional characteristic-state finite-volume flux approaches fifth order. The experimental variant has no completed shock/positivity validation.')


def profile(out,case,n,name):
    folder=SOURCE/f'independent/{case}-{n}'
    if not (folder/'reference.npz').exists():return
    ref=np.load(folder/'reference.npz')['state'];base=np.load(folder/'teno5.npz')['state'];nn=np.load(folder/'v2-selected-soft.npz')
    final=nn['state'];x=(np.arange(n)+.5)/n
    component=lambda v:v[:,0] if v.ndim==2 else v
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for value,label,color in [(ref,'Reference','#20262d'),(base,'TENO5',COLORS[0]),(final,'Gated NN-TENO5, λ=.25',COLORS[1])]:
        axes[0].plot(x,component(value),label=label,color=color,lw=1.3)
    for value,label,color in [(base,'TENO5',COLORS[0]),(final,'Gated NN',COLORS[1])]:axes[1].plot(x,component(value-ref),label=label,color=color)
    axes[0].set(ylabel='Density ρ' if ref.ndim==2 else 'u',title=f'{case}, N={n}');axes[0].legend()
    axes[1].set(ylabel='Pointwise cell-average error')
    gate=nn['neural_gate'][-1].reshape(n,-1).mean(axis=-1);correction=np.abs(nn['neural_correction'][-1]).reshape(n,-1).mean(axis=-1)
    axes[2].plot(x,gate,color=COLORS[2],label='Active fraction across fields / LF signs');axes[2].set(ylabel='Gate fraction',xlabel='x / domain length')
    right=axes[2].twinx();right.plot(x,correction,color=COLORS[1],label='Mean |reconstruction correction|');right.set_ylabel('|Correction|');axes[2].legend(fontsize=8,loc='upper left');right.legend(fontsize=8,loc='upper right')
    info=read(folder/'reference.json');kind=info['kind']
    caveat='Numerical reference; ranking subject to measured Cauchy uncertainty.' if 'numerical' in kind else 'Analytic finite-volume cell-average reference.'
    save(fig,out,name,f'{case} N={n}. {caveat} Gate/correction values come from the selected method trajectory. Euler correction is in characteristic split-flux space and is not a conservative-state correction.')
    if name=='scalar_discontinuity':
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
        g=nn['neural_gate'].reshape(len(nn['times']),n,-1).mean(axis=-1)
        c=np.abs(nn['neural_correction']).reshape(len(nn['times']),n,-1).mean(axis=-1)
        for ax,value,label in [(axes[0],g,'Gate active fraction'),(axes[1],c,'Mean |correction|')]:
            artist=ax.pcolormesh(x,nn['times'],value,shading='nearest',cmap='viridis');fig.colorbar(artist,ax=ax,label=label);ax.set_ylabel('Physical time')
        axes[1].set_xlabel('x / domain length')
        save(fig,out,'gate_evolution','Gate and reconstruction-correction evolution along the selected scalar numerical trajectory, sampled at 22 times. This is measured numerical output, not an illustration.')


def final_summary(out):
    path=SOURCE/'independent/benchmark.json'
    if not path.exists():return
    results=read(path)['results']
    wall=SOURCE/'wall-closure/benchmark.json'
    if wall.exists():results+=read(wall)['results']
    rows=[r for r in results if r['method']=='v2-selected-soft']
    fig,ax=plt.subplots(figsize=(11,6));labels=[f'{r["case"]} N={r["cells"]}' for r in rows]
    for k,r in enumerate(rows):
        if r.get('relative_l1') is not None:
            color=COLORS[1] if r.get('ranking_resolved') and r['case']!='independent-woodward-colella' else '#b3bec8'
            ax.barh(k,r['relative_l1'],color=color)
        else:
            ax.text(.03,k,'FAILED — no accuracy score' if not r['diagnostics']['valid_for_accuracy_claim'] else 'No accuracy reference',va='center',color=COLORS[3],fontsize=8)
    ax.set_yticks(np.arange(len(rows)),labels,fontsize=8);ax.axvline(1,color='#333',ls=':');ax.set(xlabel='Selected gated-soft L1 / TENO5',title='Prospective long-time benchmark — including failed stress tests')
    ax.set_ylim(len(rows)-.3,-.7)
    ax.legend(handles=[Patch(facecolor=COLORS[1],label='Analytic or Cauchy-resolved ranking'),
                       Patch(facecolor='#b3bec8',label='Reference uncertainty unresolved')],loc='lower right',fontsize=7)
    save(fig,out,'final_benchmark','Prospective cases declared after validation selection and before evaluation. Gray indicates unresolved reference ranking, invalid stages or the mirror-only wall defect. The wall-closed rerun is a distinct physical-boundary correction with no strength tuning. Numerical-reference Cauchy tests qualify interpretation. These cases do not support universal superiority.')


def performance(out):
    path=SOURCE/'performance/performance.json'
    if not path.exists():return
    rows=read(path)['results'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,physics in zip(axes,['scalar','euler']):
        for k,method in enumerate(['teno5','gated-soft','gated-hard','neural-soft']):
            data=[r for r in rows if r['physics']==physics and r['method']==method]
            ax.loglog([r['warmed_median_seconds'] for r in data],[r['l1'] for r in data],'o-',label=method,color=COLORS[k])
            for r in data:ax.annotate(str(r['cells']),(r['warmed_median_seconds'],r['l1']),fontsize=7)
        ax.set(xlabel='Warmed trajectory execution [s]',ylabel='Mean absolute state error',title=physics);ax.legend(fontsize=8)
    save(fig,out,'accuracy_runtime','Synchronized CPU float64 execution, median of nine repeats after warm-up; compilation reported separately. Labels show grid cells. One fixed IVP per physics; no GPU or general speedup claim.')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,physics in zip(axes,['scalar','euler']):
        for k,method in enumerate(['teno5','gated-soft','gated-hard','neural-soft']):
            data=[r for r in rows if r['physics']==physics and r['method']==method]
            ax.loglog([r['cells'] for r in data],[r['l1'] for r in data],'o-',label=method,color=COLORS[k])
        ax.set(xlabel='Cells N',ylabel='Mean absolute state error',title=physics);ax.legend(fontsize=8)
    save(fig,out,'accuracy_resolution','Measured accuracy against grid resolution on the same timed IVPs. Compare with accuracy_runtime; a same-grid improvement alone is not a speedup.')


def carousel(out):
    pages=[('01','Capturing shocks accurately','A research internship in numerical PDEs and scientific machine learning.\nHigh resolution matters; oscillations and instability matter too.','scalar_discontinuity'),
           ('02','Classical structure, learned decisions','TENO5 combines candidate reconstructions. A small neural network\nchanges their selection; the numerical flux still drives the PDE update.','pipeline'),
           ('03','A deterministic smoothness gate','5 → 64 → 32 → 16 → 8 → 3. Separate scalar and Euler checkpoints.\nNeural corrections activate only where TENO5 rejects a candidate.','smoothness_gate'),
           ('04','Train against mathematical solutions','6,144 independent analytic PDE trajectories; N = 64, 128, 256.\nFinite-volume cell-average targets. No experimental sensor data.','strength_families_scalar'),
           ('05','Validation results are mixed','Full validation favors soft λ = .25: scalar macro ratio .9770; Euler .9479.\nLarge strengths can fail. Independent results are shown with caveats.','correction_strength'),
           ('06','Accuracy needs numerical evidence','Critical points and general nonlinear Euler expose accuracy limits.\nTiming includes CPU inference cost; no universal speedup is claimed.','nonlinear_euler_consistency'),
           ('07','A reproducible investigation','Yasser BAOUZIL · supervision: Radouan BOUKHARFANE\nFoundations: Stevens & Colonius; WENO/TENO literature.\nCode, measured results, negative findings and reproduction commands.','final_benchmark')]
    with PdfPages(out/'NN_TENO5_RESEARCH_CAROUSEL.pdf') as pdf:
        for number,title,body,graphic in pages:
            fig=plt.figure(figsize=(8,10),facecolor='#102331')
            fig.text(.07,.94,f'NN-TENO5  /  RESEARCH NOTES                                    {number}',color='#67d5cb',fontsize=10,weight='bold')
            fig.text(.07,.86,title,color='white',fontsize=23,weight='bold',wrap=True)
            fig.text(.07,.76,body,color='#d0dfe5',fontsize=11,linespacing=1.7,va='top')
            image_path=out/'figures'/f'{graphic}.png'
            if image_path.exists():
                ax=fig.add_axes([.05,.14,.90,.48]);ax.imshow(plt.imread(image_path));ax.axis('off')
            else:
                fig.text(.08,.43,'Study incomplete — no numerical plot substituted.',color='white',fontsize=13)
            fig.text(.07,.07,'github.com/xxxxxxxx15339/Neural-Shock-Capturing',color='#67d5cb',fontsize=10)
            fig.text(.07,.035,'Validation-selected results are not an independent-test guarantee.',color='#c1d0d8',fontsize=8)
            pdf.savefig(fig,facecolor=fig.get_facecolor());plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'docs/portfolio');args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);figdir=out/'figures'
    if figdir.exists() and any(figdir.iterdir()):raise FileExistsError('Use a new portfolio output directory')
    figdir.mkdir(exist_ok=True)
    diagrams(figdir);strengths(figdir);convergence(figdir)
    profile(figdir,'independent-advection-square',192,'scalar_discontinuity')
    profile(figdir,'independent-sod',192,'euler_sod')
    profile(figdir,'independent-shu-osher',256,'shu_osher')
    final_summary(figdir);performance(figdir);carousel(out)
    (out/'FIGURE_CAPTIONS.md').write_text('# Reproducible figure captions\n\n'+'\n\n'.join(f'**{name}** — {caption}' for name,caption in CAPTIONS.items())+'\n')
    manifest=dict(generator='scripts/unrolled_physics_loss/Build_V2_Portfolio.py',generator_sha256=sha256_file(__file__),sources={str(p.relative_to(ROOT)):sha256_file(p) for p in SOURCE.rglob('*.json')},files={str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*') if p.is_file() and p.name!='figures_manifest.json'})
    (out/'figures_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
