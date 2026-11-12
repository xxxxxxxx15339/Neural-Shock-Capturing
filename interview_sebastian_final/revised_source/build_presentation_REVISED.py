#!/usr/bin/env python3
"""Build the KIT interview in the supplied V4 visual language.

Run from this revised_source directory with numpy, matplotlib, Pillow, lxml:
  python write_notes_REVISED.py
  python build_presentation_REVISED.py --prepare
  tectonic --keep-logs --outdir assets/math assets/math/equations.tex
  python build_presentation_REVISED.py --build
Compile ../Sebastian_Research_Interview_REVISED.tex from its parent directory.
Then run finalize_REVISED.py here. Originals are never an output target.

The finalization script preserves the original V4 PDF landing page exactly.
The PPTX retains its original editable landing objects. Other text/diagrams are
native DrawingML; equations have vector SVG + PNG fallback and editable TeX.
No solver is run and no experimental measurement is changed.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, sys, shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import subprocess
import numpy as np
from PIL import Image
from lxml import etree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
NAME='Sebastian_Research_Interview_REVISED'
ASSETS=HERE/'assets'; MATH=ASSETS/'math'
sys.path.insert(0,str(ASSETS))
import ooxml_support as base
base.INK=base.ORANGE=base.TEAL='242424'
base.MUTED='595959';base.LIGHT='F5F5F5';base.BORDER='DEDEDE'
base.FONT='Arial';base.SLIDE_COUNT=9
INK,MUTED,LIGHT,LINE='242424','595959','F5F5F5','DEDEDE'
GOOD,BAD,TEAL='15803D','B91C1C','157F89'
NS=base.NS; REL=base.REL; CT=base.CT
jobs={}; arts={}; registry={}; preparing=False

def key(expr,size,color):return json.dumps([expr,size,color])
def savefig(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(ASSETS/f'{name}.{ext}',dpi=260,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def charts():
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':13,
        'axes.labelcolor':'#595959','text.color':'#242424','axes.edgecolor':'#888888',
        'xtick.color':'#595959','ytick.color':'#595959','axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'path','mathtext.fontset':'cm'})
    rows=list(csv.DictReader((ASSETS/'euler_instability_trace.csv').open()))
    fig,ax=plt.subplots(figsize=(7.1,3.85))
    styles=['-','--','-.',':','-'];colors=['#242424','#157F89','#A36327','#777777','#B91C1C']
    for mult,ls,c in zip([1.,1.5,2.,3.,5.],styles,colors):
        r=[v for v in rows if float(v['alpha_margin'])==mult];good=[v for v in r if v['finite']=='True']
        x=[int(v['step']) for v in good];y=[float(v['max_density']) for v in good]
        ax.plot(x,y,label=f'{mult:g}×',c=c,ls=ls,lw=2.05,marker='o',ms=3.5)
        if any(v['finite']!='True' for v in r):ax.scatter([x[-1]+1],[y[-1]],marker='x',s=65,c=c,lw=2)
    ax.set_yscale('log');ax.set_xlabel('SSPRK3 step (zero-based)');ax.set_ylabel('Maximum density')
    ax.grid(axis='y',c='#E3E3E3',lw=.7);ax.legend(frameon=False,ncol=5,loc='lower center',bbox_to_anchor=(.5,1.015),fontsize=12)
    ax.set_ylim(1,1e46);ax.set_yticks([1e0,1e10,1e20,1e30,1e40]);fig.tight_layout()
    savefig(fig,'instability_controlled_caption')
    d=json.loads((ASSETS/'nonlinear_euler.json').read_text())
    fig,ax=plt.subplots(figsize=(6.65,4.0))
    for solver,label,c,marker in [('historical-split-flux','Historical split flux','#595959','o'),
                                  ('experimental-fv-state-flux','Experimental state + Rusanov','#157F89','s')]:
        r=next(r for r in d['results'] if r['solver']==solver and r['method']=='teno5')['rows']
        ax.loglog([v['cells'] for v in r],[v['l1'] for v in r],c=c,marker=marker,lw=2.1,ms=7,label=label)
    ax.set_xlabel('Cells N');ax.set_ylabel('Spatial mean-state $L^1$ error')
    ax.grid(which='major',c='#E3E3E3',lw=.7);ax.legend(frameon=False,fontsize=12,loc='upper right')
    ax.set_xticks([40,80,160,320,640],labels=['40','80','160','320','640'])
    fig.tight_layout();savefig(fig,'nonlinear_consistency_readable')
    data=json.loads((ASSETS/'independent_benchmark.json').read_text())['results']
    cases=[('independent-sod',192,'Sod'),('independent-contact',192,'Contact'),
           ('independent-riemann-0',192,'Riemann 0'),('independent-riemann-1',192,'Riemann 1'),
           ('independent-riemann-2',192,'Riemann 2'),('independent-riemann-3',192,'Riemann 3')]
    fig,ax=plt.subplots(figsize=(6.8,3.1))
    labels=[]
    for y,(case,n,label) in enumerate(cases):
        a=next(r for r in data if r['case']==case and r['cells']==n and r['method']=='teno5')
        b=next(r for r in data if r['case']==case and r['cells']==n and r['method']=='v2-selected-soft')
        ratio=b['primary_error']/a['primary_error'];c='#999999' if '*' in label else '#15803D' if ratio<1 else '#B91C1C'
        ax.plot([1,ratio],[y,y],c=c,lw=3);ax.scatter([ratio],[y],c=c,s=58,zorder=3)
        ax.text(1.122,y,f'{ratio:.4f}',va='center',ha='right',fontsize=14,color=c)
        labels.append(f'{label} · N={n}')
    # An unresolved reference ranking is not plotted as an accuracy score.
    labels.append('Shu–Osher · N=256')
    ax.text(.9,6,'REFERENCE UNRESOLVED',color='#777777',va='center',fontsize=12.4)
    ax.axvline(1,c='#242424',ls='--',lw=1);ax.set_yticks(range(7),labels=labels,fontsize=14.5)
    ax.set_ylim(6.55,-.55);ax.set_xlim(.875,1.13);ax.set_xticks([.9,.95,1.,1.05,1.1]);ax.set_xlabel(r'$r_{L^1}=E^{NN}_{L^1}/E^{TENO5}_{L^1}$',fontsize=15)
    ax.tick_params(axis='x',labelsize=13)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=8);ax.grid(axis='x',c='#EEEEEE',lw=.6)
    fig.tight_layout();savefig(fig,'euler_benchmark_excerpt')
    nn=np.load(ASSETS/'sod192_selected.npz')['state'];cl=np.load(ASSETS/'sod192_teno5.npz')['state']
    f=np.load(ASSETS/'sod192_reference.npz');ref=f['state'] if 'state' in f else f['reference']
    x=(np.arange(192)+.5)/192
    fig,ax=plt.subplots(figsize=(4.8,1.65))
    for a,label,c,ls in [(ref,'Exact avg.','#242424','-'),(cl,'TENO5','#84909A','--'),(nn,'V2 soft/.25','#157F89','-')]:
        ax.plot(x,a[:,0],label=label,c=c,lw=2.0,ls=ls)
    # Exact same N192 arrays; explicitly labeled zoom exposes the recorded undershoot.
    ax.set_xlim(.776,.805);ax.set_ylim(.12445,.12615)
    ax.set_xticks([.78,.79,.80]);ax.set_yticks([.1245,.1250,.1260])
    ax.set_ylabel(r'$\rho$',rotation=0,labelpad=8);ax.tick_params(labelsize=11.5)
    ax.legend(frameon=False,ncol=3,fontsize=10.2,loc='lower center',bbox_to_anchor=(.5,1.00))
    ax.set_xlabel('x · zoom just right of the Sod shock',labelpad=0,fontsize=11.3)
    fig.tight_layout(pad=.4);savefig(fig,'sod192_density_inset')

class Slide(base.Slide):
    def __init__(self,deck,n,title,section):
        registry[n]=[];self._in_math=False
        if f'ppt/slides/_rels/slide{n}.xml.rels' not in deck.parts:
            root=ET.Element(f'{{{REL}}}Relationships')
            ET.SubElement(root,f'{{{REL}}}Relationship',Id='rId1',Type=NS['r']+'/slideLayout',Target='../slideLayouts/slideLayout1.xml')
            deck.parts[f'ppt/slides/_rels/slide{n}.xml.rels']=base.xml(root)
        super().__init__(deck,n,title,section)
        for node in self.root.xpath("//a:rPr[@sz='2900']",namespaces=NS):node.set('sz','2700')
        for ob in registry[n]:
            if ob.get('type')=='text' and ob['size']==29:ob['size']=27
        if n>9:
            value=f'BACKUP {chr(65+n-10)} / 03'
            for node in self.root.xpath('//a:t',namespaces=NS):
                if node.text==f'{n:02d} / 9':node.text=value
            for ob in registry[n]:
                if ob.get('text')==f'{n:02d} / 9':ob['text']=value;ob['x']=10.65;ob['w']=2.05
            for sp in self.tree.findall('p:sp',NS):
                if sp.xpath('.//a:t[text()="'+value+'"]',namespaces=NS):
                    base.transform(sp.find('p:spPr',NS),10.65,7.12,2.05,.26)
            self.root.set('show','0')

    def shape(self,x,y,w,h,fill_color=None,stroke=None,radius=False,name='Panel'):
        ob=super().shape(x,y,w,h,fill_color,stroke,radius,name)
        if fill_color or stroke:registry[self.number].append(dict(type='shape',x=x,y=y,w=w,h=h,fill=fill_color,stroke=stroke,radius=radius))
        return ob
    def panel(self,x,y,w,h,color=LIGHT):
        return self.shape(x,y,w,h,fill_color=color,radius=True)
    def arrow(self,x,y,w=.36,color=MUTED):
        return self.txt('→',x,y,w,.38,min(23,w*72*.95),color,align='ctr')
    def txt(self,value,x,y,w,h,size=18,color=INK,bold=False,align='l',valign='t',margins=0,line_spacing=1.08,name='Text'):
        ob=super().txt(value,x,y,w,h,size,color,bold,align,valign,margins,line_spacing,name)
        registry[self.number].append(dict(type='text',text=value,x=x,y=y,w=w,h=h,size=size,color=color,bold=bold,align=align,spacing=line_spacing))
        return ob
    def picture(self,blob,x,y,w,h,name='Picture',svg=None,center=True):
        im=Image.open(BytesIO(blob));aspect=im.width/im.height
        aw,ah=min(w,h*aspect),min(h,w/aspect);px=x+(w-aw)/2 if center else x;py=y+(h-ah)/2
        if not self._in_math:
            filename='art_'+hashlib.sha256(blob).hexdigest()[:14]+'.png'
            (ASSETS/filename).write_bytes(blob)
            registry[self.number].append(dict(type='image',path='revised_source/assets/'+filename,x=px,y=py,w=aw,h=ah,name=name))
        return super().picture(blob,x,y,w,h,name,svg,center)
    def equation(self,expr,x,y,w,h,size=24,color=INK,align='l'):
        k=key(expr,size,color);jobs[k]=dict(latex=expr,fontsize=size,color=color)
        if preparing:return
        png,svg,nw,nh=arts[k];scale=min(1.,w/nw,h/nh);ew,eh=nw*scale,nh*scale
        px=x+(w-ew)/2 if align=='ctr' else x;py=y+(h-eh)/2
        self._in_math=True
        super().picture(png,px,py,ew,eh,name='Equation: '+expr,svg=svg,center=False)
        self._in_math=False
        registry[self.number].append(dict(type='equation',latex=expr,x=px,y=py,w=ew,h=eh,size=size*scale,color=color))
        self.deck.equations.append(dict(slide=self.number,latex=expr,effective_point_size=round(size*scale,2)))
    def citation(self,text):self.txt(text,.66,6.80,11.95,.18,8.5,MUTED)
    def band(self,text,y=6.32,color=INK,size=16):
        self.shape(.6,y,.042,.35,fill_color=color)
        self.txt(text,.77,y-.01,11.9,.47,size,color)

UPDATE=r'\bar U_i^{n+1}=\bar U_i^n-\frac{\Delta t}{\Delta x}\left(\widehat F_{i+1/2}-\widehat F_{i-1/2}\right)'

def deck_content(deck):
    s=Slide(deck,2,'Learning inside a classical shock-capturing solver','Research background · scientific machine learning')
    s.picture(deck.parts['ppt/media/image2.png'],.6,1.77,4.95,3.65,name='Original V4 schlieren image; visual motivation, not a project experiment')
    s.txt('My technical work · scientific-ML internship',5.94,1.80,6.7,.35,20,INK,True)
    s.panel(5.85,2.39,6.85,1.04)
    s.equation(r'\partial_t U+\partial_xF(U)=0',6.12,2.57,6.25,.67,size=32)
    s.txt('Modular ENO baseline + WENO/TENO 3, 5, 7\nConvergence checks; Sod / Shu–Osher benchmarks\nCharacteristic Euler + neural stencil decisions\nDifferentiable JAX rollouts; classical/NN blending\nControlled investigation of numerical failures',5.96,3.66,6.55,1.66,17.3,MUTED,line_spacing=1.32)
    labels=[('Original WENO-NN','Coefficient corrections'),('NN-TENO / oracle','Candidate selection'),('JAX / unrolled','Rollout loss + blending'),('Later V2','Classical gate + NN')]
    for j,(a,b) in enumerate(labels):
        x=.6+3.08*j;s.panel(x,5.64,2.87,.72);s.txt(a,x+.16,5.76,2.55,.23,16,INK,True);s.txt(b,x+.16,6.06,2.55,.21,12.5,MUTED)
    s.band('Objective: improve solution accuracy while retaining numerical reliability.',y=6.49,size=17)
    s.citation('Project scope: src/schemes/ · src/core/ · scripts/legacy_imitation_training/ · scripts/unrolled_physics_loss/')
    s.finish('V4 visual language; distinguish four research generations.')

    s=Slide(deck,3,'Where the network enters the Euler solver','Numerical framework · actual Euler pipeline')
    s.equation(r'U=(\rho,m,E)^{\mathsf T},\quad F(U)=\begin{pmatrix}m\\m^2/\rho+p\\m(E+p)/\rho\end{pmatrix}',.75,1.75,5.60,1.52,size=25)
    s.txt('Conservative forward-Euler building block',6.65,1.91,5.90,.31,17,INK,True)
    s.equation(UPDATE,6.62,2.39,6.02,.79,size=24)
    boxes=['Conserved\nstate U','Roe basis\nL and R','LF split\nfluxes','TENO / NN\nreconstruction','Shared-face\nflux','SSPRK3\nstages']
    for j,t in enumerate(boxes):
        x=.6+j*2.055
        if j==3:
            s.shape(x,3.57,1.82,.94,fill_color=LIGHT,stroke=INK,radius=True)
            s.txt(t,x+.075,3.69,1.67,.55,16,INK,True,align='ctr')
            s.txt('LEARNED DECISION',x+.075,4.31,1.67,.16,9.8,MUTED,True,align='ctr')
        else:
            s.panel(x,3.57,1.82,.94);s.txt(t,x+.075,3.80,1.67,.55,16,INK,align='ctr')
        if j<5:s.arrow(x+1.82,3.85,.235)
    s.txt('Default path: reconstruct characteristic split fluxes, not physical left/right interface states.',.68,4.75,12.,.44,17,MUTED)
    s.panel(.60,5.35,8.30,.91)
    s.equation(r'\mathcal R_{V2}=\mathcal R_T+G\lambda\left(\mathcal R_{NN}-\mathcal R_T\right)',.88,5.51,7.74,.55,size=29)
    s.txt('Later V2\nG: classical TENO trouble gate\nλ: controlled correction strength',9.18,5.38,3.42,1.02,15.5,MUTED,line_spacing=1.25)
    s.citation('Implemented: euler_jax.py:166–254 · gated_teno5_jax.py:19–67. SSPRK3 reevaluates reconstruction at every stage.')
    s.finish('State/flux equations, exact pipeline and later V2 policy.')

    s=Slide(deck,4,'Conservation does not ensure positive Euler states','Observed failure · density and internal energy')
    s.equation(r'p=(\gamma-1)\left(E-\frac{m^2}{2\rho}\right)',.77,1.64,7.65,.90,size=29)
    s.equation(r'\rho>0,\qquad p>0',9.0,1.78,3.58,.64,size=29)
    s.equation(r'\boxed{\delta p\approx(\gamma-1)\left(\delta E-v\,\delta m+\tfrac12v^2\delta\rho\right)}',.79,2.58,11.85,.65,size=25)
    s.txt('v = m/ρ · linearization about an admissible reference state',.89,3.29,11.6,.22,13.5,MUTED)
    cases=[('Oracle NN-TENO5 · Sod, N=120','Non-positive density · step 6',BAD),
           ('V2 soft/.75 · Riemann, N=128','Negative pressure · step 13, stage 1',BAD),
           ('V2 soft/.25 · same short IVP','15 steps · all raw stages admissible',GOOD)]
    for j,(a,b,c) in enumerate(cases):
        y=3.67+j*.50;s.panel(.6,y,12.12,.43);s.txt(a,.80,y+.10,5.6,.25,15.7,INK,True);s.txt(b,6.56,y+.10,5.95,.25,15.7,c,True)
    s.txt('Neural flux → Euler update → possible negative internal energy → invalid pressure',.78,5.30,11.85,.33,16,INK,True)
    s.txt('Smaller corrections / repair remain empirical.\nWeights and SSPRK3 alone do not ensure positivity.\nInvalid p can break sound speeds and the Roe basis.',.78,5.82,6.74,.79,14.1,MUTED,line_spacing=1.20)
    s.panel(7.79,5.77,4.93,.86)
    s.txt('Established · not implemented',7.99,5.94,4.51,.26,15.7,INK,True)
    s.txt('Conservative flux limiter + admissible\nlow-order baseline + stage CFL.',7.99,6.25,4.51,.34,13.5,MUTED,line_spacing=1.08)
    s.citation('Observed: audit §§3.1/3.3. Theory: Hu–Adams–Shu (2013); Zhang–Shu (2012); Gottlieb–Shu–Tadmor (2001).')
    s.finish('Observed failure vs empirical mitigation vs established unimplemented remedy.')

    s=Slide(deck,5,'The dissipation sweep did not isolate the cause','Historical JAX experiment · a confounded control')
    s.txt('LF multiplier increased · timestep fixed',.70,1.72,7.0,.34,19,INK,True)
    s.picture((ASSETS/'instability_controlled_caption.png').read_bytes(),.59,2.23,7.40,3.77,name='Measured historical trace; alpha varied at fixed dt',svg=(ASSETS/'instability_controlled_caption.svg').read_bytes())
    s.equation(r'F^\pm(U)=\frac12\left(F(U)\pm\alpha U\right)',8.12,1.88,4.55,.76,size=25)
    s.equation(r'\mathrm{CFL}_{LF}\sim\frac{\alpha\Delta t}{\Delta x}',8.17,2.81,4.43,.69,size=26)
    for y,a,b in [(3.73,'Fixed Δt confounds the stability scale','Neural inputs changed too. Earlier divergence\ndoes not show that dissipation itself is harmful.'),
                  (4.62,'Roe conditioning remains a hypothesis','No trace establishes it as the initiating cause.'),
                  (5.51,'Next: matched CFL + classical TENO5','First-invalid RK stages and cond(R) tracing.')]:
        s.txt(a,8.15,y,4.50,.27,15.4,INK,True);s.txt(b,8.15,y+.34,4.50,.45,14.1,MUTED)
    s.txt('×: next non-finite step. 1.5× ends finite but enormous.',.83,6.00,7.1,.23,12.7,MUTED)
    s.shape(.6,6.26,.042,.43,fill_color=INK)
    s.txt('The experiment was confounded: dissipation, effective timestep restriction,',.77,6.25,11.9,.24,15.0,INK)
    s.txt('and neural inputs changed simultaneously.',.77,6.49,11.9,.24,15.0,INK)
    s.citation('Data: models/euler_instability_trace.csv. Test: EulerInstability_TraceCapture.py:66–114. No unique cause established.')
    s.finish('Replot exact stored values; remove causal title; expose fixed-dt confound.')

    s=Slide(deck,6,'Direct state repair breaks the conservative budget','Attempted fix · computational survival versus physical balance')
    s.equation(UPDATE,.72,1.78,7.46,.77,size=25)
    s.equation(r'\Delta M_{\rm repair}=\Delta x\sum_i(\rho_i^{after}-\rho_i^{before})',8.42,1.79,4.19,.77,size=21)
    for x,title in [(.6,'Implemented · post-update repair'),(6.79,'Established · flux limiting (not tested)')]:
        s.panel(x,2.91,5.93,2.28);s.txt(title,x+.22,3.10,5.49,.35,17.7,INK,True)
    def flow(x,labels):
        for j,t in enumerate(labels):
            xx=x+j*1.84;s.shape(xx,3.76,1.51,.56,fill_color='FFFFFF',stroke=LINE);s.txt(t,xx+.06,3.91,1.39,.32,13.8,INK,align='ctr')
            if j<2:s.arrow(xx+1.52,3.88,.30)
    flow(.85,['Flux update','Invalid stage','Overwrite cell'])
    flow(7.04,['NN flux','Limit face flux','Update cells'])
    s.txt('One repaired stage · final RK-weighted source',.90,4.50,5.24,.26,14.6,INK,True)
    s.txt('Δmass ≈ +0.001611 · Δenergy ≈ +0.000889',.90,4.86,5.24,.26,15.5,BAD)
    s.txt('Shared face flux preserves the discrete budget.\nNeeds an admissible baseline, factors and CFL.',7.05,4.53,5.33,.57,15.5,MUTED,line_spacing=1.18)
    s.txt('Post-repair penalties once hid violations.\nGuarded training ≠ unguarded validation.',.76,5.55,6.01,.75,17,MUTED,line_spacing=1.30)
    s.txt('Not implemented or validated\nin this neural solver.',7.03,5.55,5.42,.74,17,MUTED,line_spacing=1.30)
    s.band('A finite repaired trajectory does not certify the unguarded learned solver.',6.39,size=17)
    s.citation('Measured: results/repair/repair.json, soft/.75, one repaired stage row. Conservative remedy: Hu–Adams–Shu (2013).')
    s.finish('Compare an implemented overwrite and a proposed shared-face limiter without conflating them.')

    s=Slide(deck,7,'Validation gains did not imply universal transfer','Later V2 · frozen model and selected soft strength .25')
    s.picture((ASSETS/'euler_benchmark_excerpt.png').read_bytes(),.55,1.74,7.07,3.45,name='Euler ratios; unresolved Shu ranking omitted; blast failures in native callout',svg=(ASSETS/'euler_benchmark_excerpt.svg').read_bytes())
    s.txt('0.94794',7.93,1.83,4.63,.55,34,GOOD,True)
    s.txt('Validation macro L¹ ratio\n307 short Euler IVPs · no invalid stages',7.96,2.53,4.67,.76,16.8,MUTED,line_spacing=1.28)
    s.picture((ASSETS/'sod192_density_inset.png').read_bytes(),7.83,3.36,4.89,1.67,name='Explicit Sod shock-tail zoom; unchanged N192 density arrays',svg=(ASSETS/'sod192_density_inset.svg').read_bytes())
    s.txt('Sod N=192: 4.3% lower L¹, but TV rises\nand density undershoot ≈ 3.51×10⁻⁴.',7.98,5.06,4.61,.61,15.7,MUTED,line_spacing=1.20)
    s.panel(.65,5.28,6.96,1.10)
    s.txt('Woodward–Colella blast · N=128, 256',.86,5.42,6.56,.26,17.3,INK,True)
    s.txt('FAILED — no accuracy score',.86,5.78,6.56,.32,21,BAD,True)
    s.txt('V2 non-finite; classical TENO5 completed both.',.86,6.12,6.56,.20,13.8,MUTED)
    s.txt('Possible contributors · not isolated causes',7.98,5.75,4.69,.22,13.9,INK,True)
    s.txt('Shock strength, interactions, reflective walls.\nState feedback; training / deployment mismatch.',7.98,6.03,4.69,.43,13.6,MUTED,line_spacing=1.15)
    s.shape(.6,6.49,.042,.24,fill_color=INK)
    s.txt('Short-horizon validation did not guarantee stability under stronger out-of-distribution Euler dynamics.',.77,6.48,11.9,.30,15.6,INK)
    s.citation('Frozen V2 only. Data: full-validation/summary.json · independent/benchmark.json · wall-closure/benchmark.json.')
    s.finish('Show favorable, unfavorable, unresolved and failed Euler outcomes; selected validation is not a guarantee.')

    s=Slide(deck,8,'The classical FV pipeline also limited accuracy','Consistency investigation · general nonlinear smooth Euler data')
    s.picture((ASSETS/'nonlinear_consistency_readable.png').read_bytes(),.60,1.83,7.08,4.38,name='Exact cell-average nonlinear Euler consistency diagnostic',svg=(ASSETS/'nonlinear_consistency_readable.svg').read_bytes())
    s.equation(r'\overline{F(U)}\ne F(\bar U)',8.00,1.94,4.48,.69,size=30)
    s.txt('Nonlinear flux evaluation and\ncell averaging do not commute.',8.02,2.94,4.60,.73,18,MUTED,line_spacing=1.25)
    s.txt('Historical FV path: order ≈ 2.01\nState + Rusanov: order ≈ 5.00',8.02,3.93,4.59,.73,19,INK,True,line_spacing=1.28)
    s.txt('Experimental remedy\nConsistent FV state reconstruction.\nShock/positivity behavior unvalidated.',8.02,5.07,4.64,1.00,16.5,MUTED,line_spacing=1.24)
    s.band('This limitation affects classical TENO5 too; it is not a neural-only failure.',6.35,size=17)
    s.citation('Data: nonlinear_euler.json; semidiscrete exact-average divergence. WENO7 details → Backup B. Wall correction ≠ blast robustness.')
    s.finish('Expose averaging/flux inconsistency and distinguish smooth experimental remedy from shock validation.')

    s=Slide(deck,9,'Learn within verifiable numerical constraints','Research direction · reliable machine-learning-assisted methods')
    cards=[('Observed limitations','Negative stages · repair sources\nTransfer gaps · accuracy/oscillations'),
           ('Established techniques','Flux limiting · invariant domains\nMoments · conditional SSP'),
           ('Open research','Admissible learned corrections\nDG/SIAC moments · residual checks')]
    for j,(a,b) in enumerate(cards):
        x=.6+4.13*j;s.panel(x,1.76,3.86,1.23);s.txt(a,x+.19,1.94,3.48,.30,18,INK,True);s.txt(b,x+.19,2.39,3.48,.53,14.4,MUTED,line_spacing=1.20)
    s.txt('Shared-face limited flux',.78,3.19,6.55,.29,18,INK,True)
    s.equation(r'\widehat F^{limited}=\widehat F^{LO}+\theta\left(\widehat F^{NN}-\widehat F^{LO}\right)',.78,3.72,6.76,.66,size=27)
    s.txt('Independent space–time residual',8.04,3.19,4.56,.29,18,INK,True)
    s.equation(r'\mathscr R=\partial_t\widehat U+\partial_xF(\widehat U)',8.03,3.72,4.56,.66,size=27)
    s.txt('θ in [0,1] alone does not ensure positivity. A residual alone does not certify an error bound.',.79,4.62,11.80,.30,15.0,MUTED)
    s.txt('Giesselmann, Kwon & Krumscheid (2026) · RKDG + SIAC',.79,5.10,11.85,.31,18,INK,True)
    s.txt('Begin with checkable smooth-state, moment and stability assumptions; then test near shocks.\nThe convection–diffusion RKDG–SIAC paper does not certify this neural Euler solver.',.79,5.55,11.85,.58,15.2,MUTED,line_spacing=1.16)
    s.band('Can learned high-order reconstruction improve numerical accuracy while retaining\nprovable structural constraints and independently verifiable error control?',6.18,size=17.2)
    s.citation('Hu–Adams–Shu (2013) · Guermond et al. (2018) · Gottlieb et al. (2001) · arXiv:2604.01200. Discussion; backups follow.')
    s.finish('Close with constrained fluxes and independent residual assessment; no inherited neural-Euler guarantee.')

    s=Slide(deck,10,'Euler reconstruction and SSPRK3: the full map','Backup A · mathematical formulation')
    s.txt('Face characteristic decomposition',.78,1.79,6.0,.32,20,INK,True)
    s.equation(r'\lambda_j=v-c,\ v,\ v+c,\qquad c=\sqrt{\gamma p/\rho}',.78,2.31,6.0,.66,size=24)
    s.equation(r'g^\pm_{i+1/2,k}=\tfrac12 L_{i+1/2}\left(F(U_k)\pm a_{i+1/2}U_k\right)',.78,3.24,6.0,.72,size=24)
    s.equation(r'\widehat F_{i+1/2}=R_{i+1/2}\left[\mathcal R(g^+)+\mathcal R(g^-)\right]',.78,4.30,6.0,.74,size=24)
    s.txt('Same face basis for both LF directions.\nNegative branch: shift by one cell and reverse.\nDefault a: maximum speed of adjacent cells.',.79,5.47,6.05,.95,16.1,MUTED,line_spacing=1.28)
    s.txt('SSPRK3 · admissible at every stage',7.29,1.79,5.3,.32,19,INK,True)
    s.equation(r'U^{(1)}=U^n-\Delta tD(U^n)',7.31,2.43,5.16,.62,size=25)
    s.equation(r'U^{(2)}=\tfrac34U^n+\tfrac14\left[U^{(1)}-\Delta tD(U^{(1)})\right]',7.31,3.43,5.16,.73,size=24)
    s.equation(r'U^{n+1}=\tfrac13U^n+\tfrac23\left[U^{(2)}-\Delta tD(U^{(2)})\right]',7.31,4.47,5.16,.73,size=24)
    s.equation(r'\rho>0,\qquad E-\frac{m^2}{2\rho}>0',7.31,5.65,5.16,.67,size=25)
    s.citation('D(U) = (F̂right − F̂left)/Δx. Implementation: euler_jax.py:166–254; numerics_jax.py:92–115. SSP: Gottlieb et al. (2001).')
    s.finish('Detailed algorithm backup; stage restrictions explicit.')

    s=Slide(deck,11,'WENO7: a defect and an empirical mitigation','Backup B · floating-point conditioning')
    s.equation(r'\alpha_k=\frac{d_k}{(\epsilon+\beta_k)^2},\qquad\omega_k=\frac{\alpha_k}{\sum_j\alpha_j}',.83,1.87,11.7,.91,size=31)
    s.txt('Reproduced coefficient-layer test',.80,3.17,5.8,.35,21,INK,True)
    s.txt('100,000 stored normalized Euler stencils\nε = 10⁻⁶: 10 non-finite coefficient rows\nε = 10⁻⁴: 0 in the same forward test',.81,3.78,5.89,1.20,19,MUTED,line_spacing=1.38)
    s.txt('Unresolved NaN mechanism',7.28,3.17,5.21,.35,19,INK,True)
    s.equation(r'(10^{-6})^2=10^{-12}\gg 10^{-38}',7.30,3.77,5.10,.68,size=27)
    s.txt('Expanded β polynomials can cancel.\nAn instrumented near-flat row has β < 0.\nExact NaN operation remains unisolated.',7.30,4.78,5.18,1.04,17,MUTED,line_spacing=1.30)
    s.band('Epsilon mitigation removed sampled NaNs; it did not prove order or Euler stability.',6.38,size=16.5)
    s.citation('WenoNetworks.py:106–137 · commit 753d526 · audit §§3.6/10.4. Backend/graph changes can affect bad-row counts.')
    s.finish('Reproducible defect and mitigation; do not repeat incorrect epsilon-squared underflow claim.')

    s=Slide(deck,12,'DG and SIAC: reconstruction before certification','Backup C · research connection and assumptions')
    s.txt('Moment-preserving approximation',.80,1.77,6.00,.34,20,INK,True)
    s.equation(r'U_h|_{I_i}=\sum_{\ell=0}^{q}U_{i,\ell}\varphi_\ell(x)',.80,2.36,6.00,.70,size=27)
    s.equation(r'\widehat U=K_h*U_h,\qquad\int K=1,\quad\int x^rK(x)\,dx=0',.80,3.33,6.00,.78,size=24)
    s.txt('r = 1,…,s: chosen reproduction moments.\nUnit mass ≠ cellwise mean preservation.\nSigned kernels need not preserve positivity.',.82,4.48,5.91,1.12,16.3,MUTED,line_spacing=1.32)
    s.txt('Independent residual assessment',7.31,1.77,5.20,.34,20,INK,True)
    s.equation(r'\mathscr R=\partial_t\widehat U+\partial_xF(\widehat U)',7.32,2.39,5.13,.69,size=28)
    s.txt('Relative entropy needs regular reconstruction\nand suitable admissible state ranges.\nSharp gradients can weaken error bounds.\nA residual sensor alone is not a certificate.',7.32,3.43,5.17,1.56,17.0,MUTED,line_spacing=1.33)
    s.txt('2026 paper: periodic Cartesian RKDG + SIAC;\nconvection–diffusion with compatible entropy.\nNo learned filter or neural Euler shock theorem.',7.32,5.33,5.15,1.0,15.3,MUTED,line_spacing=1.25)
    s.txt('Primary references · full metadata and links in Sources_and_Evidence.md',.78,6.39,11.85,.25,12.6,INK,True)
    s.citation('Hu et al., JCP 242 (2013) 169–180 · Zhang & Shu, JCP 231 (2012) 2245–2258 · Gottlieb et al., SIAM Rev. 43 (2001) 89–112.')
    s.txt('Guermond et al., SISC 40 (2018) A3211–A3239, doi:10.1137/17M1149961 · Giesselmann–Kwon–Krumscheid, arXiv:2604.01200.',.66,6.59,11.95,.18,8.5,MUTED)
    s.finish('DG/SIAC scope and assumptions; future learned extension clearly separated.')

def tex_escape(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    value=''.join(mapping.get(c,c) for c in s)
    supers=dict(zip('⁻⁰¹²³⁴⁵⁶⁷⁸⁹','-0123456789'))
    return re.sub('[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+',lambda m:r'\textsuperscript{'+''.join(supers[c] for c in m[0])+'}',value)

def write_tex():
    lines=[r'\documentclass[aspectratio=169]{beamer}',r'\usepackage{amsmath,amssymb,mathrsfs,fontspec,tikz}',
        r'\setsansfont{Liberation Sans}',r'\setmainfont{Liberation Sans}',r'\usefonttheme[onlymath]{serif}',
        r'\geometry{paperwidth=960.009bp,paperheight=540bp}',r'\setbeamertemplate{navigation symbols}{}',
        r'\setbeamertemplate{headline}{}\setbeamertemplate{footline}{}',r'\setbeamersize{text margin left=0pt,text margin right=0pt}',
        r'\definecolor{ink}{HTML}{242424}',r'\hypersetup{pdftitle={Research Interview — Yasser Baouzil — KIT},pdfauthor={Yasser Baouzil}}',
        r'\begin{document}',r'\begin{frame}[plain]',r'\begin{tikzpicture}[remember picture,overlay]',
        r'\node[anchor=north west,inner sep=0pt] at(current page.north west){\includegraphics[width=\paperwidth,height=\paperheight]{revised_source/assets/V4_landing-1.pdf}};',
        r'\end{tikzpicture}',r'\end{frame}']
    for n,items in registry.items():
        lines += [f'% Slide {n}: '+('MAIN' if n<=9 else 'BACKUP'),r'\begin{frame}[plain]',r'\begin{tikzpicture}[remember picture,overlay]']
        for ob in items:
            x,y,w,h=[float(ob[k]) for k in ['x','y','w','h']]
            at=f'([xshift={x:.6f}in,yshift=-{y:.6f}in]current page.north west)'
            if ob['type']=='shape':
                opts=[]
                if ob['fill']:opts.append('fill={rgb,255:red,'+str(int(ob['fill'][0:2],16))+';green,'+str(int(ob['fill'][2:4],16))+';blue,'+str(int(ob['fill'][4:6],16))+'}')
                if not ob['stroke']:opts.append('draw=none')
                else:
                    c=ob['stroke'];opts.append('draw={rgb,255:red,'+str(int(c[:2],16))+';green,'+str(int(c[2:4],16))+';blue,'+str(int(c[4:],16))+'}')
                if ob['radius']:opts.append('rounded corners=3bp')
                end=f'([xshift={x+w:.6f}in,yshift=-{y+h:.6f}in]current page.north west)'
                lines.append(r'\path['+','.join(opts)+'] '+at+' rectangle '+end+';')
            elif ob['type']=='text':
                c=ob['color'];color='{rgb,255:red,'+str(int(c[:2],16))+';green,'+str(int(c[2:4],16))+';blue,'+str(int(c[4:],16))+'}'
                align={'l':'left','ctr':'center','r':'right'}[ob['align']]
                content=tex_escape(ob['text']).replace('\n',r'\\ ')
                bold=r'\bfseries ' if ob['bold'] else ''
                font=f"\\fontsize{{{ob['size']:.2f}bp}}{{{ob['size']*ob['spacing']:.2f}bp}}\\selectfont {bold}"
                lines.append(r'\node[anchor=north west,inner sep=0pt,outer sep=0pt,text='+color+f',align={align},text width={w:.6f}in,font={{{font}}}] at '+at+' {'+content+'};')
            elif ob['type']=='image':
                lines.append(r'\node[anchor=north west,inner sep=0pt] at '+at+f' {{\\includegraphics[width={w:.6f}in,height={h:.6f}in]{{{ob["path"]}}}}};')
            elif ob['type']=='equation':
                cx=x+w/2;cy=y+h/2;pt=ob['size'];color=ob['color']
                lines.append(r'\node[inner sep=0pt,anchor=center] at '+f'([xshift={cx:.6f}in,yshift=-{cy:.6f}in]current page.north west)'+f' {{\\color[HTML]{{{color}}}\\fontsize{{{pt:.3f}bp}}{{{pt*1.2:.3f}bp}}\\selectfont $\\displaystyle {ob["latex"]}$}};')
        lines += [r'\end{tikzpicture}',r'\end{frame}']
    lines += [r'\end{document}']
    (OUT/f'{NAME}.tex').write_text('\n'.join(lines)+'\n')

def package(deck):
    # Keep the supplied native landing slide, replacing only subsequent visuals.
    root=ET.fromstring(deck.parts['ppt/presentation.xml']);lst=root.find('p:sldIdLst',NS)
    for node in list(lst):lst.remove(node)
    rels=ET.fromstring(deck.parts['ppt/_rels/presentation.xml.rels'])
    for rel in list(rels):
        if rel.get('Type').endswith(('/slide','/notesMaster')):rels.remove(rel)
    for n in range(1,13):
        rid=f'rIdInterview{n}';ET.SubElement(rels,f'{{{REL}}}Relationship',Id=rid,Type=NS['r']+'/slide',Target=f'slides/slide{n}.xml')
        q=base.e('p:sldId',lst,id=300+n);q.set(f'{{{NS["r"]}}}id',rid)
    deck.parts['ppt/presentation.xml']=base.xml(root);deck.parts['ppt/_rels/presentation.xml.rels']=base.xml(rels)
    for name in list(deck.parts):
        m=re.fullmatch(r'ppt/slides/(?:_rels/)?slide(\d+)\.xml(?:\.rels)?',name)
        if m and int(m.group(1))>12:del deck.parts[name]
        if name.startswith(('ppt/notesSlides/','ppt/notesMasters/')):del deck.parts[name]
    types=ET.fromstring(deck.parts['[Content_Types].xml'])
    for q in list(types):
        if q.get('PartName','').startswith(('/ppt/notesSlides/','/ppt/notesMasters/')):types.remove(q)
        m=re.fullmatch(r'/ppt/slides/slide(\d+)\.xml',q.get('PartName',''))
        if m and int(m.group(1))>12:types.remove(q)
    present={q.get('PartName') for q in types}
    for n in range(1,13):
        name=f'/ppt/slides/slide{n}.xml'
        if name not in present:ET.SubElement(types,f'{{{CT}}}Override',PartName=name,ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
    deck.parts['[Content_Types].xml']=base.xml(types)
    add_notes(deck)
    output=OUT/f'{NAME}.pptx';deck.save(output)
    with ZipFile(output) as z:parts={k:z.read(k) for k in z.namelist()}
    core=ET.fromstring(parts['docProps/core.xml']);dc='{http://purl.org/dc/elements/1.1/}'
    if core.find(dc+'title') is not None:core.find(dc+'title').text='Research Interview — KIT — Yasser Baouzil'
    parts['docProps/core.xml']=base.xml(core)
    app=ET.fromstring(parts['docProps/app.xml'])
    for node in app.iter():
        if ET.QName(node).localname=='Slides':node.text='12'
    parts['docProps/app.xml']=base.xml(app)
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        for k,v in parts.items():z.writestr(k,v)

def add_notes(deck):
    doc=json.loads((HERE/'notes_content.json').read_text())
    types=ET.fromstring(deck.parts['[Content_Types].xml'])
    for n in range(1,13):
        note=doc[str(n)]
        guidance=''
        if n<=9:
            guidance=f"\n\nEQUATION GUIDE\n{note['equation_guide']}\n\nOBSERVATIONS\n{note['observed']}\n\nUNCERTAINTIES AND REMEDIES\n{note['limits_and_remedies']}"
        text=f"Slide {n}: {note['title']}\nDuration: {note['duration']}\n\nSCRIPT\n{note['script']}{guidance}\n\nMATHEMATICAL TERMS\n{note['terms']}\n\nPREPARATION\n{note['preparation']}\n\nQUESTIONS\n"+'\n\n'.join(q['q']+'\n'+q['a'] for q in note['qa'])
        root=base.e('p:notes');cs=base.e('p:cSld',root);tree=base.e('p:spTree',cs)
        nv=base.e('p:nvGrpSpPr',tree);base.e('p:cNvPr',nv,id='1',name='');base.e('p:cNvGrpSpPr',nv);base.e('p:nvPr',nv);base.e('p:grpSpPr',tree)
        sp=base.e('p:sp',tree);nv=base.e('p:nvSpPr',sp);base.e('p:cNvPr',nv,id='2',name='Speaker notes');base.e('p:cNvSpPr',nv);base.e('p:ph',base.e('p:nvPr',nv),type='body',idx='1');base.e('p:spPr',sp)
        body=base.e('p:txBody',sp);base.e('a:bodyPr',body);base.e('a:lstStyle',body)
        for line in text.splitlines():
            p=base.e('a:p',body);r=base.e('a:r',p);base.e('a:rPr',r,lang='en-US');base.e('a:t',r).text=line
        base.e('a:masterClrMapping',base.e('p:clrMapOvr',root))
        path=f'ppt/notesSlides/notesSlide{n}.xml';deck.parts[path]=base.xml(root)
        rel=ET.Element(f'{{{REL}}}Relationships');ET.SubElement(rel,f'{{{REL}}}Relationship',Id='rId1',Type=NS['r']+'/slide',Target=f'../slides/slide{n}.xml')
        deck.parts[f'ppt/notesSlides/_rels/notesSlide{n}.xml.rels']=base.xml(rel)
        sr=ET.fromstring(deck.parts[f'ppt/slides/_rels/slide{n}.xml.rels'])
        for q in list(sr):
            if q.get('Type').endswith('/notesSlide'):sr.remove(q)
        ET.SubElement(sr,f'{{{REL}}}Relationship',Id='rIdInterviewNotes',Type=NS['r']+'/notesSlide',Target=f'../notesSlides/notesSlide{n}.xml')
        deck.parts[f'ppt/slides/_rels/slide{n}.xml.rels']=base.xml(sr)
        ET.SubElement(types,f'{{{CT}}}Override',PartName='/'+path,ContentType='application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml')
    deck.parts['[Content_Types].xml']=base.xml(types)

def make_math_source():
    ls=[r'\documentclass{article}',r'\usepackage{amsmath,amssymb,mathrsfs,lmodern,xcolor}',r'\usepackage[active,tightpage]{preview}',r'\setlength\PreviewBorder{1pt}',r'\begin{document}']
    for j in jobs.values():ls.extend([r'\begin{preview}',r'\hbox{',f"\\color[HTML]{{{j['color']}}}\\fontsize{{{j['fontsize']}}}{{{1.2*j['fontsize']}}}\\selectfont",r'$\displaystyle '+j['latex']+'$',r'}',r'\end{preview}'])
    ls += [r'\end{document}'];MATH.mkdir(exist_ok=True);(MATH/'equations.tex').write_text('\n'.join(ls)+'\n');(MATH/'equations.json').write_text(json.dumps(list(jobs.values()),indent=2))

def load_math():
    jlist=json.loads((MATH/'equations.json').read_text());pdf=MATH/'equations.pdf'
    for n,j in enumerate(jlist,1):
        png=MATH/f'eq{n:03d}.png';svg=MATH/f'eq{n:03d}.svg'
        if not png.exists() or png.stat().st_mtime<pdf.stat().st_mtime:subprocess.run(['pdftocairo','-f',str(n),'-l',str(n),'-singlefile','-png','-transp','-r','300',str(pdf),str(png.with_suffix(''))],check=True)
        if not svg.exists() or svg.stat().st_mtime<pdf.stat().st_mtime:subprocess.run(['pdftocairo','-svg','-f',str(n),'-l',str(n),str(pdf),str(svg)],check=True)
        root=ET.parse(str(svg)).getroot();w=float(root.get('width').removesuffix('pt'))/72;h=float(root.get('height').removesuffix('pt'))/72
        arts[key(j['latex'],j['fontsize'],j['color'])]=(png.read_bytes(),svg.read_bytes(),w,h)

def main():
    global preparing
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--build',action='store_true');args=p.parse_args()
    preparing=args.prepare
    if preparing:charts()
    else:load_math()
    deck=base.Deck(ASSETS/'V4_editable_template.pptx')
    # New assets must never overwrite media referenced by the native V4 cover.
    deck.media=max([int(m[1]) for name in deck.parts if (m:=re.search(r'/refined_(\d+)\.',name))]+[0])
    deck_content(deck)
    if preparing:make_math_source();print('Prepared charts and',len(jobs),'LaTeX equations.');return
    package(deck);write_tex();(HERE/'slide_layout.json').write_text(json.dumps(registry,indent=2));(HERE/'validation/equation_inventory.json').write_text(json.dumps(deck.equations,indent=2))
    print('Built editable PPTX, Beamer source, slide manifest and equation inventory.')

if __name__=='__main__':main()
