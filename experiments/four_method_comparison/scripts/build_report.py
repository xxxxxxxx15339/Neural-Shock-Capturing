"""Render scientific reports from machine-readable measurements, without hand-entered scores."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts import methods as m
from experiments.four_method_comparison.scripts.collect_existing_results import csv_write
import numpy as np

OUT=m.OUT;RAW=OUT/'results/raw'

def table(headers,records):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in records])
def f(x):return 'N/A' if x is None else f'{x:.7g}'
def p(x):return 'N/A' if x is None else f'{x:+.3f}%'
def gain(a,b):return None if a is None or b is None or b==0 else 100*(1-a/b)

def main():
    rows=json.loads((RAW/'evaluated_rows.json').read_text());manifest=json.loads((OUT/'configs/benchmark_manifest.json').read_text())
    cases={c['id']:c for c in manifest['cases']}
    def get(case,n,method):return next((r for r in rows if r['case']==case and r['cells']==n and r['method']==method),{})
    headline=[('sod',128,'density_l1'),('sod',128,'l1'),('lax',128,'l1'),('shu_osher',128,'l1'),
              ('shu_osher',128,'regional_density_l1'),('burgers_pre',128,'l1'),('burgers_post',128,'l1'),
              ('burgers_hat',128,'l1'),('advection_square',128,'l1'),('advection_smooth',256,'l1')]
    records=[];pct=[]
    for case,n,metric in headline:
        vals=[get(case,n,k).get(metric) for k in m.MAIN]
        records.append([case,n,metric,*[f(v) if get(case,n,k).get('success') else 'FAILED' for k,v in zip(m.MAIN,vals)]])
        pct.append([case,n,metric,p(gain(vals[1],vals[0])),p(gain(vals[2],vals[0])),p(gain(vals[3],vals[0])),p(gain(vals[3],vals[2]))])
    primary=table(['Benchmark','N','Metric','TENO5','NENO6','NN-TENO5','V2 learned head'],records)
    percents=table(['Benchmark','N','Metric','NENO6 vs TENO5','NN vs TENO5','V2 vs TENO5','V2 vs NN'],pct)
    ab=[]
    for case in manifest['ablation']['cases']:
        pure=get(case,128,'head_backbone_only');fixed=get(case,128,'head_fixed_blend');learned=get(case,128,'learned_head')
        ab.append([case,f(pure.get('l1')) if pure.get('success') else 'FAILED',f(fixed.get('l1')) if fixed.get('success') else 'FAILED',f(learned.get('l1')),
                   p(gain(learned.get('l1'),pure.get('l1'))),p(gain(learned.get('l1'),fixed.get('l1')))])
    abtable=table(['N=128 case','Pure same backbone','Fixed same backbone','Learned same backbone','Learned vs pure','Learned vs fixed'],ab)
    stability=[];failures=[]
    for method in m.MAIN:
        rs=[r for r in rows if r['method']==method];stress=[r for r in rs if cases[r['case']].get('robustness')]
        fail=[r for r in rs if not r['success']]
        stability.append([m.NAMES[method],f'{sum(r["success"] for r in rs)}/{len(rs)}',f'{sum(r["success"] for r in stress)}/{len(stress)}',
            f(max(r['conservation_max_abs'] for r in rs)),f(min(r['raw_min_density'] for r in rs if r['physics']=='euler' and r['raw_min_density'] is not None)),
            f(min(r['raw_min_pressure'] for r in rs if r['physics']=='euler' and r['raw_min_pressure'] is not None))])
        for r in fail:failures.append([r['case'],r['cells'],m.NAMES[method],r['failure']['step'],r['failure']['stage'],f(r['actual_time']),f(r['raw_min_density']),f(r['raw_min_pressure'])])
    stabletable=table(['Method','All completed','Stress completed','Max budget residual','Raw min rho incl failures','Raw min p incl failures'],stability)
    failtable=table(['Case','N','Method','First failed step','Stage','Last complete time','Raw min rho','Raw min p'],failures) if failures else 'No main-method failures.'
    runtable=[];runtimes=[];budget=[]
    for case in ('sod','lax','shu_osher','burgers_post','advection_square'):
        base=get(case,128,'teno5').get('runtime_median_s')
        for method in m.MAIN:
            r=get(case,128,method);runtime=r.get('runtime_median_s')
            runtable.append([case,m.NAMES[method],f(runtime),f(r.get('runtime_iqr_s')),f(runtime/base) if runtime and base else 'N/A',f(r.get('compile_rollout_seconds'))])
    runtime_table=table(['N=128 case','Method','Warm median s','IQR s','Cost / TENO5','Rollout compile s'],runtable)
    for physics in ('euler','scalar'):
        for n in sorted({r['cells'] for r in rows}):
            for method in m.MAIN:
                rs=[r for r in rows if r['method']==method and r['physics']==physics and r['cells']==n]
                valid=[r for r in rs if r.get('l1') is not None]
                ratios=[r['l1']/get(r['case'],n,'teno5')['l1'] for r in valid if get(r['case'],n,'teno5').get('l1',0)>0]
                runtimes.append(dict(physics=physics,cells=n,method=method,count=len(rs),failures=sum(not r['success'] for r in rs),
                    metric_cases=len(valid),mean_error=np.mean([r['l1'] for r in valid]) if valid else None,
                    median_error=np.median([r['l1'] for r in valid]) if valid else None,
                    maximum_error=max((r['l1'] for r in valid),default=None),
                    median_paired_ratio=np.median(ratios) if ratios else None,maximum_paired_ratio=max(ratios,default=None),
                    interpretation='Descriptive within physics/grid; not pooled as a headline; no independent-trajectory CI'))
    csv_write(OUT/'results/family_statistics.csv',runtimes)
    for case in ('sod','lax','shu_osher','burgers_post'):
        for method in ('neno6','nn_teno5','learned_head'):
            for n in (128,256):
                r=get(case,n,method)
                if not r.get('runtime_median_s') or r.get('l1') is None:continue
                controls=[get(case,k,'teno5') for k in (128,256)]
                controls=[a for a in controls if a.get('runtime_median_s') and a.get('l1') is not None]
                if not controls:continue
                b=min(controls,key=lambda a:abs(np.log(a['runtime_median_s']/r['runtime_median_s'])))
                ratio=r['runtime_median_s']/b['runtime_median_s']
                budget.append(dict(case=case,method=method,cells=n,teno_cells=b['cells'],
                    method_seconds=r['runtime_median_s'],teno_seconds=b['runtime_median_s'],cost_ratio=ratio,
                    error_ratio=r['l1']/b['l1'],within_25percent_budget=.8<=ratio<=1.25,
                    interpretation='Nearest measured classical cost among128/256; exact measured grids only'))
    csv_write(OUT/'results/nearest_measured_budget.csv',budget)
    orders=[]
    for method in m.MAIN:
        previous=None
        for n in (64,128,256):
            r=get('advection_smooth',n,method)
            if r.get('l1') is None:continue
            order=None if previous is None else np.log2(previous/r['l1'])
            orders.append(dict(case='advection_smooth',method=method,cells=n,dt=r['dt'],l1=r['l1'],order=order))
            previous=r['l1']
    csv_write(OUT/'results/smooth_convergence.csv',orders)
    ordertable=table(['Method','N','L1','Observed p'],[[m.NAMES[r['method']],r['cells'],f(r['l1']),f(r['order'])] for r in orders])
    lam=json.loads((RAW/'lambda_statistics.json').read_text())
    lamtable=table(['Case','N','Mean lambda','Min','Max','Std','Smooth mean','Shock mean','Corr local error reduction'],
        [[r['case'],r['cells'],f(r['mean']),f(r['minimum']),f(r['maximum']),f(r['std']),f(r['mean_smooth']),f(r['mean_shock']),f(r.get('correlation_local_error_reduction'))] for r in lam])
    archive=json.loads((RAW/'archive_reproduction_summary.json').read_text())
    reference_report=json.loads((RAW/'reference_sensitivity.json').read_text())
    reftable=table(['Comparison at target grid','N','Mean conserved L1 difference','Density L1 difference'],
        [[r['coarser']+' → '+r['finer'],r['cells'],f(r['l1']),f(r['density_l1'])] for r in reference_report])
    exact_cases=['sod','lax','burgers_pre','burgers_post','burgers_hat','advection_square','advection_smooth']
    improvements={}
    for method in ('nn_teno5','learned_head','neno6'):
        options=[]
        for case in exact_cases:
            for n in cases[case]['grids']:
                r=get(case,n,method);b=get(case,n,'teno5')
                g=gain(r.get('l1'),b.get('l1'))
                if g is not None:options.append((g,case,n))
        improvements[method]=dict(best=max(options),worst=min(options)) if options else None
    mix_options=[]
    for case in exact_cases:
        for n in cases[case]['grids']:
            a=get(case,n,'learned_head');b=get(case,n,'nn_teno5');g=gain(a.get('l1'),b.get('l1'))
            if g is not None:mix_options.append((g,case,n))
    findings=dict(improvements=improvements,best_v2_vs_original=max(mix_options) if mix_options else None,
        worst_v2_vs_original=min(mix_options) if mix_options else None,total_rollouts=len(rows),main_rollouts=sum(r['method'] in m.MAIN for r in rows),
        total_failures=sum(not r['success'] for r in rows),main_failures=sum(not r['success'] for r in rows if r['method'] in m.MAIN))
    geometry=list(csv.DictReader((OUT/'results/wall_geometry_control.csv').open())) if (OUT/'results/wall_geometry_control.csv').exists() else []
    findings['supplemental_wall_runs']=len(geometry)
    findings['supplemental_wall_failures']=sum(r['success']=='False' for r in geometry)
    (RAW/'headline_findings.json').write_text(json.dumps(findings,indent=2))
    results=f'''# Verified four-method results

Evidence status for all following scores: **NEW MATCHED EVALUATION**. Full precision
values and failure records are in `results/raw/evaluated_rows.json`; every displayed
number is generated from that file. The hashed protocol preceded the tests.
{len(rows)} measured rollouts include {findings['main_rollouts']} main-method cases and
same-backbone controls; {findings['total_failures']} failures are retained.

## Model identities

TENO5 is the unchanged classical implementation. NENO6 is the verified
paper-linked author checkpoint (6→64→32→16→8→4), ported to the common solver.
NN-TENO5 is the original Keras HDF5 classifier with calibrated hard thresholds.
V2 is **phase21 rich learned head for Euler**, **run6 linear learned head for scalar**.
These are different physics-specific checkpoints, not one universal model.
Euler applies the same learned head separately to each of three characteristic
fields and both flux-splitting directions, giving six lambdas per interface.
The scalar blending formula applies to each such reconstruction before projection
to one shared conservative face flux. Plotted means average these six values;
they are not a single fluid-level learned coefficient.
The later dataset-V2 fixed-lambda checkpoint is excluded from this identity.
See `AUDIT.md` and `CHECKPOINT_INVENTORY.csv` for checksums and training provenance.

## Main comparison

{primary}

Euler `l1` is the arithmetic mean of density, momentum and energy L1. Scalar
`l1` is mean absolute scalar error. Entries from different norms are not averaged
into a headline percentage. NN-TENO5 and learned-head backbones differ.

## Percentage improvement (positive means smaller error)

{percents}

V2-vs-NN describes the whole saved method difference. It does not isolate the head.
Zeros/ties and regressions are preserved; no threshold or coefficient was fitted
to these tests. Archived results have different settings and remain separate.

## Same-backbone causal control

All three columns below share the selected phase21/run6 backbone. Fixed lambda
is the predeclared sigmoid(-3)={manifest['fixed_lambda']:.17g}.

{abtable}

Positive learned-vs-fixed values quantify the incremental value of an adaptive
head over this specific fixed coefficient. Negative values reject that advantage
on that case. This is not an optimized-fixed-head comparison or a general head
capacity theorem. A pure-backbone failure demonstrates the value of reducing the
neural contribution under this protocol, not that learning lambda is necessary.

## Physical admissibility and discrete conservation

{stabletable}

Raw minima include the first invalid stage. These minima are not post-repair
values. Completed fractions include every declared case. Conservation residuals
include boundary exchange and are measured for accepted steps up to the stopping
time. Wall mass/energy fluxes are explicitly zero; momentum need not be conserved
inside a reflecting box because wall pressure exchanges momentum.

{failtable}

All failed candidates and last admissible states are retained in NPZ files.
No finite full-time error is assigned to failure. No direct state clipping,
averaging, positivity repair, retry or low-order fallback is used in these runs.
This is not a positivity theorem. Strong-blast scores are stability only.
For the point grid, mirrored outer faces lie half a mesh beyond the first/last
sample: these are Woodward–Colella-style stress cases, not certified canonical
blast accuracy results on a wall-aligned cell-average grid.

## Smooth full-solution convergence

{ordertable}

The transport problem has exact references and dt≤.15*dx^(5/3). These are observed
full-solution orders, not formal guarantees for Euler/Burgers or shock problems.
SSPRK3 remains third order in time. NENO6 means six-point stencil; the author
optimized coefficients and finite-precision implementation must be judged by
their observed convergence, not the digit in the name. No state-flux solver
correction is introduced into this comparison.

## Shu–Osher numerical-reference sensitivity

{reftable}

The final reference is TENO5 N=2048 at dt/2, interpolated to evaluation points.
These differences estimate sensitivity, not rigorous uncertainty bounds. Small
reported gains of the same size as reference changes are unresolved as accuracy
claims. `reference_sensitivity` in every Shu row records errors against each
ladder member, so ranking sensitivity is inspectable. An exact shock-entropy
solution and an independently different reference solver are unavailable.

## Mixing-head behavior

{lamtable}

Smooth/shock masks are the lower half/top decile of measured density-gradient
magnitudes. Correlations are descriptive across dependent spatial samples.
There is no calibrated-uncertainty interpretation. The rich head uses only
three probabilities and five normalized split-flux values, not Mach, pressure
gradient or the entire proposed physics vector. The same-backbone ablation,
not lambda variability by itself, determines whether adaptivity helps.

## Computational costs

{runtime_table}

Timings use one JAX CPU environment, synchronized three-repeat warmed device
rollouts; IQR describes repeat dispersion. Compilation and host diagnostic time
are separate. CPU inference ports are compared; this is not TensorFlow-vs-JAX
framework speed. Nearest measured cost comparisons across128/256 appear in
`results/nearest_measured_budget.csv`; only rows flagged within25% have roughly
comparable costs, and no interpolated equal-cost accuracy is invented.

## Archived results and reproduction discrepancies

The evidence catalog records every archived scalar with source hash/JSON pointer.
Original scalar and phase18 Euler entry points were rerun into this directory.
Across {archive['compared_values']} checked values, {archive['mismatches']} exceed
1e-10, with maximum absolute discrepancy {archive['max_abs_difference']:.9g}.
The phase18 Euler values, classical scalar results and smooth neural scalar
results reproduce. The original-network square-wave result and its threshold
controls differ slightly; current CPU repeats are deterministic. Archive identity
metadata are insufficient to establish the cause. They remain **ARCHIVED /
REPRODUCTION DISCREPANCY**, not silently marked reproduced. The threshold=1
control is not unconditional all-keep because exactly saturated sigmoid outputs
still meet the >= discard rule. Archived Burgers T=.6 is pre-shock (2/pi≈.6366),
despite its historical smooth-to-shock name. A fidelity-factor acceptance is not
necessarily an improvement. See `results/archive_reproduction.csv`.

## Interpretation and limitations

This establishes reproducible, case-specific differences among authentic saved
reconstructions under a common solver. It separates whole-method and same-backbone
head effects and retains failures and computational costs. It does not establish
universal neural superiority, positivity/TVD/entropy guarantees, fifth-order
nonlinear Euler accuracy, multi-seed training generalization or untouched-test
performance. No trained artifact is promoted from this small comparative suite.
The strongest existing baseline should be chosen per application only with
independent validation. Source/checkpoint integrity and tests are recorded in
`results/raw/package_verification.json` and `logs/`.

## Reproduction

Run `bash experiments/four_method_comparison/reproduce.sh`. Existing verified
run caches are retained; numerical fixtures can be independently regenerated
with `REEXPORT=1`. Exact individual commands and environment paths are in that
script and `requirements_notes.md`. No training, checkpoint writing or GitHub
push is part of reproduction.
'''
    smooth=get('advection_smooth',256,'learned_head');smooth_base=get('advection_smooth',256,'teno5')
    smooth_ratio=smooth['l1']/smooth_base['l1']
    scalar_head=get('burgers_pre',128,'learned_head');scalar_fixed=get('burgers_pre',128,'head_fixed_blend')
    smooth_diag=json.loads((RAW/'smooth_input_diagnostics.json').read_text()) if (RAW/'smooth_input_diagnostics.json').exists() else []
    diagtable=table(['N=256 initial smooth profile','Any-discard fraction','Discarded candidate fraction','Mean lambda'],
        [[m.NAMES[r['method']],f(r['any_candidate_discard_fraction']),f(r['discarded_candidate_fraction']),f(r.get('mean_lambda'))]
         for r in smooth_diag if r['cells']==256 and r['stage']=='initial'])
    growth=[r for r in rows if r.get('finite_growth_warning')]
    growthtable=table(['Case','N','Method','Finite final L1','Max abs state','Amplification'],
        [[r['case'],r['cells'],m.NAMES[r['method']],f(r.get('l1')),f(r['maximum_absolute_value']),f(r['amplification_factor'])] for r in growth])
    geometrytable=table(['Wall-aligned case','N','Method','Completed','Last complete time','Raw min pressure','First failed stage'],
        [[r['case'],r['cells'],m.NAMES[r['method']],r['success'],r['actual_time'],r['raw_min_pressure'],r['first_invalid_stage']] for r in geometry])
    timecontrols=list(csv.DictReader((OUT/'results/smooth_time_control.csv').open())) if (OUT/'results/smooth_time_control.csv').exists() else []
    timecontroltable=table(['N=256 method','L1 at dt','L1 at dt/2','Solution change L1','Change / original error'],
        [[m.NAMES[r['method']],r['dt_l1'],r['dt_half_l1'],r['temporal_change_l1'],r['temporal_change_fraction']] for r in timecontrols])
    budgettable=table(['Case','Neural method / N','TENO5 N','Neural seconds','TENO5 seconds','Time ratio','Error ratio'],
        [[r['case'],m.NAMES[r['method']]+' / '+str(r['cells']),r['teno_cells'],f(r['method_seconds']),f(r['teno_seconds']),f(r['cost_ratio']),f(r['error_ratio'])]
         for r in budget if r['within_25percent_budget']])
    supplement=f'''
## Measured conclusions and diagnostic controls

The scalar learned head improves pre-shock Burgers N=128 by
{p(gain(scalar_head['l1'],scalar_fixed['l1']))} relative to the **same-backbone
fixed blend**, in addition to its main-model improvement. This is a genuine
case-specific incremental head result for the older scalar3→1 head. It does not
establish a benefit for the richer Euler head, which slightly worsens all three
N=128 Euler ablations relative to the predeclared fixed blend. The Euler lambda
range is only about2.24e-5, around .04736, so it behaves almost like fixed mixing.

The smooth control is a material regression: scalar V2 L1 at N=256 is
{f(smooth['l1'])}, versus {f(smooth_base['l1'])} for TENO5/original NN,
an error ratio of {smooth_ratio:.6g}. The absolute errors remain small, but the
observed order drops to about3 instead of5. The main extremum statistics include
this control; it is not omitted to make neural gains look larger.

{diagtable}

These rejection fractions are measured on the analytically smooth initial
stencils. They identify active neural stencil removal where classical selection
keeps all candidates. This is consistent with lower-order neural contributions
controlling the nonzero blend's asymptotic error; it is a diagnostic mechanism,
not a proof of the full nonlinear scheme's order. The author
[NENO6 paper, section4.1](https://arxiv.org/pdf/2207.08500) uses an optimized
**fourth-order** background constraint, not sixth order; its full neural method's
order must still be measured. Our observed NENO6 smooth orders are in the table.

The separately predeclared finest-grid time-step control gives:

{timecontroltable}

The solution changes are small compared with the measured smooth errors. Thus
the observed order loss is not explained by the tested SSPRK3 timestep alone.
Scaling dt as dx^(5/3) makes temporal truncation scale as dx^5; the dt/2 test
quantifies its actual contribution. This is not fifth-order temporal convergence.

Finite completion does not imply scalar stability. The pure scalar head-backbone
advection control grows catastrophically while remaining finite, and is retained
with its full large error and an amplification warning:

{growthtable}

The tenfold amplification flag is descriptive and does not terminate the run or
replace its error. Its absolute budget residual is retained, and roundoff
normalization accounts for the very large absolute state magnitudes. No claim
of a physically useful completed solution is made for that control.

### Comparable measured computational budgets

{budgettable}

These are measured pairs within25% in runtime, without cost interpolation. For
post-shock Burgers, original NN at N=128 costs almost the same as TENO5 N=256
and has about1.58 times its error. Thus its same-grid accuracy advantage does
not translate into an advantage at this measured computational budget. The Shu
pair costs more and its small error difference remains reference-sensitive.

### Physical wall-alignment control

The initial historical point grid puts reflected outer faces at half nodes.
A separate protocol was declared before running a wall-aligned control on
x=(i+.5)dx, physical walls0/1, the same pressure regions and all four methods.
This supplements rather than replaces the original stress results. It performs
no clipping, uses the same wall-flux contract and has no accuracy reference.

{geometrytable}

Source: `configs/wall_geometry_control.json` and
`results/wall_geometry_control.csv`. These {len(geometry)} supplemental runs
include {findings['supplemental_wall_failures']} raw stage failures. They confirm
that the neural blast failures persist under physical wall alignment. Strong
contrast is complementary: NENO6 completes the declared cases while TENO5,
original NN and learned head fail; all methods have regime-dependent robustness.

### Validation and preservation

`logs/cpu_regression.log` records the complete CPU regression suite; legacy neural
construction/load tests are in `logs/legacy_smoke.log`. The initial new entropy
test incorrectly demanded exact mass quadrature from arbitrary point samples of
a discontinuity; it was corrected to symmetric moving-frame samples with the
same strict tolerance. The numerical implementation was unchanged. Initial logs
are preserved and documented, including a supplementary filename collision
fixed with unique sanitized cache keys and identity checks before using results.
The Lax contact diagnostic initially selected the stronger nearby shock inside
its broad window. Exact-wave-separation caps and a dedicated exact-Lax regression
test correct this; the primary L1 errors and percentage comparisons are unchanged.

Original metadata thresholds are compared at their exact float64 values against
float32 classifier outputs. All27 original-network cases were rerun after this
precision audit: no final-state value or success status changed. Initial Sod
timings were repeated without overlapping archival numerical jobs, with identical
final states. All first-pass outputs are retained in preflight subdirectories.
Actual solver-input Keras mask/probability parity is checked separately in
`results/raw/actual_input_parity.json` in addition to the initial random fixtures.

The general recommended baseline remains **classical TENO5**. Neither saved
learned-head specialization is supported as a universal upgrade: scalar gains
are case-specific, Euler adaptivity offers no measured improvement over fixed
mixing here, smooth accuracy regresses and stress failures remain. NENO6 is a
verified research comparator, with favorable and unfavorable cases of its own.
No model was retrained or promoted on these exposed tests.
'''
    results+=supplement
    (OUT/'RESULTS.md').write_text(results)
    nnb=improvements['nn_teno5']['best'];nnw=improvements['nn_teno5']['worst'];mix=max(mix_options)
    summary=f'''# Research interview summary

## A. Research evolution

**TENO5 → NENO6 inspiration → NN-TENO5 → NN-TENO5-V2 with a learned mixing head.**

TENO5 applies deterministic smoothness-based candidate selection. Li, Fu and
Adams' NENO6 demonstrates learned stencil selection with six inputs and four
candidate decisions. Our NN-TENO5 uses five inputs and three decisions, with
TENO5 candidate polynomials and a separately trained original classifier. V2
adds an independently parameterized learned coefficient blending classical and
neural reconstructions. Its backbone was also fine-tuned, so a main-model
comparison alone cannot identify the head's causal contribution.

## B. Main measured findings

{primary}

Evidence: **NEW MATCHED EVALUATION**, full precision in CSV/JSON.
Among the declared main exact-reference problems/grids, the largest original-NN
L1 improvement is {p(nnb[0])} on {nnb[1]} N={nnb[2]}; its worst is {p(nnw[0])}
on {nnw[1]} N={nnw[2]}. The largest V2-vs-original-NN difference on those cases
is {p(mix[0])} on {mix[1]} N={mix[2]}. These are extrema of this exposed suite,
not estimates of generalization. Inspect the same-backbone learned-vs-fixed
column below before crediting adaptivity rather than simply weak mixing.

{abtable}

{stabletable}

Runtime overhead is case-specific; see `RESULTS.md` for repeated median/IQR
and compilation costs. No universal speedup or positivity guarantee is supported.
Shu small gains require the numerical-reference sensitivity qualification.

At Sod N=128, warmed CPU medians in seconds are
TENO5={get('sod',128,'teno5')['runtime_median_s']:.6g},
NENO6={get('sod',128,'neno6')['runtime_median_s']:.6g},
original NN={get('sod',128,'nn_teno5')['runtime_median_s']:.6g}, and
learned-head V2={get('sod',128,'learned_head')['runtime_median_s']:.6g}.
The strongest same-grid exact-reference improvement among these saved methods
is NENO6's {p(improvements['neno6']['best'][0])} on
{improvements['neno6']['best'][1]} N={improvements['neno6']['best'][2]}.
This is a measured error reduction, not a universal efficiency advantage.

## C. Slide-ready material

Use the table above, keeping N/time/metric and evidence labels visible.
Three profile plots: `figures/01_sod_density_comparison.png`,
`figures/03_shu_osher_density_zoom.png`, and `figures/04_burgers_comparison.png`.
Use the lambda plot and same-backbone table to explain what the head adds.
If discussing Lax, use `figures/02_lax_density_comparison.png`.

Architecture diagram specification:

```mermaid
flowchart LR
  S[Five characteristic split-flux stencil values] --> N[Min-range normalization]
  N --> B[Classifier 5-64-32-16-8-3]
  B --> P[Three troubled probabilities]
  S --> T[Classical TENO5 reconstruction]
  P --> R[Hard neural candidate selection]
  S --> R
  P --> H[Separate rich head 8-8-1 sigmoid]
  N --> H
  T --> M[Shared interface flux blend]
  R --> M
  H --> M
  M --> D[Conservative flux difference and SSPRK3]
```

The rich head has81 parameters; the backbone has3155. Scalar run6 instead has
a3→1 head with4 parameters. Neither consumes the full proposed PDF feature vector.
For Euler, the same head is reused per characteristic field and per split-flux
direction, then projected to one conservative face flux; interface lambda plots
show the mean/range of those six scalar outputs.

Three conclusions: (1) verified saved methods yield case-dependent gains and
regressions; (2) fixed/learned mixing can temper an unstable backbone, while the
increment over a fixed blend must be measured separately; (3) conservation,
admissibility, accuracy and runtime are distinct measured properties.

Three limitations/open questions: (1) historical validation exposure and single
saved training seeds limit generalization claims; (2) raw strong-shock failures
and nonlinear solver consistency remain unresolved; (3) reference uncertainty
and neural runtime costs can outweigh small apparent gains. The expanded physics
vector is future work, not a feature of these measured checkpoints.

## D. Defensible oral explanations

**Why neural selection?** Deterministic thresholds are fixed; learned local
selection might better balance dissipation and resolution. Our tests decide
whether that benefit is realized; classification accuracy alone does not.

**What does NENO6 contribute?** The published six-point/four-candidate learned
selection design motivated this line of work. We located its authentic author
weights and verified reconstruction parity. Our Euler/Burgers results are matched
ports, not the publication's complete solver reproductions.

**How is NN-TENO5 different?** It has five values, three TENO5 candidates and
different training/normalization/weights. It is not NENO6 with another name.

**Why a separate head?** Probabilities describe candidate trouble; they need not
be a useful direct neural-intervention coefficient. A separate parameterized
head can learn that blending decision. Its actual benefit is the learned-vs-fixed
same-backbone measurement, not the size of its network.

**Does blending guarantee positivity?** No. A convex blend of reconstructed
fluxes alone does not prove density/pressure positivity of RK stage states.
We measure raw stages and retain failures without repairs.

**Reconstruction or solver improvement?** The underlying historical formulation
is held fixed in this package. The experimental finite-volume state-flux solver
is a different research contribution and its gains are not credited here.

**Consistent across PDEs?** Only the case-specific tables support conclusions.
Both favorable and unfavorable cases are shown; no universal improvement is
claimed. Scalar and Euler heads are different saved specializations.

**How expensive?** All methods run in one JAX CPU implementation. Compilation
is separated from three repeated synchronized warmed rollouts; the measured
median/IQR and cross-grid cost comparison are reported. No GPU speed claim follows.
'''
    summary+=f'''
## E. Essential negative findings to present

The scalar3→1 head improves pre-shock Burgers N=128 by
{p(gain(scalar_head['l1'],scalar_fixed['l1']))} over the same-backbone fixed blend.
The richer Euler head does not beat that fixed-blend control on Sod/Lax/Shu at
N=128 and its lambda is nearly constant. Scalar V2's smooth N=256 error ratio is
{smooth_ratio:.6g} relative to TENO5, with observed order about3 rather than5.
The pure scalar head backbone's finite advection blow-up is retained, and all
three neural methods fail the wall-aligned blast controls. NENO6 instead handles
the strong-contrast cases that defeat the other three. These outcomes support
case-specific investigation, not an unqualified upgrade claim.

The NENO6 paper's optimized background constraint is fourth order; six denotes
stencil width. Full neural convergence can degrade and is measured in Figure06.
Small Shu differences remain reference-sensitive. Use TENO5 as the general
research baseline while investigating which learned corrections are useful.
'''
    (OUT/'PRESENTATION_SUMMARY.md').write_text(summary)
    (OUT/'LIMITATIONS.md').write_text('''# Limits of the evidence

- NENO6 weights and scalar reconstruction are authentic and author-code parity
  is tested. The author repository does not supply an Euler/Burgers solver or
  training code; these results port its reconstruction into the common solver.
  They must not be represented as reproducing every published NENO6 result.
- Phase21 Euler and run6 scalar are different trained heads/backbones. The later
  dataset-V2 fixed blend is not a learned-head checkpoint. The original HDF5
  classifier is another backbone; use the explicit ablation for causal claims.
- The full proposed PDF physical input vector is not consumed by these artifacts.
  Historical head training uses soft selection, evaluated here with its authentic
  hard inference rule; Euler historical training used nonconservative repairs.
- Point-sampled historical Euler split flux and scalar state-flux/Rusanov paths
  are retained. This does not test the newer finite-volume state-flux correction
  or prove high-order nonlinear Euler accuracy.
- Canonical problems were historically exposed. The protocol is frozen for this
  comparison, but is not a new untouched holdout. No retraining/multi-seed
  variability or reliable independent-trajectory bootstrap interval is available.
- Raw failures are retained. No unconditional positivity, TVD or entropy-stability
  property follows from a convex flux blend. Blast is a stability stress case,
  with half-node outer wall faces for the historical point grid; no certified
  canonical cell-average blast accuracy reference is available.
- Shu reference uses one classical solver, a resolution ladder and dt check.
  Sensitivity estimates are not rigorous bounds; small error differences can be
  unresolved. Only final-time Shu error is available, not a converged time-error
  reference history. Exact-reference Sod/Burgers snapshots provide time controls.
- Archived original-network square-wave metrics have a small deterministic CPU
  reproduction discrepancy. Missing immutable original checkpoint/environment
  metadata prevent a definitive root-cause attribution. This evidence is not
  silently classified as successfully reproduced. The historical threshold=1
  all-keep probe discards saturated-one probabilities and is not unconditional.
- Runtime is CPU-specific, three warmed repeats, no core-affinity control. It
  measures common JAX ports, not the original frameworks. Failed endpoint error
  and complete-rollout runtime are absent, not replaced with finite guesses.
  Closest measured computational budgets may still differ materially.
- The earliest Euler checkpoint has nonfinite weights. Resume dictionaries are
  not inference checkpoint schemas. Missing legacy best-HDF5 references already
  absent from the previous baseline were not reconstructed or relabeled.
''')
    print(json.dumps(findings,indent=2))

if __name__=='__main__':main()
