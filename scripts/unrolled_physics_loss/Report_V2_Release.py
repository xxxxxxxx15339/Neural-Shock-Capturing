#!/usr/bin/env python3
"""Render the release research report from measured artifacts, without simulations."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.differentiable.artifacts import write_json
from src.differentiable.model_io import sha256_file

RESULTS=ROOT/'docs/research_release/results'


def read(path):return json.loads(path.read_text())


def write_derived(path,data):
    if path.exists():
        if read(path)!=data:raise FileExistsError('Derived diagnostics changed; choose a new report output directory')
    else:write_json(path,data)


def build(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    lines=['# NN-TENO5 V2 research release — measured findings','',
      'This report distinguishes existing checkpoint scores, full-validation policy selection, prospective final cases and experimental numerical variants. No network weights were changed. The original test split was not accessed for selection. All values are mathematical numerical experiments, not measured flow data.','',
      '## Full-validation correction study','',
      'All 615 scalar and 307 Euler validation trajectories were evaluated at six strengths with separate soft/hard candidate policies: **11,064 method–trajectory rollouts**. Every comparison uses the same case-specific physical time, 15 SSPRK3 steps, grid, boundary and reference.','',
      '| Physics | Policy | λ | Macro L1 ratio | 95% interval | Weighted ratio | Failures |','|---|---|---:|---:|---|---:|---:|']
    report=read(RESULTS/'full-validation/summary.json')
    for r in report['summary']:
        ci=r['macro_ci95'];lines.append(f'| {r["physics"]} | {r["mode"]} | {r["strength"]:g} | {r["macro_relative_l1"]:.6f} | {ci[0]:.6f}–{ci[1]:.6f} | {r["trajectory_weighted_relative_l1"]:.6f} | {r["failures"]} |')
    lines += ['',
      'Ratios exclude constant controls and family×resolution baseline errors ≤1e-10; absolute norms remain reported. Macro averages weight family×resolution equally. Weighted ratios divide trajectory mean errors. Bootstrap intervals use 2,000 paired stratified trajectory resamples, seed 20261020. Intervals condition on valid cases and do not correct for policy selection or quantify training-seed/reference uncertainty. Failed policies cannot be selected.','',
      '**Validation selection:** soft λ=.25 for both existing backbones. Scalar training used λ=.75; this inference calibration improves its full-validation score without retraining. Euler retains its training strength .25. Hard masks are not equivalent to the trained soft policy.','',
      '[Per-case measurements](results/full-validation/per_case.json), [bootstrap/group summary](results/full-validation/summary.json), [all-norm distributions and extrema](results/derived-validation/statistics.json), [per-case ratios](results/derived-validation/per_case_relative.json).','',
      '## Gate behavior and long-time prospective evaluation','']
    independent=RESULTS/'independent/benchmark.json'
    if independent.exists():
        rows=read(independent)['results']
        wall=RESULTS/'wall-closure/benchmark.json'
        if wall.exists():rows+=read(wall)['results']
        lines += ['The manifest declared 11 IVPs, each at two grids (22 configurations). Twenty original non-blast configurations completed. The original finest blast temporal reference was stopped for CPU cost. A separate closed-wall smoke with a 128/256/512 ladder and dt/2 exposed selected-model failure before t=.001. Full-time closed-wall blast results therefore assess stability only, with no accuracy reference or error score. Four Riemann IVPs are independently seeded; other entries are prospective canonical/stress configurations. The two grids of one IVP are not independent trajectories. The final suite did not retune any model or correction strength. The original 2048-cell blast temporal protocol remains incomplete.','',
          '| Case | N | TENO5 L1 | Selected soft L1 | Ratio | Valid raw stages | Reference ranking |','|---|---:|---:|---:|---:|---|---|']
        for r in rows:
            if r['method']!='v2-selected-soft':continue
            b=next(x for x in rows if x['case']==r['case'] and x['cells']==r['cells'] and x['method']=='teno5')
            fmt=lambda v:f'{v:.6g}' if isinstance(v,(float,int)) else 'unavailable'
            status='mirror-only wall invalid' if r['case']=='independent-woodward-colella' else 'resolved' if r.get('ranking_resolved') else 'unresolved / no reference'
            lines.append(f'| {r["case"]} | {r["cells"]} | {fmt(b.get("l1"))} | {fmt(r.get("l1"))} | {fmt(r.get("relative_l1"))} | {r["diagnostics"]["valid_for_accuracy_claim"]} | {status} |')
        lines += ['',
          'All six policies, checkpoints, profiles, sampled error-growth curves, gate activation/rejected candidates, soft/hard decisions and correction distributions are saved in [the benchmark artifacts](results/independent/). Historical learned blends use their own P0 checkpoints and policy names. They are comparisons on matched configurations, not the same method generation.','',
          'Advection and Euler Riemann references are analytic cell averages. Burgers post-shock/interaction and Shu–Osher use numerical references, a three-grid ladder and a finest-grid dt/2 comparison. Twice the sum of final spatial and temporal Cauchy changes is an uncertainty proxy, not a rigorous bound. A ranking is marked resolved only if the proxy is below 25% of the method margin and both methods have admissible raw stages. Full-time blast accuracy is unassessed after its failed neural smoke; no numerical reference is fabricated. Frozen P0 point-sampled references cannot replace these cell averages.','']
        lines += ['**Wall-boundary correction:** the original mirror-only blast configuration is retained as an unfinished reference study; a separate random-state wall audit exposed its defective neural wall contract. The soft network need not obey reflection/sign symmetry; random admissible wall stencils measured nonzero mass/energy flux. `reflective-wall` explicitly sets numerical wall mass/energy flux to zero while retaining wall momentum flux. Its dedicated regression verifies closed-wall conservation. The separately named closed-wall run freezes the same λ/model. Reference computations now save/cache each grid/timestep result immediately; after the failed neural smoke, the full-time run is explicitly stability-only. This is a correctness correction, not final-benchmark tuning. Admissible raw stages alone do not validate a physical boundary.','']
        gate_rows=[]
        for p in (RESULTS/'independent').glob('*/v2-selected-soft.npz'):
            a=np.load(p)
            if 'neural_gate' not in a:continue
            g=a['neural_gate'];c=a['neural_correction'];rejected=a['neural_rejected'];weights=a['neural_normalized_weights'];soft=a['neural_soft_keep'];hard=a['neural_hard_keep']
            pattern=np.sum(rejected.astype(int)*(2**np.arange(3)),axis=-1)
            gate_rows.append(dict(case=p.parent.name,gate_fraction=float(g.mean()),positive_correction_fraction=float(np.mean(c>0)),negative_correction_fraction=float(np.mean(c<0)),
                  mean_absolute_correction=float(np.mean(np.abs(c))),max_absolute_correction=float(np.max(np.abs(c))),
                  rejected_fraction_per_candidate=np.mean(rejected.reshape(-1,3),axis=0).tolist(),
                  rejection_pattern_counts=np.bincount(pattern.ravel(),minlength=8).tolist(),
                  soft_keep_quantiles=np.quantile(soft.reshape(-1,3),[.05,.5,.95],axis=0).tolist(),
                  effective_weight_quantiles=np.quantile(weights.reshape(-1,3),[.05,.5,.95],axis=0).tolist(),
                  gated_weight_quantiles=np.quantile(weights[g],[.05,.5,.95],axis=0).tolist() if g.any() else None,
                  hard_soft_mean_absolute_difference=float(np.mean(np.abs(soft-hard))),
                  activated_negligible_correction_fraction=float(np.mean((np.abs(c)<1e-12*np.maximum(np.abs(a['neural_base']),1.))[g])) if g.any() else None,
                  fallback_fraction=float(a['neural_fallback'].mean())))
        write_derived(output.parent/'gate_diagnostics.json',dict(results=gate_rows,
             interpretation='Time-sampled diagnostics across all interfaces/fields/signs of the selected numerical trajectory. Euler corrections are characteristic split-flux values. Correlation with a shock does not establish a causal error mechanism.'))
        lines += ['Observed gate/correction distributions are summarized in [gate_diagnostics.json](gate_diagnostics.json). Solution/error plots show both favorable and unfavorable comparisons. A lower final error alone does not establish that oscillations decreased; component extrema, overshoot/undershoot and TV are provided for that separate assessment.','',
          'For example, at 192-cell Sod the selected method lowers L1 error from .00265136 to .00253802, while density TV rises from .877466 to .879756 and density undershoot relative to the exact final range reaches .000350685 (TENO5: zero). At 192-cell long-time advection, error increases by 12.65% and TV increases from 2.400000 to 2.400435. These observations support a mixed accuracy/variation tradeoff, not a universal sharpening claim.','',
          'Gate activation is not synonymous with a substantial correction: in the time-sampled 192-cell advection trajectory, the gate activates on 56.76% of direction/interface observations, but 45.78% of active corrections are below 1e-12×max(|classical reconstruction|,1). In Sod these fractions are 37.31% and 62.87%. Potential sensitivity to small numerical variations is a hypothesis; these statistics alone do not isolate its cause.','']
    else:lines += ['**Incomplete:** the independent worker has not finished; per-case artifacts may exist, but no completed-suite result is claimed.','']
    lines += ['## Smooth spatial and temporal accuracy','']
    convergence=RESULTS/'temporal-cpu/convergence.json'
    if not convergence.exists():convergence=RESULTS/'convergence/convergence.json'
    if convergence.exists():
        r=read(convergence)
        lines += ['Spatial: 40/80/160/320/640 cells; exact cell averages; semidiscrete divergence isolates spatial error. Completed CPU rollout controls: 40/80/160 cells at common physical time .05, dt proportional to dx^(5/3), and dt/2 at every completed grid. Both checkpoints use selected λ=.25. **The original full temporal protocol through 640 was stopped for CPU cost and remains incomplete**; its exit 130 and progress log are preserved. This is a resource limit, not a numerical pass.','',
           '| Physics | Method | Profile | Last spatial L1 order | Last rollout L1 order |','|---|---|---|---:|---:|']
        for s in r['spatial']:
            match=[v for v in r['rollout'] if v['physics']==s['physics'] and v['method']==s['method'] and v['profile']==s['profile']]
            value=f'{match[0]["rows"][-1]["order_l1"]:.4f}' if match else 'not run'
            lines.append(f'| {s["physics"]} | {s["method"]} | {s["profile"]} | {s["rows"][-1]["order_l1"]:.4f} | {value} |')
        lines += ['',
          '[Completed CPU norm/order data and stage diagnostics](results/temporal-cpu/convergence.json), [selected spatial ladder](results/selected-spatial/convergence.json), [incomplete extended temporal status](results/convergence/status.json). Regular smooth spatial errors approach fifth order. Smooth critical-point profiles do not establish fifth order, and neural-only approaches third order. Interpret rollout rates with absolute errors and temporal Cauchy changes; no uncompleted fine-grid temporal result is claimed.','']
    else:lines += ['**Incomplete:** the expensive fine-grid neural Euler rollout study was stopped. The selected spatial ladder is complete through 640; reduced CPU temporal controls are separate. Completed print summaries are operational progress, not a substitute for finalized arrays and metadata. No completed full temporal study is claimed here.','']
    nonlinear=read(RESULTS/'nonlinear-euler/nonlinear_euler.json')
    lines += ['A broader nonlinear smooth Euler instantaneous test gives a different conclusion:','',
      '| Solver | Reconstruction | Last spatial L1 order |','|---|---|---:|']
    for r in nonlinear['results']:lines.append(f'| {r["solver"]} | {r["method"]} | {r["rows"][-1]["order_l1"]:.4f} |')
    lines += ['',
      'For nonlinear fluxes, F(cell-average U) differs from cell-average F(U) at O(dx²). The historical Euler split-flux path feeds the former into a flux reconstruction with cell-average coefficients, explaining the measured approximately second-order consistency on this general finite-volume test. The constant-pressure/velocity entropy wave is a special linear-in-density case.','',
      'An **optional experimental** characteristic-state reconstruction followed by a Rusanov flux approaches fifth order on this spatial diagnostic. It preserves the baseline and has dedicated consistency/conservation tests; its shock behavior, positivity and training remain unvalidated. No replacement checkpoint or general fifth-order neural claim is made. [Measured data](results/nonlinear-euler/nonlinear_euler.json).','',
      '## Conservation, admissibility and historical repair','',
      'All 11,064 validation rollouts have per-stage observations and conservative boundary budgets. The largest conservation residual across valid full-validation trajectories is approximately 1.42e-14. Selected λ=.25 Euler has minimum stage density 0.09545, minimum pressure 0.08357 and maximum effective CFL 0.33403. Scalar selected maximum effective CFL is 0.29781. These finite-corpus observations are not a general positivity or CFL stability theorem.','',
      'On the failing strong-correction Euler validation case, soft λ=.75 with the historical repair changes integrated mass by about 1.61115e-3 and energy by 8.88843e-4 (one repaired stage row). Hard λ=1 changes mass by 4.31769e-3 and energy by 2.98722e-3 (three rows). With repair increments included, balance residuals return to roundoff. Without them the physical balance error remains. At λ=.25 this case requires no repair. Original training intervention histories were not recorded at every update and cannot be reconstructed from sparse logs. [Repair evidence](results/repair/repair.json).','',
      '## Euler warm-up input mismatch','',
      'Training-data-only diagnostics compare four trajectories per family×grid. Warm-up characteristic states use initial snapshots; positive/negative solver flux inputs also use later exact snapshots. Mean normalized marginal Wasserstein distances are approximately .04549 (positive flux) and .02768 (negative flux). Teacher BCE rises from .67622 on warm-up states to .69319/.68924 on split fluxes; hard teacher disagreement rises from .43714 to .49922/.49790. These macro summaries weight family×grid equally.','',
      'The input mismatch is measurable after normalization. It does not establish that correcting it improves PDE outcomes: the sample times differ, inputs are exact-reference snapshots rather than neural trajectories, and teacher agreement is not numerical accuracy. The original warm-up remains available; `--pretraining-inputs split-flux` supplies an explicit future ablation. No retraining or causal improvement is claimed. [Diagnostic data](results/input-mismatch-r2/input_mismatch.json).','',
      '## Computational cost','']
    perf=RESULTS/'performance/performance.json'
    if perf.exists():
        r=read(perf);lines += ['CPU float64, one fixed IVP per physics, four grids, one warm-up plus nine synchronized repeats. Production kernels exclude diagnostic/export overhead; lowering, compilation and first execution are separate.','',
          '| Physics | N | Method | Warm median [s] | Compile [s] | L1 |','|---|---:|---|---:|---:|---:|']
        for row in r['results']:
            if row['method'] in ('teno5','gated-soft'):lines.append(f'| {row["physics"]} | {row["cells"]} | {row["method"]} | {row["warmed_median_seconds"]:.6g} | {row["compilation_seconds"]:.5g} | {row["l1"]:.6g} |')
        lines += ['',
          '[Timing repeats, per-step/cell/interface costs, temporary memory estimates and equal-budget/equal-accuracy comparisons](results/performance/performance.json). Hardware is recorded in [timing provenance](results/performance/provenance.json). Process RSS is a lifetime high-water mark; temporary storage is a compiler estimate. These measurements do not establish a general speedup or GPU performance.','']
    else:lines += ['**Incomplete:** isolated timing has not finished. No computational speedup is claimed. Reproduction commands are provided.','']
    lines += ['## Engineering and scientific release assessment','',
       'The final CPU regression suite passed **117 tests and 34 subtests**; the original baseline was 97 tests and 34 subtests. The separate Python 3.7 legacy network smoke passed **2 tests**. All 498 protected files matched the baseline inventory. The recorded environment passes pip check; a fresh V2 environment and GPU path were not independently installed/tested. CI is configured but has not run remotely.','',
       'A qualified public portfolio of **research implementation and rigorous investigation** is supported. A paper-level claim of universally superior, fifth-order, conservative positivity-preserving or faster NN-TENO5 is **not supported**. Outstanding work includes general Euler finite-volume shock validation, critical-point accuracy, stable conservative positivity treatment, controlled multi-seed training and warm-up ablations, broader/stronger reference studies and hardware portability.','',
       '[Acceptance matrix](ACCEPTANCE.md) · [reproduction commands](REPRODUCE.md) · [milestone record](MILESTONES.md) · [portfolio materials](../portfolio/LINKEDIN_POST.md). No push, merge or public publication was performed.','']
    output.write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'docs/research_release/RESULTS.md')
    args=p.parse_args();build(args.output)
