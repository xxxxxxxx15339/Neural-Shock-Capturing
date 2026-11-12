"""Assemble a measured report and verify protected artifacts. No PDE execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from experiments.accuracy_upgrade.study import OUT,ROOT,write,VARIANTS
from src.differentiable.v2_verify import verify


def load(p):return json.loads((OUT/p).read_text())
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in rows)+'\n'
def num(x):return '—' if x is None else f'{x:.6g}'
def emit(path,text):
    with (OUT/path).open('x') as f:f.write(text)


def run():
    spatial=load('results/spatial.json');euler=load('results/euler_actual_stages/summary.json')
    scalar=load('results/scalar/summary.json');train=load('checkpoints/paired_pilot/summary.json')
    hold=load('results/final_holdout/statistics.json');perf=load('results/performance.json')
    longtime=load('results/long_rollout/summary.json')
    # Reference arrays already generated; recompute comparisons against the
    # corrected SSPRK wrapper without overwriting the original reference study.
    shu=load('results/shu_reference/comparisons.json');refined=[]
    from src.differentiable.evaluation import errors
    for row in shu:
        n=row['cells'];v=row['variant'];ref=np.load(OUT/f'results/shu_reference/reference_N{n}.npz')['final']
        final=np.load(OUT/f'results/euler_actual_stages/shu_osher_{n}_{v}.npz')['final']
        base=np.load(OUT/f'results/euler_actual_stages/shu_osher_{n}_historical_teno.npz')['final']
        refined.append(dict(**{k:x for k,x in row.items() if k not in errors(final,ref)},**errors(final,ref),
                            corrected_ranking_resolved=row['uncertainty_proxy']<.25*abs(errors(final,ref)['l1']-errors(base,ref)['l1'])))
    if not (OUT/'results/shu_reference/comparisons_actual_stages.json').exists():
        write(OUT/'results/shu_reference/comparisons_actual_stages.json',refined)
    # Full inventory check, beyond the release's standard protected subset.
    manifest=load('baseline_manifest.json');failures=[]
    for name,expected in manifest['preserved_files'].items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:failures.append(name)
    preservation=dict(passed=not failures,files_checked=len(manifest['preserved_files']),failures=failures,
        release=verify(ROOT/'docs/research_release/results',protected_only=True),
        tracked_release=verify(ROOT/'docs/research_release/results',protected_only=True,tracked_only=True),
        existing_tracked_diff=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines())
    preservation['archived_missing_files_unchanged']=(preservation['release']['failures']==manifest['release_preservation']['failures'])
    write(OUT/'results/preservation_final.json',preservation)
    assert (preservation['passed'] and preservation['tracked_release']['passed']
            and preservation['archived_missing_files_unchanged'] and not preservation['existing_tracked_diff'])
    smooth_rows=[]
    for kind in ['density','momentum','pressure','coupled','critical']:
        rs=[next(r for r in spatial if r['profile']==kind and r['variant']==v and r['cells']==320) for v in ['historical_teno','fv_teno','fv_v2_soft25']]
        smooth_rows.append([kind]+[num(r['order']) for r in rs])
    convergence=table(['Profile','Historical TENO5 order','FV TENO5 order','FV saved V2 order'],smooth_rows)
    failures_table=table(['Variant','Failed configurations / 16','Min accepted density','Min accepted pressure','Max |mass budget|','Max |momentum budget|','Max |energy budget|'],[
        [v,sum(not r['success'] for r in euler if r['variant']==v)]+[
            num(min(r[key] for r in euler if r['variant']==v and r['success'])) for key in ['minimum_density','minimum_pressure']]+[
            num(max(abs(r['conservation_residual'][k]) for r in euler if r['variant']==v and r['success'])) for k in range(3)] for v in VARIANTS])
    error_table=table(['Case, N=128','Historical TENO5 L1','Historical V2 L1','FV TENO5 L1','FV V2 L1','Safe FV TENO5 L1','Safe FV V2 L1'],[
        [case]+[num(next(r for r in euler if r['case']==case and r['cells']==128 and r['variant']==v).get('l1')) for v in
         ['historical_teno','historical_v2_soft25','fv_teno','fv_v2_soft25','safe_fv_teno','safe_fv_v2_soft25']] for case in ['sod','lax','two_rarefactions','contrast','near_vacuum']])
    component_table=table(['Sod N=128 variant','Density L1','Momentum L1','Energy L1'],[[r['variant']]+[num(x) for x in r['component_l1']]
          for r in euler if r['case']=='sod' and r['cells']==128])
    train_table=table(['Seed','Variant','Matched-probe warm BCE','Warm classification accuracy','Warm validation L1','Final validation L1','Validation failures','Wall seconds, incl. shared compilation'],[
        [r['seed'],r['variant'],num(r['warm_classifier_bce']),num(r['warm_classifier_accuracy']),num(r['warmup_validation']['mean_l1']),num(r['final_validation']['mean_l1']),r['final_validation']['failures'],num(r['wall_seconds'])] for r in train])
    paired=[]
    for seed in [3101,3102]:
        a=next(r for r in train if r['seed']==seed and r['variant']=='A');b=next(r for r in train if r['seed']==seed and r['variant']=='B')
        paired.append(100*(b['final_validation']['mean_l1']/a['final_validation']['mean_l1']-1))
    hold_table=table(['N','Variant','Mean paired L1 ratio to historical TENO5','95% paired bootstrap interval','Worst trajectory ratio'],[
        [r['cells'],r['variant'],num(r['mean_ratio']),f'{r["ci95"][0]:.5f}–{r["ci95"][1]:.5f}',num(r['worst_ratio'])] for r in hold
        if r['baseline']=='historical_teno' and r['variant'] in ['fv_teno','historical_v2_soft25','fv_v2_soft25','safe_fv_v2_soft25']])
    scalar_table=table(['Case','N','TENO5 L1','Saved V2 / TENO5 L1','Historical learned / TENO5 L1'],[
        [case,n]+[num(next(r for r in scalar if r['case']==case and r['cells']==n and r['method']==m)[key]) for m,key in [('teno5','l1'),('v2_soft25','l1_ratio'),('p0_learned','l1_ratio')]]
        for case in ['sine','fourier','gaussian','square','burgers_smooth','burgers_hat'] for n in [64,128,256]])
    performance_table=table(['N','Variant','Warmed simulation s','Compile s','Seconds / step','L1'],[
        [r['cells'],r['variant'],num(r['warm_median_s']),num(r['compile_s']),num(r['warm_per_step_s']),num(r['l1'])] for r in perf['rows'] if r['cells']==128])
    long_table=table(['N','Frozen pilot','Sod t=.2 L1','Valid'],[[r['cells'],r['variant'],num(r.get('l1')),r['success']] for r in longtime])
    shu_table=table(['N','Variant','Density L1','Aggregate L1','Reference uncertainty proxy','Ranking resolved'],[
        [r['cells'],r['variant'],num(r['component_l1'][0]),num(r['l1']),num(r['uncertainty_proxy']),r['corrected_ranking_resolved']] for r in refined
        if r['variant'] in ['historical_teno','historical_v2_soft25','fv_teno','fv_v2_soft25']])
    report=f'''# NN-TENO5 accuracy upgrade: measured research report

## 1. Objective and decision

Improve accuracy, admissibility and generalization while preserving existing numerical methods and artifacts. **The finite-volume consistency correction is supported; a generally superior upgraded neural method is not established.** Keep the original default and checkpoints. Use the new state-flux path and conservative safeguard as explicitly selected research options.

The completed work covers the Euler consistency correction, conservative safeguards, paired input-matching pilots, scalar/Euler controls, a frozen six-trajectory confirmation set and isolated CPU cost measurements. A richer mixing head and full end-to-end retraining are **deferred**, not presented as completed improvements. Mixed paired training effects and substantial stress-case regressions do not justify increasing model complexity.

## 2. Baseline and verified limitations

Starting commit: `{manifest['git_commit']}`; branch `{manifest['branch']}`. CPU: {manifest['cpu']}; Python 3.11.15, JAX/jaxlib 0.10.2, NumPy 2.4.6, Optax .2.8, h5py 3.16.0. The baseline CPU suite passed 117 tests and 37 subtests. Forty-four established smoke results reproduce archived norms/validity to 1e-12. The separate legacy environment passed two tests. `pip check` passed. A fresh install and GPU execution were not performed.

The historical Euler FV inputs are inconsistent for general nonlinear fluxes; V2 state warm-up differs from split-flux inference; historical repairs can alter conservation; fixed/learned corrections have case-dependent error and stability. The PDF feature vector exists separately and is absent from the saved five-input architecture. The saved V2 checkpoints contain backbones with fixed mixing, while historical heads consume probabilities alone or probabilities plus normalized stencils. Exact weights-only historical training provenance cannot be fully reconstructed. See [audit and mathematical notes](AUDIT_AND_METHOD.md) and [baseline manifest](../baseline_manifest.json).

## 3. Implemented changes and mathematical justification

`src/differentiable/accuracy_upgrade.py` adds characteristic **state-average** reconstruction followed by a two-state Rusanov flux, canonical bidirectional state stencils, periodic/transmissive/reflecting-wall support, optional face admissibility scaling, a global LF reference flux, global shared-flux correction, per-FE-like-stage checks and bounded whole-step retries. Historical factories and source arithmetic are unchanged. No conservative average is clipped or repaired.

The consistency correction addresses `F(Ubar) != average(F(U))`. Conservative updates use one shared flux per interface. The fallback's ideal-gas LF convexity argument requires a valid wave-speed bound and CFL restriction; the implementation also verifies candidate states. A global segment coefficient uses convex density/pressure admissibility, not an assertion that arbitrary high-order mixing is positive. Accepted states and recomputed flux updates are checked. Rejected timesteps advance zero time and contribute zero exchange. See the audit notes for assumptions, boundary budgets, stop-gradient decisions and applicability of the cited FV/DG/FD literature. No general TVD, entropy stability, fifth-order-in-time or unconditional positivity theorem is claimed.

## 4. Protocol

Exact averages and exact face-flux differences isolate spatial error on N=40/80/160/320. Entropy-wave rollouts use N=40/80/160, t=.02, SSPRK3, dt≈.2 dx^(5/3), and dt/2 controls. Shock runs use N=64/128 and identical proposed timesteps across seven variants. Safeguarded retries, if needed, are reported separately. Sod/Lax/rarefaction/contrast/near-vacuum references are analytic Riemann averages. Blast is stability-only. Short-time Shu–Osher uses a 256/512/1024 reference ladder and finest dt/2; its empirical uncertainty is explicitly retained.

Paired A/B/C use seeds 3101/3102, architecture 5→64→32→16→8→3 (3155 parameters), matched 150 warm-up steps and 60 physics updates, K=15, Adam 2e-4 and gradient clipping at 1. A deliberately retains mismatched state warm-up; B uses canonical historical split-flux inputs; C uses canonical FV state inputs with face scaling. Training uses N=64 and full N=64/128 validation. This is smaller than saved V2's 300-update, three-grid budget. Pilot training uses explicit rejection instead of the original nonconservative guard; no rejection occurs. Fixed final checkpoints prevent post-hoc checkpoint selection. Training and validation trajectory IDs are disjoint; historical test datasets are never opened as datasets.

The fresh six-Riemann holdout was seeded and hashed before training. All candidate weights were frozen before its evaluation. It is confirmation only and does not select hyperparameters. Paired bootstrap intervals resample independent trajectories separately at each grid (2000 draws, seed 2026111202), condition on the fixed candidates and do not quantify training-seed or reference uncertainty. Six moderate Riemann IVPs are a narrow population.

Euler aggregate errors average unscaled conserved components with different units; component norms are also retained. Scalar families and grid resolutions are reported separately. No scalar/Euler combined headline statistic is used.

## 5. Smooth accuracy

Last spatial orders, N=160→320:

{convergence}

The coupled test recovers order ≈5 with both classical and saved V2 reconstruction. **This is a numerical-formulation result.** Density-only and pressure-only tests are special profiles and do not reveal the full historical limitation. Momentum-only and critical-point FV cases remain near fourth order. The always-retain-candidates diagnostic recovers approximately fifth order on those same profiles while the TENO selector rejects 34 and 78 candidate observations respectively at every tested grid. This controlled ablation implicates smooth-region candidate selection; no unvalidated selector modification is promoted.

The completed entropy-wave time-dependent ladder approaches fifth error reduction when dt shrinks as dx^(5/3). SSPRK3 itself remains third order in time. The dt/2 Cauchy controls and all absolute errors are in `results/temporal.json`. This special linear-in-density Euler solution is not a general nonlinear time-dependent convergence proof.

## 6. Euler accuracy, admissibility and conservation

Mean conserved-component L1; a dash means failed trajectory, not zero error:

{error_table}

Sod component errors:

{component_table}

Failures and conservative boundary-budget residuals across 16 configurations (eight cases × two grids). Blast smoke/full runs are correlated stress cases, not independent statistical samples:

{failures_table}

Both safeguarded FV variants complete all 16 tested configurations with admissible accepted states, including the full t=.038 reflecting-wall blast. This is finite-corpus evidence only. The historical classical path completes the blast at these settings; it fails the stronger contrast case. Unlimited FV and saved V2 paths fail eight configurations each. Their failures remain visible with first failed step, raw stage diagnostics and partial-time metadata.

At 128-cell two-rarefactions the safeguarded neural error is about twice the safeguarded classical error. The global safeguard can suppress reconstruction far from the troubled location; it is deliberately conservative and can be dissipative. It is retained as an option. In smooth entropy-wave runs it is inactive and reproduces the corresponding unsafeguarded errors to floating-point tolerance. Conservation residuals account for actual boundary fluxes; reflecting-wall momentum exchange is a wall-pressure force, while wall mass/energy fluxes are zero. Large absolute blast energy residuals must be interpreted against its large energy scale and many timesteps, not confused with exact arithmetic.

`results/euler_actual_stages/` is authoritative. `results/euler/` preserves an earlier, overly strict raw-wrapper diagnostic; the corrected wrapper is checked against original SSPRK3. Safeguarded kernels intentionally check FE-like updates more strictly than raw SSPRK convex stages. No repair is used to produce any accepted result.

Short-time Shu–Osher (t=.15, length 10):

{shu_table}

The uncertainty proxy is twice the sum of the final spatial and temporal Cauchy changes. It is not a rigorous error bound. No ranking is claimed when it exceeds 25% of the method margin. This short-time case is not evidence for canonical t=1.8 shock–acoustic accuracy.

## 7. Matched-input training and generalization

{train_table}

Matched B versus mismatched A changes final validation mean L1 by **{paired[0]:+.5f}%** and **{paired[1]:+.5f}%** for the two paired seeds. This inconsistent, small effect does not establish improved numerical generalization. Classification changes alone are insufficient. C has lower overall validation error, but it also changes the solver; the effect cannot be attributed solely to corrected pretraining. All six pilots have zero recorded nonfinite gradients, zero rejected physics updates and zero validation failures. Detailed losses, gradient norms, IDs, seeds, checkpoint hashes and budgets are saved beside separate warm/final checkpoints. Training times include compilation reuse and are not a fair A/B/C throughput comparison.

Frozen confirmation set, paired against historical classical TENO5:

{hold_table}

These rows isolate the classical numerical correction and the additional saved neural correction. The complete table also includes both seeds of every pilot and comparisons to **FV classical** TENO5. A/B effects remain small or unfavorable on confirmation; no holdout retuning is performed. Seed 3102 matched B is worse than paired A at both grids. The favorable saved V2 + FV Riemann results do not erase the rarefaction/stress regressions.

Longer frozen-checkpoint Sod evaluation at t=.2, with sampled analytic error curves:

{long_table}

These runs contain many more than 15 SSPRK3 steps. All shown trajectories complete. They do not test arbitrary long-time turbulent or shock-interaction behavior. See `results/long_rollout/` and figures for the full error curves.

## 8. Unchanged scalar controls

{scalar_table}

The scalar solver was not changed. Burgers smooth is pre-shock; Burgers top-hat includes a moving entropy shock before rarefaction–shock interaction. Its shock positions, speeds, widths, TV and oscillation metrics are saved. No generic post-interaction accuracy is claimed. Smooth advection/Fourier/Gaussian and discontinuous square cases include L1/L2/Linf, overshoot, undershoot, TV, runtime and conservation. Errors remain case- and resolution-dependent; saved neural policies are not universally better.

## 9. Computational cost

Isolated CPU Sod t=.15, N=128; nine synchronized warmed production repeats, without host diagnostic/export overhead:

{performance_table}

`results/performance.json` contains N=64/128/256, per-step costs, compilation/first execution, executable temporary-memory estimates, process lifetime RSS and discrete-grid equal-budget comparisons. It measures production kernels after other numerical jobs have finished. RSS is not a per-model allocation and compiler memory estimates are not GPU peak memory. Neural corrections can be slower enough that classical TENO5 at a finer measured grid wins under a fixed budget. Do not infer a speedup from lower same-grid error. Exploratory per-case runtime fields include host diagnostics and sometimes concurrent jobs; they are not the headline timing evidence.

## 10. Gates, negative findings and recommendation

- **A passed narrowly:** fifth-order coupled smooth spatial behavior is reproduced. Uniform fifth order, including critical points and general full-time Euler flow, is not established.
- **B retained experimentally:** conservative safeguards support the tested strong states and walls, but global corrections and neural mixing can be inaccurate and costly. They stay opt-in.
- **C did not establish a gain:** two paired input-matching pilots have opposite small effects. Preserve both results and checkpoints.
- **D deferred:** no physics-aware head or joint fine-tuning is justified by this pilot. Existing P0 learned-head rows have their own backbones and are not a controlled head-only comparison against V2.
- **E no global promotion:** keep existing defaults; offer FV state reconstruction as the consistency experiment, and the conservative safeguard when admissibility is required. Freeze the classifier architecture and prefer classical FV as the control for any further neural experiment.

The strongest contribution is isolating a solver consistency correction and replacing hidden repairs with an explicit conservative acceptance mechanism. A defensible next step is a local conservative correction that avoids global suppression, followed by trajectory-level validation and a larger multi-seed budget. Smooth critical-point selection deserves a separate ablation. Neither is implemented or claimed here.

## 11. Limits and unfinished work

No complete N=256/512 Euler shock sweep, canonical full-time Shu–Osher reference, generic coupled Euler full-time fifth-order proof, formal high-order positivity/entropy theorem, GPU study, full original-budget retraining, physics-aware head or exhaustive hyperparameter search is completed. Scalar interaction references and blast accuracy remain unassessed. The small holdout has only six moderate Riemann trajectories and should not support broad fluid-generalization claims. The architecture is unchanged; the fresh checkpoints are pilot artifacts, not recommended replacements.

All original checkpoints/data/historical outputs present in the frozen inventory are unchanged: **{preservation['files_checked']} files verified**, and all **{preservation['tracked_release']['protected_files_checked']} tracked release-protected artifacts** pass. Seven ignored historical `.best.h5` files from the old 498-file inventory were already absent at baseline. The full archived check still reports exactly those seven missing files; none is invented or silently restored. See `results/preservation_final.json`. The CPU regression run passed 130 tests and 37 subtests; two additional independent NumPy/wrapper tests passed separately, for 132 unique CPU tests. The separate legacy environment passed two tests. Initial failing development assertions and their corrections are preserved in logs rather than erased.

## 12. Reproduction and artifacts

See [README](../README.md) for commands and [audit notes](AUDIT_AND_METHOD.md) for mathematical assumptions and primary references. Every numerical command requires a fresh output path. Use the recorded environment freeze; installing it freshly is unverified. `baseline_manifest.json` records provenance, hashes and blockers. `artifact_hashes.json` verifies the new deliverables. Reports and all plot curves derive only from recorded solver outputs.
'''
    emit('reports/FINAL_RESEARCH_REPORT.md',report)
    summary=f'''# Research interview summary

- **Problem:** NN-TENO5 gains were inconsistent and some Euler corrections produced invalid density/pressure. Historical repairs could violate conservation.
- **Previous method:** five-input candidate classifier plus classical/neural mixing; saved V2 warm-up used characteristic states, inference used characteristic split fluxes.
- **Numerical bottleneck:** reconstructing F(Ubar) as a cell-average flux limits general nonlinear Euler FV consistency. Coupled smooth order was about 2.03.
- **New method:** isolated characteristic-state reconstruction plus Rusanov flux; shared-interface LF flux correction, physical face scaling and bounded stage-aware retries. Original defaults and checkpoints are preserved.
- **What worked:** coupled spatial order ≈5.00; conservative safeguards complete all 16 tested stress configurations; favorable saved V2 + FV results on six frozen moderate Riemann trajectories.
- **What failed:** unlimited FV fails strong cases; safeguarded neural rarefaction error is about twice classical; critical-point/momentum-only order remains near four. An always-retain ablation recovers fifth order but is not a shock method.
- **Training result:** paired matched-input pilots change validation L1 by {paired[0]:+.5f}% and {paired[1]:+.5f}%; no consistent gain. Solver improvements must not be credited to training.
- **Cost:** synchronized CPU timings and equal-budget comparisons are recorded; lower same-grid error does not imply computational advantage.
- **Validation:** 132 unique CPU tests plus 37 subtests and two legacy tests pass; baseline predictions and protected artifacts are checked. No GPU or fresh-install claim.
- **Decision:** retain experimental options and the existing backbone/defaults. Defer richer mixing heads and full retraining. No universally superior neural solver is claimed.
- **Next work:** local conservative limiting, critical-point selection, full-budget multi-seed trajectory training and broader independent references.

Exact tables, scopes, uncertainty and reproduction commands are in `FINAL_RESEARCH_REPORT.md`. No unmeasured benchmark is presented as completed.
'''
    emit('reports/PRESENTATION_SUMMARY.md',summary)

if __name__=='__main__':run()
