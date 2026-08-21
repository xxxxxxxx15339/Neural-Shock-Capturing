# Implementation Guide — the Unrolled Physics Loss proposal (WENO/TENO-NN)

This is a step-by-step build plan for the Unrolled Physics Loss proposal, put forward by
Radouan Boukharfane and implemented/tested here by Yasser Baouzil. It follows the
same four ingredients the PDF lays out (input vector, dataset, unrolled physics loss,
architecture/validation) but **reorders them by real dependency**, not by the PDF's
numbering. The PDF's Étape 3 (differentiable rollout) is actually the load-bearing
prerequisite for almost everything else — you cannot meaningfully build Étape 1's feature
vector or Étape 2's dataset until you know whether a differentiable solver is feasible in
this codebase. Building in the PDF's literal order means throwing away Étape 1/2 work if
Étape 3 turns out to be a multi-week rewrite (it is — see Phase 1 below).

Each phase lists: what to build, which existing repo files to reuse or port, and a
concrete exit criterion so you know when to move on. Phases 0 and 1 are the highest-risk,
highest-value work — do not skip the spike in Phase 0.

---

## Phase 0 — Feasibility spike (1–2 days, do this first)

> **Status: done.** `DiffSpike_TENO5_Advection.py` (`scripts/unrolled_physics_loss/`) runs 6
> checks: JAX-vs-NumPy regression on a single reconstruction call and on a full 39-step
> rollout (both match to float64 precision), a finite/nonzero gradient through a
> 117-substep rollout on a smooth profile, the same check on a discontinuous profile
> (found and fixed a real NaN-gradient bug along the way — see Phase 1 write-up), one
> gradient step reducing the loss, and a 50-step training loop converging. All 6 pass.
> Run with
> `conda run -n teno-jax python scripts/unrolled_physics_loss/DiffSpike_TENO5_Advection.py`.

**Goal:** answer "can I even backpropagate through a TENO5 reconstruction + RK3 step?"
before committing to a rewrite. This is the single biggest unknown in the whole plan.

1. Pick **JAX** as the differentiation framework (see box below for why).
2. Take `teno5_candidate_values`, `teno5_smoothness_indicators`, and
   `teno5_selection_values` from `src/schemes/teno5.py` and re-implement just the *smooth
   (deterministic) path* — no NN yet — using `jax.numpy` instead of `numpy`. This is a
   near-mechanical port: the functions are already pure array math with no Python control
   flow that depends on data values (the mask logic uses `np.where`-style vectorized
   comparisons, which map directly to `jnp.where`).
3. Wire that into one `SSPRK3` step (`src/core/TimeSteppingMethods.py`) using `jax.numpy`
   arrays.
4. Take the gradient of a scalar loss (e.g. `L2` error against an exact Sod snapshot)
   **through 5–10 unrolled steps** using `jax.grad`. Confirm: (a) it runs without NaNs,
   (b) `jax.jit` compiles it in reasonable time, (c) memory doesn't explode as K grows.

**Exit criterion:** you have a runnable script that computes `∂L/∂(some scalar in the
reconstruction)` through a multi-step rollout. If this fails or is unworkably slow/memory-
hungry, that's the answer to build the rest of the plan around (e.g. drop K to 1–2 steps,
or fall back to a non-unrolled loss) — find that out in 2 days, not after 3 weeks of
Phase 1–4 work.

> **Why JAX over TF/PyTorch here:** the entire numerics stack (`src/schemes/*.py`,
> `src/core/*.py`) is written as pure NumPy functions on arrays — no classes with mutable
> state, no Python-level branching on data. That style ports to `jax.numpy` almost
> line-for-line (`np.` → `jnp.`, `np.where` stays `jnp.where`). PyTorch would require
> wrapping everything in tensors and rewriting the functional style into
> tensor-op-with-autograd idioms; TensorFlow/Keras is a non-starter because this repo
> pins **standalone Keras 2.2.4** (`src/networks/TenoNetworks.py` uses `keras.optimizers`,
> not `tf.keras`), which has no autodiff-through-a-custom-solver story at all. A compact
> 3-layer/32-neuron MLP (the Unrolled Physics Loss proposal Étape 4) is about 15 lines of `jax` code — you don't
> need Flax/Haiku for something this small.

---

## Phase 1 — Differentiable core solver (1–3 weeks, gated on Phase 0 passing)

> **Status: scalar path done; Euler path not started.** See "What was actually built" below
> for the full account, including a real bug the discontinuous-profile test caught (a
> NaN-gradient trap in a `jnp.where` normalization, not in the TENO5 math itself) and the
> design choices made along the way. The Euler port (`eulerEquations_jax.py`,
> `lax.scan`-based stepping, positivity guards) is still open — scoped out of this pass
> because the scalar spike already answers Phase 0's question, and Euler adds a second,
> separable risk (Roe eigenbasis differentiability) that doesn't need to block dataset/
> feature-vector work (Phase 2/3) starting on the scalar side.

Port the full deterministic path to JAX, keeping the existing NumPy version untouched as
a correctness oracle.

- **New module**: `src/schemes/teno5_jax.py` mirroring `teno5.py`'s
  `teno5_candidate_values`, `teno5_smoothness_indicators`, `teno5_selection_values`,
  `teno5_reconstruct`, `TENO5`/`TENO5euler` (`src/schemes/teno5.py:251,352,536,737,769`).
- **New module**: `src/core/eulerEquations_jax.py` porting `flux`, `roe_eigenbasis`,
  `getEulerFlux` (`src/core/eulerEquations.py:16,55,79`) — needed for the Euler test
  cases in the validation matrix (Sod/Lax/Sedov).
- **New module**: `src/core/TimeSteppingMethods_jax.py` porting `SSPRK3`
  (`src/core/TimeSteppingMethods.py:29`) — this is the RK3-TVD integrator the Unrolled Physics Loss proposal's
  Équation 4 calls for; it already exists in this repo, it just needs a JAX-traceable
  step loop (replace the Python `for` over timesteps with `jax.lax.scan` so `jax.grad`
  can differentiate through the whole rollout without unrolling in Python).
- **Regression test**: for every ported function, assert `jnp` output matches the
  existing NumPy version to float64 tolerance on the same random stencils (`tests/`
  already has `test_teno.py` — extend it, don't replace it).
- **Positivity/CFL guards**: `eulerSimulation._validate_state`
  (`src/core/SimulationClasses.py:65`) currently raises `FloatingPointError` on
  non-finite/non-positive state. `jax.lax.scan` bodies can't raise Python exceptions
  mid-trace — replace this with a differentiable penalty (feeds directly into
  `L_Reg`/`L_TV` in Phase 5) or a `jnp.where`-based clamp, not an exception.

**Exit criterion:** a JAX Sod-shock-tube run at N=100 over T matching the existing NumPy
`eulerSimulation.runEuler()` output to numerical tolerance, differentiable end-to-end
through `jax.grad`. **(Not yet met — scalar exit criterion below was substituted for this
pass; Euler remains open.)**

### What was actually built (scalar path)

New files, all under a new `src/differentiable/` package (not `src/schemes/`/`src/core/`
directly — see "Design choice: a separate package" below):

- `src/differentiable/teno5_jax.py` — JAX port of `teno5_candidate_values`,
  `teno5_smoothness_indicators`, `teno5_selection_values` (the log-domain softmax `chi_k`),
  and `combine_teno_candidates_jax`. Two reconstruction entry points:
  `teno5_reconstruct_jax` (hard `chi >= cutoff` cutoff, byte-for-byte match to
  `teno5.py`, used only for regression testing) and `teno5_soft_reconstruct_jax` (takes an
  externally supplied `keep_weight in [0,1]` instead of computing one from the cutoff —
  this is the one later phases hook a network into).
- `src/differentiable/numerics_jax.py` — JAX port of `FiniteVolumeMethod.partU` (periodic
  stencil gather), `FluxSplittingMethod.flux` + `LaxFriedrichs` (flux splitting), and
  `SSPRK3`'s Shu-Osher update, plus a `rollout(u0, flux_fn, dt, dx, steps)` driver.
- `src/differentiable/teno5_jax_toy.py` — a small hand-rolled JAX MLP (not the real
  Keras NN-TENO5 architecture) with the same input/output contract, used only to prove
  gradients reach network-shaped parameters through the solver.
- `tests/test_teno5_jax_regression.py` — the Phase 1 regression test, runnable with
  `conda run -n teno-jax python -m pytest tests/test_teno5_jax_regression.py -q`.
- `DiffSpike_TENO5_Advection.py` — the Phase 0 spike script (see status note above).
- `requirements-jax.txt` + a new `teno-jax` conda env — see "Design choice: environment
  isolation" below.

**Regression result:** single-call and full-39-step-rollout outputs match the existing
NumPy `Simulation.run()` to `4.4e-16` / `0.0` max absolute error (i.e. floating-point-exact
after fixing a precision bug — see below).

**A real bug the discontinuous-profile check caught:** the first version of
`teno5_jax_toy.py`'s stencil normalization used
`jnp.where(value_range > 0, (stencil - minimum) / value_range, 0.0)`. On a smooth sine
profile this worked and gave a finite, if tiny (~1e-9), gradient. On a square-wave profile
it produced **NaN gradients in every layer**. Cause: `jax.grad` evaluates the VJP of both
`jnp.where` branches before selecting one — there's no lazy short-circuit — so on a flat
region of the stencil (`value_range == 0`, which a piecewise-constant profile produces
constantly) the untaken `(stencil - minimum) / value_range` branch has an infinite local
gradient, and `0 * inf = NaN` poisons the entire backward pass through every downstream
layer, not just that one op. Fixed by guarding the *denominator*, not the *output*:
`safe_value_range = jnp.where(value_range > 0, value_range, 1.0)` then dividing directly —
when `value_range == 0` the numerator is also exactly 0, so the substituted 1.0 denominator
already gives the right forward value with no separate select needed, and there's no
untaken branch left to poison the gradient. This is the standard "double where" pattern for
avoiding NaN gradients in JAX, and it's the reason Phase 0's spike script tests a
discontinuous profile at all, not just the smooth one the Unrolled Physics Loss proposal's own example (Sod) would
have exercised only near the shock — a smooth-only test would have shipped this bug
silently, since it does not corrupt anything on smooth data.

### Design choice: a separate `src/differentiable/` package

The guide originally proposed `src/schemes/teno5_jax.py` and
`src/core/eulerEquations_jax.py` (i.e., dropping the JAX ports next to the NumPy originals
they mirror). That broke immediately: `src/networks/__init__.py` does
`from .WenoNetworks import ...` at import time, and `WenoNetworks.py` does
`from keras import backend as K` — so importing anything under `src/networks/`, even a
brand new unrelated file placed there, pulls in standalone Keras 2.2.4, which cannot be
installed in the same environment as a modern JAX (conflicting NumPy pins: Keras 2.2.4
wants `numpy==1.18.5`, JAX in 2026 wants something far newer). `src/schemes/__init__.py`
turned out not to have this problem directly, but relying on that felt fragile.

The fix actually taken: every JAX port lives in a brand new top-level package,
`src/differentiable/`, that the legacy Keras-based packages never import from and that
never imports the legacy packages except the pure-NumPy, Keras-free modules needed for
regression testing (`src/schemes/teno5.py`, `src/core/SimulationClasses.py`, etc., which
have no Keras dependency themselves). This means: (a) the JAX code runs in a plain
`jax + numpy` environment with zero Keras-related install pain, (b) the existing
Keras-based training/eval scripts are completely untouched and still run exactly as before
(re-verified: `tests/test_teno.py` and `tests/test_core.py` pass unchanged in `base`,
`tests/test_networks_smoke.py` passes unchanged in `weno-nn`).

### Design choice: environment isolation (`teno-jax` conda env)

This repo already has three conda envs: `weno-nn` (the pinned legacy Keras 2.2.4 / TF
1.15 / NumPy 1.18.5 stack used for real training), `zero-to-hero` (numpy/matplotlib only,
used for the course-document plots), and `base`. None of them has JAX, and `weno-nn`
*cannot* have JAX added to it without breaking its pin (JAX needs a NumPy far newer than
1.18.5). Rather than force a version conflict or install JAX into `base` (shared with
whatever else runs there), a fourth env, `teno-jax` (Python 3.11, `jax`+`numpy`+
`matplotlib`+`pytest`, no Keras), was created specifically for this package — documented
in `requirements-jax.txt`. This mirrors the repo's own existing pattern of splitting
`requirements.txt` (legacy Keras stack) from `requirements-numerical.txt` (plain
numpy/matplotlib work) — one more environment split for one more dependency conflict,
not a new pattern.

### Design choice: unrolled Python loop over `jax.lax.scan`, for now

The guide originally called for `jax.lax.scan` in the timestepping port, for compile-time
and memory reasons at large `K`. The spike instead unrolls the SSPRK3 substep loop as
three explicit lines (it's always exactly 3 substeps with fixed coefficients — see the
docstring in `numerics_jax.py` for the derivation) and unrolls the outer K-step rollout as
a plain Python `for` loop (`rollout()` in `numerics_jax.py`). This was sufficient to answer
Phase 0/1's question — regression matches exactly, gradients are finite, training loss
decreases — at the K used here (39 steps × 3 substeps = 117). It was not fast: the 50-step
training-loop demo (Part 4 of the spike script) took several minutes un-jitted, because
JAX's eager-mode per-op dispatch overhead is significant and this rollout issues on the
order of 117 × ~10 ops × 50 iterations of it. `jax.jit` around the whole rollout, and
`jax.lax.scan` if `jit` alone isn't enough, are the next things to reach for once Phase 6
needs to actually train at scale — deferred here rather than built speculatively, per the
guide's own Phase 0 instruction to find out what's *needed* before building further.

---

## Phase 2 — Feature vector expansion (Étape 1, 3–5 days)

> **Status: done, with two documented spec gaps.** `src/data/feature_vector.py` builds
> `u_normalized`, `scheme_linear_weights`, `eta`, `j_right` (and `j_left`/Mach/pressure
> where the input allows it — see below). Validated on the real 75,000-row oracle dataset
> (`data/teno5ExactStencils.csv`): no NaNs, `u_normalized` matches the existing
> normalization exactly, all 11 tests in `tests/test_feature_vector.py` pass. Two things in
> the Unrolled Physics Loss proposal's own Équation 3 turned out to be underspecified enough that guessing silently
> would have been worse than flagging them — both are spelled out in the module's
> docstring, summarized here:
>
> 1. **`c̃_WENO/TENO` has no formula in the PDF, only a name.** Implemented as the
>    scheme's fixed `LINEAR_WEIGHTS` triple broadcast per sample (the only
>    data-independent "scheme identity" quantity not already covered by `η_k`). If the
>    professor meant something else (e.g. the deterministic scheme's own keep-decision on
>    this stencil), that's a different column — worth a 30-second clarifying question
>    before Phase 3 generates a full dataset around the wrong interpretation.
> 2. **Computing both `J_{j-1/2}` and `J_{j+1/2}` needs 6 raw stencil values, not 5.**
>    `teno5_reconstruct` always returns the *right*-face value of whatever window it's
>    given — that's the whole solver's convention, not a bug — so getting the *left* face
>    for the same cell needs the window shifted one cell left, i.e. `[j-3..j+2]` instead of
>    `[j-2..j+2]`. The existing oracle dataset only stores 5 values per sample (checked
>    directly against `data/teno5ExactStencils.csv`'s actual shape), so
>    `scalar_feature_vector` degrades gracefully: given 5 values it returns only
>    `j_right`; given 6 it returns both. Widening a *new* dataset generator's window by one
>    point is a small change (Phase 3, not here) — the synthetic samplers in
>    `GenerateTrainingData_TENOExact.py` already compute cell averages from a closed-form
>    profile parameterized by cell center, so one more stencil offset is enough.

Build the Unrolled Physics Loss proposal's 9-value `X_j` (Équation 3) as a preprocessing function, not inside the
network. This is additive on top of what `GenerateTrainingData_TENOExact.py` already
does.

| Term in `X_j` | Formula | Source in repo |
|---|---|---|
| `ū_s` | local min-max stencil normalization | already implemented — `normalize_teno_stencils` in `src/schemes/teno_common.py:4` |
| `c̃_WENO/TENO` | linear weight identity of the scheme | `LINEAR_WEIGHTS` constant, `src/schemes/teno5.py:52` |
| `η_0, η_1, η_2` | `τ5 / (β_k + ε)` | `β_k` from `teno5_smoothness_indicators` (`teno5.py:251`); `τ5 = |β_0 - β_2|` is one extra line |
| `J_{j±1/2}` | `|u^L - u^R|` interface jump | `teno5_reconstruct` (`teno5.py:536`) already computes left/right reconstructed interface values — take their difference |
| `M_j` | local Mach number, Euler only | needs `velocity/soundspeed` from `src/core/eulerEquations.py` primitive-variable helpers |
| `|ΔP|/P` | relative pressure gradient, Euler only | one finite difference on the pressure array already computed in `eulerEquations.py:flux` |

Write this as `src/data/feature_vector.py` with a scalar-PDE version (7 features: no
`M_j`, no `ΔP/P`) and a Euler version (all 9). Keep the classifier output unchanged
(3 troubled-probabilities, sigmoid, BCE) — the Unrolled Physics Loss proposal doesn't propose changing what the
network predicts, only what it sees.

> **Resolved: one shared network, not two, with an explicit `is_euler` flag.** Scalar and
> Euler samples having different feature counts doesn't need two architectures — pad the
> two Euler-only features (`M_j`, `|ΔP|/P`) to a fixed value for scalar samples (standard
> null-feature padding), and add a 10th feature, `is_euler` (0 or 1). The flag matters:
> without it, `M_j = 0` is ambiguous between "genuinely zero Mach number" (a real, valid
> Euler value — e.g. a stationary contact discontinuity) and "not applicable, this is a
> scalar sample" — the network has no way to tell those apart from the padded value alone.
> One shared network is also the better choice on its own merits here, not just the
> simpler one: the core of `X_j` (stencil, `η_k`, BVD jump) means the same thing regardless
> of physics, so training jointly gives the network more signal for the part of the task
> (candidate selection) that's actually shared, rather than splitting a already-small
> (3-layer/32-neuron) model's data across two smaller problems.

**Exit criterion:** `feature_vector.py` run over the existing
`data/teno5ExactStencils.csv` reproduces the 5-value stencil in columns 0–4 and adds the
4 new engineered columns, with no NaNs across the full oracle dataset (75,000 rows). **Met**
— see status note above (the dataset only supports the `j_right` half of the BVD pair, by
design, not by oversight).

---

## Phase 3 — Dataset: 40% synthetic / 60% PDE-projected (Étape 2, 1–2 weeks)

> **Status: audit run, go/no-go answered — the 60% PDE-projected bucket needs a real
> architectural fix, not just better labeling.** `PDEDataset_NonIdentifiability_Audit.py`
> reused this repo's own existing real-PDE data pipeline
> (`GenerateTrainingData_EulerExact.py`'s Riemann + Shu-Osher-shock-interaction
> categories — no new simulation code needed) to generate ~149,000 samples from 420
> trajectories, and measured — not estimated — how often near-identical local stencils
> from different trajectories need different correct answers:
>
> | Quantization | Value-regression conflict rate | Classification-label conflict rate |
> |---|---|---|
> | coarse (1 decimal) | 79.9% | 78.1% (barely improved) |
> | fine (2 decimals) | 51.9% | 37.1% (meaningfully improved) |
>
> The mitigation this guide proposed (oracle keep/discard labels instead of raw value
> regression) **does help** — real, measured, not assumed — cutting the conflict rate by
> roughly a third at the finer quantization. But it does not come close to solving the
> problem: even after the fix, more than a third of near-duplicate stencils in this
> dataset still demand different correct labels depending on which trajectory they came
> from. This confirms, with numbers instead of a citation to the earlier WENO7-NN
> failure, that **the fix has to be architectural** (see point 3 below, and Phase 5's
> unrolled-loss framing) — no amount of relabeling the current 5-point-stencil
> representation will resolve information that genuinely isn't in the stencil.
> **This is worth raising with the professor directly: the Unrolled Physics Loss proposal's plan, as written, does
> not have an answer for this, and "60% real-PDE data" is a substantial fraction of the
> proposed training set to build on a representation with a measured >1/3 conflict
> rate.**

**The 40% synthetic slice is mostly done.** `GenerateTrainingData_TENOExact.py` already
generates sine, double-sine, polynomial, step, kink families with closed-form cell
averages and oracle labels — this *is* the Unrolled Physics Loss proposal's "profils analytiques" bucket. Re-weight
the family mix to match the Unrolled Physics Loss proposal's description (steep discontinuities, sine waves,
smoothed pulses, Fourier–Chebyshev combinations) if you want a closer match; the
generation machinery doesn't need to change.

**The 60% PDE-projected slice is new, and it's the part with a known failure mode —
read this before building it.** This project already tried "train on real high-resolution
PDE solutions projected to a coarse grid" once, for WENO7-NN (`Results.md` Phase 2,
referenced in the earlier oracle-dataset work): stencils that look nearly identical at
the local level required genuinely different correct answers depending on **upstream
wave history the network never sees** (non-identifiability). the Unrolled Physics Loss proposal's 60% PDE bucket
(advection, Shu-Osher, Sod/Lax/Woodward-Colella at N≥4096 projected to N∈[100,400]) is
structurally the same setup.

Mitigations to build in, not optional polish:
1. **Label with the oracle procedure, not direct value regression.** Reuse the
   exhaustive-mask oracle labeling from `GenerateTrainingData_TENOExact.py` (keep/discard
   per candidate, not "predict the exact flux") — classification labels are far less
   sensitive to the non-identifiability problem than direct regression targets, because
   many different discard patterns can be equally "not wrong."
2. **Track and report a duplicate/near-duplicate stencil audit** before training: bin
   stencils by shape, check how often near-identical inputs get conflicting oracle labels
   across different PDE runs. This turns the past failure into a measured go/no-go gate
   instead of a repeat surprise.
3. If the audit shows meaningful conflict, don't try to fix it with more data — that's
   what failed last time. The fix is architectural (e.g. add a short trailing-history
   feature, or accept the unrolled loss as the mechanism that resolves ambiguity via
   trajectory-level feedback rather than pointwise supervision — see Phase 5).

> **Resolved, not just proposed: don't oracle-label the 60% PDE bucket at all — use it
> only as `(u0, exact_final)` rollout supervision for Phase 5's unrolled loss.** Point 3
> above named this as a possible fix; Phase 6 then built and validated it without knowing
> it was answering this question. The scalar training run that beat TENO5 on 2/3 held-out
> cases (Results.md Phase 10) used *zero* oracle labels — just random Riemann-style
> `u0`/exact-solution pairs, fed straight into the unrolled loss. This isn't a coincidence
> that happens to sidestep the problem; it's why it works. Pointwise oracle labeling forces
> a function `stencil_shape -> one label`, which is provably impossible to satisfy when the
> true answer depends on context outside the stencil (two different trajectories, same
> local shape, different correct labels — exactly what the audit measured). Training via
> the unrolled loss instead asks for weights that produce a good *aggregate* trajectory
> outcome per training example; it never needs to resolve which of two conflicting labels
> is "the" answer for a given shape, because each example supplies its own
> self-consistent target end-to-end. The exhaustive-mask oracle-labeling machinery
> (`GenerateTrainingData_TENOExact.py`'s `oracle_labels`) stays exactly as useful as before
> for the 40% synthetic bucket, where per-stencil answers genuinely are well-posed (no
> upstream history) — it's specifically the 60% PDE-projected bucket that should skip
> labeling and go straight to rollout supervision.

**Exit criterion:** a combined dataset (feature vectors from Phase 2, oracle labels) with
the 40/60 split, plus a written conflict-rate number from the audit in step 2 — even if
that number is bad, it's the answer the Unrolled Physics Loss proposal's plan needs before scaling up data
generation further. **Met** — the number is bad (see status note above), and now it's a
number, not a hunch.

---

## Phase 4 — Projection layer (Étape 4, 2–3 days)

> **Status: done.** `src/differentiable/projection_jax.py`'s `project_keep_weight`
> (`mode='soft'` for training, `mode='hard'` for inference). Turned out the actual
> consistency guarantee was already built in Phase 1 — `combine_teno_candidates_jax`
> already falls back to the stencil's center value whenever every candidate's weight is
> zero, for hard or soft `keep_weight` alike (see that function's docstring). This module
> doesn't add new math on top of it; it gives "network output -> keep_weight" one
> explicit, tested definition instead of leaving it as ad hoc `1 - sigmoid(...)` inline at
> each call site (which is what the Phase 0 toy network did). `tests/test_projection_jax.py`
> passes the exit criterion below, plus an explicit all-troubled-falls-back-to-center check.

the Unrolled Physics Loss proposal calls for a "hard constraint layer guaranteeing reconstruction consistency" but
doesn't specify what constraint. Concretely, for NN-TENO5 the only thing that needs
guaranteeing is: **the candidate weights the network effectively selects must stay a
valid convex combination consistent with the TENO fallback structure** (i.e. never all
three candidates discarded with no fallback, matching the existing
`combine_teno_candidates` fallback behavior in `src/schemes/teno_common.py`). Implement
this as a final layer that:
- takes the 3 raw sigmoid outputs,
- applies the same fallback-if-all-discarded rule `combine_teno_candidates` already
  enforces deterministically,
- is differentiable (soft version of the discard mask during training, hard threshold at
  inference — this mirrors how the classifier already works, just makes the fallback
  path visible to the gradient).

**Exit criterion:** projection layer output is a strict no-op relative to current
`NN-TENO5` inference behavior when thresholded at 0.5 — verify against
`TENO5_vs_NNTENO5_Scalar.py`'s existing comparison harness before trusting it in
training. **Met** — verified directly against `teno5.py`'s own `teno5_neural_reconstruct`
(a fake model driving both paths with identical probabilities), not just self-consistency,
to `1e-12`.

---

## Phase 5 — Unrolled Physics Loss (Étape 3, 1 week, gated on Phase 1)

> **Status: done, with one confirmed-underspecified term.** `src/differentiable/losses_jax.py`
> implements all four terms plus `unrolled_physics_loss` (weighted sum + per-term
> breakdown dict, for the logging Phase 6 needs during a λ sweep). `L_Reg` is a generic
> `jax.tree_util`-based pytree L2 distance, not hardcoded to one network shape, so it
> keeps working if Phase 6 swaps in a different architecture. `L_BVD` has no formula
> anywhere in the Unrolled Physics Loss proposal — of everything built across Phase 0-5, this is the single most
> underspecified point in the whole document, more so than `c̃_WENO/TENO` in Phase 2. The
> reading implemented (network's own interface jump penalized only when it *exceeds* the
> deterministic TENO5 scheme's jump at that interface) is grounded in Phase 2's own BVD
> jump construction, not invented from nothing, but it is a guess and should be confirmed
> before any result built on it gets reported as validating the Unrolled Physics Loss proposal's design.
> `tests/test_losses_jax.py` (4/4 passing) verifies all four terms are finite on a real
> Phase 1 rollout, `L_Reg` is exactly zero at the anchor and positive away from it, and
> `jax.grad(L_total)` returns finite, nonzero gradients.

Implement Équations 5–8 as JAX-differentiable functions operating on the Phase 1 rollout
output:

- `L_MSE`: direct port, Équation 6 — trivial once Phase 1's rollout returns `u^{n+K}`.
- `L_TV`: Équation 7, `TV(v) = Σ|v_{i+1}-v_i}|` — one line in `jnp`; the `max(0, ·)` hinge
  means it only penalizes *increases* in total variation, i.e. it's a soft TVD
  constraint, not a hard one — say so explicitly if reporting this later, since it will
  not by itself guarantee no new oscillations.
- `L_BVD`: the Unrolled Physics Loss proposal doesn't give this term's formula (only names it) — the natural
  definition consistent with the rest of the doc is the interface-jump term `J_{j+1/2}`
  from Phase 2, penalized when it exceeds the deterministic TENO5 scheme's own jump at
  the same interface. Confirm this reading before implementing since it's the one
  underspecified equation in the PDF.
- `L_Reg`: Équation 8, `‖c_NN - c_base‖²` — this is exactly the anchor-weight penalty
  already built and validated in `Train_NNTENO5.py` (`--warm-start-path`/
  `--anchor-weight`, Results.md Phase 9), just reframed as one term in a combined loss
  instead of the whole objective. Reuse that code path rather than rewriting it.

Sweep `(λ1, λ2, λ3, λ4)` the same way Phase 9 swept `anchor_weight` — start with `λ4`
(regularization to the safe baseline) dominant, the same lesson learned from Phase 8/9:
unregularized oracle-style training destabilized the scheme, and anchoring to the
deterministic baseline was what fixed it.

**Exit criterion:** all four loss terms compute without NaN on a Phase 1 rollout, and
`jax.grad(L_total)` returns finite gradients w.r.t. network weights. **Met.**

---

## Phase 6 — Training loop (3–5 days)

> **Status: scalar path — real positive result (2/3 beats_teacher). Euler path — trains,
> but a genuine numerical instability blocks it, root-caused but not solved.** This is the
> single most important section in this whole document — it's the first real test of
> whether the Unrolled Physics Loss proposal's core idea (train via a K-step unrolled physics loss) actually beats
> TENO5, not just whether it's implementable. Full account below; `Train_NNTENO5_UnrolledJAX.py`
> and `Train_NNTENO5euler_UnrolledJAX.py` are the scripts, both runnable via
> `conda run -n teno-jax python -u <script>`.
>
> **Scalar advection, three runs, each fixing what the last one's diagnostics exposed:**
>
> | Run | Change from previous | held-out jump, K steps | held-out jump, 3x K | held-out smooth sine | Result |
> |---|---|---|---|---|---|
> | 1 | baseline (raw `L_TV`, `L_Reg` sums, Eq. 7/8 literal) | loses (0.142 vs 0.118) | loses (0.210 vs 0.125) | loses badly (0.051 vs 0.0048, ~10x worse) | 0/3 |
> | 2 | normalized `L_TV`/`L_Reg` (mean, not sum) | loses (0.147 vs 0.118) | loses (0.182 vs 0.125) | loses badly (0.044 vs 0.0048) | 0/3 |
> | 3 | + imitation-pretrain first, anchor to that (not to random init) | **beats** (0.115 vs 0.118) | **beats** (0.116 vs 0.125) | near-tie (0.0049 vs 0.0048, ~3.5% worse) | **2/3** |
>
> Run 1→2 diagnoses a real, generalizable bug: the Unrolled Physics Loss proposal's Eq. 7 (`L_TV`, a raw sum) and
> Eq. 8 (`L_Reg`, a raw sum) both grow with grid size / parameter count, while Eq. 6
> (`L_MSE`, a mean) doesn't — at `CELLS=64` this let `L_TV` dominate the gradient by
> roughly 4x even after its 0.1 weight, training the network to minimize total variation
> over getting the answer right. Fixing the scale mismatch (mean instead of sum) made the
> logged per-term losses sane but **did not by itself close the gap to TENO5** — run 2
> still lost on every case, including the smooth one, where even reproducing TENO5's own
> near-trivial behavior should be easy. That's the tell that the real problem was the
> *optimization*, not the loss weighting: training a `5→64→32→16→8→3` network from a random
> initialization, cold, against a loss that only gives feedback after a K-step rollout, is
> a hard, sparse optimization problem. Run 3's fix is exactly the pattern already proven in
> this project's own Results.md Phase 9 (anchor-weight fine-tuning from a warm-started
> baseline, not training from scratch): imitation-pretrain the network to reproduce
> deterministic TENO5's own reconstruction first (500 steps, dense per-stencil MSE, no
> rollout, **0.6 seconds** — a completely different, much better-conditioned optimization
> problem than the unrolled one), then use that pretrained checkpoint as both the starting
> point and the `L_Reg` anchor for the unrolled fine-tune. That closed the gap: 2 of 3
> held-out cases now genuinely beat deterministic TENO5, including generalizing to 3x the
> trained rollout horizon (never seen during training), with the third (smooth sine) a
> near-tie rather than a 10x loss.
>
> **This is a real, if narrow, positive result for the Unrolled Physics Loss proposal's central claim**, on the
> scalar advection case specifically: an unrolled-physics-loss-trained network that beats
> the deterministic scheme against the real (exact) solution, on held-out problems,
> including beyond the training horizon. It required a fix the proposal itself doesn't
> mention (imitation-pretrain first) and a loss-scaling fix its literal equations don't
> account for (normalize `L_TV`/`L_Reg`) — both are now documented, reusable findings, not
> guesses.
>
> **Euler did not reach the same result, and the reason is a real finding, not a bug left
> unfixed.** Reusing the pretrained scalar weights as an Euler warm start/anchor (valid
> because imitation-pretraining is representation-agnostic — see
> `Train_NNTENO5euler_UnrolledJAX.py`'s docstring) immediately produced NaN losses at step
> 0 of training. Tracing this down (full debugging trail, not guesswork) found, in order:
> (1) a genuine missing data-validity filter — `GenerateTrainingData_EulerExact.py`'s own
> `generated_states` already rejects vacuum-forming random Riemann samples
> (`if np.any(state[:, 0] <= 0): continue`); the new sampler didn't, so one bad sample in
> an 8-example batch poisoned the whole batch — fixed by adding the same filter
> (`sample_riemann_problem`'s rejection loop); (2) fixing that alone didn't help, because
> the actual NaN source was an *intermediate* SSPRK3 substage: `ssprk3_step` calls
> `flux_fn` on its internal `u1`/`u2` values before producing the final state, and the
> positivity guard was only being applied to the final output, not those intermediate
> substages — fixed in `numerics_jax.ssprk3_step` itself (now takes `state_guard` and
> applies it after each of its three internal stages); (3) with substage guarding wired
> through, step-by-step tracing of the one training example that was NaN'ing (a steep,
> ~p=9 random Riemann problem — not the canonical Sod case) showed density hitting a
> `1e-6` floor at step 6 and the max state value exploding to `5.7e144` by step 7 — flooring
> density that aggressively makes the local sound speed (`~sqrt(pressure/density)`)
> explode, which is what actually drove the blowup, not the density floor itself. Raising
> the floor to a physically reasonable scale (`eps=0.05`, matching the problem's O(1)
> magnitudes) and adding a velocity cap (needed too — flooring density while leaving
> momentum untouched spikes `velocity = momentum/density` right where density was just
> floored) delayed the blowup (step 8 → step 11) but **did not prevent it**. Testing the
> standard Lax-Friedrichs robustness fix — over-estimate the dissipation coefficient
> `alpha` with a safety margin — gave a **non-monotonic, counter-intuitive result**: a 1.5x
> margin delayed blowup further (state stays finite but reaches `6.2e18` by K=15, still a
> practical blowup); a 2x-5x margin made it **worse**, blowing up earlier (step 3, not
> step 8-11). More dissipation making a Lax-Friedrichs scheme *less* stable rules out "the
> wave-speed bound is just a bit too small" as the explanation and points instead at the
> Roe eigenbasis itself (`roe_eigenbasis_jax`'s `jnp.linalg.inv` of a 3x3 matrix built from
> Roe-averaged velocity/enthalpy/sound-speed) becoming ill-conditioned as the state
> develops under an evolving, not-yet-Euler-trained network's reconstruction — a
> genuinely different, harder failure mode than anything the scalar case exposed, because
> scalar linear advection's stability never depends on the reconstructed *value* feeding
> back into a wave-speed or eigenbasis calculation the way a nonlinear system's does.
>
> **This was the point to stop patching and report a finding instead of continuing to
> guess.** Masking NaN examples out of a batch average doesn't work either, for a related
> reason to Phase 0's "double `jnp.where`" trap: even with the bad example's *loss value*
> zeroed out, its *gradient* is typically already NaN from the same forward computation,
> and multiplying a NaN gradient by a zero mask still gives NaN, not zero — fixing this
> properly needs the network to be bypassed for a diverging example before the NaN is
> created (a "circuit breaker"), which is a real, nontrivial mechanism to build correctly,
> not a one-line patch. **Recommendation for the professor**: training NN-TENO5-style
> networks for systems of conservation laws via an unrolled physics loss has a real,
> substantive stability risk beyond what the Unrolled Physics Loss proposal's plan anticipates — positivity
> preservation (which Table 1 already asks for, as an *evaluation* criterion) turns out to
> also be a *training-time* prerequisite, not just an outcome to check afterward. Options
> worth discussing: a curriculum (train on mild Riemann problems first, only gradually
> introduce steep ones), an explicit stability-preserving projection on the characteristic
> decomposition itself (a real research contribution, not an implementation afterthought),
> or scoping the unrolled-loss training to scalar/mildly-nonlinear problems (where this
> pass's positive result already stands) and keeping Euler on oracle/imitation-style
> training (Results.md Phases 6-9) for now.
>
> **Update — the curriculum option was tried (Results.md Phase 11), backed by a deep
> literature review (six papers read in full, not abstract-scanned).** Um et al.
> ("Solver-in-the-Loop", NeurIPS 2020) report this exact failure mode and prescribe
> exactly this fix. Tested directly: curriculum on K (2→4→8→15) measurably delayed the
> instability (full K=2 stage completed vs. immediate failure at K=15 cold) but still
> diverged partway through K=4. Combined with the non-monotonic dissipation-margin result
> above, this rules out "just needed a gentler schedule" as the whole story and points at
> the structural fix instead: a November 2025 paper ("Learning Under Laws") diagnoses
> exactly why a state clamp (this project's `euler_positivity_guard`) isn't enough — it
> "only enforces positivity, not conservation or entropy" — and demonstrates a proper
> differentiable projection onto all four constraints jointly (Dykstra's method,
> ~10-12% overhead), which was not attempted here. That's the concrete next step, not
> another schedule or margin sweep.

- `K` (rollout length) is a new hyperparameter with no precedent in this repo — start
  small (K=1, equivalent to today's per-stencil training) and increase only after
  confirming Phase 0's memory/compile-time behavior scales acceptably. Unrolling further
  multiplies both.
- Reuse `Train_NNTENO5.py`'s CLI/metadata conventions (`--warm-start-path`,
  `--anchor-weight`, split files, metadata JSON) for consistency with the rest of the
  project's tooling, but the training step itself must call into the new JAX rollout —
  this is not a drop-in change to the existing Keras `model.fit()` call.
- Log per-term loss values (`L_MSE`, `L_TV`, `L_BVD`, `L_Reg`) separately during
  training, not just the weighted sum — needed to diagnose which term dominates/collapses
  during the `λ` sweep.

**Exit criterion:** a trained checkpoint plus a training-curve CSV in the same format as
`models/nn_teno5_oracle_history.csv`, with per-term loss columns. **Met for scalar**
(`models/nn_teno5_unrolled_jax_{history,params}_run3_pretrained.*`); **not met for Euler**
(training runs and produces a checkpoint, but that checkpoint reflects a NaN-corrupted
optimizer state, not a meaningful result — see status note).

---

## Phase 7 — Validation matrix (Étape 4's Table 1, 3–5 days)

> **Status: run, and the result adds a real, important nuance to Phase 6's positive
> finding.** `ValidationMatrix_ProjPDF.py` runs the run3_pretrained network from Phase 6
> against canonical versions of the Unrolled Physics Loss proposal's own test problems (a fixed-width step profile
> over `final_time=1.0`, and Burgers 1D pre-shock) rather than the same held-out-but
> similar random profiles Phase 6's evaluation used. Result: **it loses to deterministic
> TENO5 on both** — Advection L2 0.140 vs. TENO5's 0.099; Burgers (explicitly
> out-of-distribution — the network never saw Burgers' nonlinear flux during training) L2
> 0.0030 vs. 0.0008.
>
> This is not a contradiction of Phase 6's 2/3 result, it's a scope finding: Phase 6's
> held-out cases stayed close to the training distribution (similar random square-wave/sine
> shapes, evaluated at the trained horizon K and at 3xK). This validation matrix's
> `final_time=1.0` is about **8.5x** the trained rollout horizon (`K x dt ~= 0.117`), and a
> fixed step width the training distribution's random range (0.15-0.5) happens to sit at
> the edge of. **The honest characterization of Phase 6's result: real, but narrow** — an
> unrolled-physics-loss-trained network can beat TENO5 on problems similar in kind and
> time-horizon to what it trained on, and generalizes some way past that (3x K, Phase 6),
> but this pass found no evidence it generalizes to canonical benchmark-scale problems
> without being trained closer to that scale/scope. A production Phase 6 would need to
> train on a K and problem-shape distribution that actually spans the Unrolled Physics Loss proposal's Table 1 cases,
> not treat Table 1 as a pure hold-out set for a narrower training distribution.
>
> Shu-Osher and Sod (Euler) rows are **not evaluated for NN-TENO5euler** — reported
> honestly as not evaluated, not as a loss — since Phase 6's Euler training did not
> converge to a meaningful checkpoint (see that section's status note). Deterministic
> TENO5euler's own behavior on these is already established elsewhere (Results.md;
> Phase 1's regression tests, matching real NumPy to `5.5e-11`).

Map the Unrolled Physics Loss proposal's benchmark table directly onto existing scripts, extending each with the
metric the PDF asks for:

| the Unrolled Physics Loss proposal test case | Metric | Existing script to extend |
|---|---|---|
| Advection Linéaire | L1/L2 error, shock width, TV ≤ const | `WENO5_vs_WENO5NN_StepAdvection.py`, `Evaluate_WENO5NN_StepAdvection.py` |
| Burgers 1D | error spectrum vs Δx, shock-speed conservation | `WENO5_vs_WENO5NN_Burgers.py` (already has multi-resolution sweep data — see Results.md Phase 1) |
| Shu-Osher 1D | spectral peak amplitude | `WENO5_vs_WENO5NN_ShuOsher.py`, `WENO5_vs_WENO7NN_ShuOsher.py` |
| Sod / Sedov | positivity of P, ρ | `WENO5_vs_WENO5NN_Sod.py`, `TENO5_vs_NNTENO5_Euler.py` — positivity is already enforced/checked via `eulerSimulation._validate_state` (`SimulationClasses.py:65`); Sedov has no existing script and needs a new IC in `src/initial_conditions/InitialConditions.py` |

Every one of these must also pass the **existing** `beats_teacher` real-solution
acceptance criterion (Results.md Phase 6: NN error vs. exact solution ≤ deterministic
scheme's error vs. exact solution) — the Unrolled Physics Loss proposal's table checks internal consistency
(positivity, TV, spectral peaks) but not "did this get closer to the truth than TENO5
already does," which per your own framing earlier in this project is the actual bar that
matters. Don't drop that check just because the Unrolled Physics Loss proposal's table doesn't ask for it.

**Exit criterion:** a results table with one row per test case, both the Unrolled Physics Loss proposal's own
metric and the `beats_teacher` win/tie/loss column, written to `Results.md` as a new
Phase entry — matching the format already used for Phases 5–9. **Met** — see Results.md
Phase 10, and the status note above for the two rows that couldn't be filled in.

---

## Suggested order and go/no-go gates

```
Phase 0 (spike)  ──▶  PASS? ──no──▶ stop, report finding, re-scope K/architecture
      │ yes
      ▼
Phase 1 (solver rewrite)  [longest phase — budget the most time here]
      │
      ├──▶ Phase 2 (feature vector)  ──┐
      │                                 ├──▶ Phase 3 (dataset + non-identifiability audit)
      └──▶ (can run in parallel) ───────┘
                                          │
                                          ▼
                                   audit clean? ──no──▶ revisit Phase 3 mitigation before scaling
                                          │ yes
                                          ▼
                                   Phase 4 (projection layer)
                                          │
                                          ▼
                                   Phase 5 (unrolled loss)
                                          │
                                          ▼
                                   Phase 6 (training)
                                          │
                                          ▼
                                   Phase 7 (validation matrix + beats_teacher)
```

Phases 2 and 3 don't depend on Phase 1 finishing and can start in parallel once Phase 0
has de-risked the overall approach. Everything from Phase 4 onward needs Phase 1's
differentiable rollout to exist.

Total realistic estimate: **5–8 weeks** for a first end-to-end result, dominated by
Phase 1 (solver rewrite) and Phase 3 (dataset curation + non-identifiability audit) — not
the network architecture itself, which is the smallest piece of this plan.
