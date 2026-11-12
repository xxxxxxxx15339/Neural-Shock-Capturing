# Audit and method notes

The starting point is `f9d4c2ae7e6009430248f71774996d952e08abbd`.
The only initial untracked material was `EULER_STABILITY_RESEARCH_AUDIT.md`
and `interview_sebastian_final/`; both are left untouched. The experimental
branch is `research/accuracy-upgrade-2026-11-12`. Nothing is pushed.

## What the source actually does

| Generation | Inference input | Reconstruction and flux | Mixing input/objective |
|---|---|---|---|
| Legacy Keras NN-TENO5 | min–max normalized five-point stencil | candidate discard masks; scalar state reconstruction or Euler characteristic split flux, depending on caller | supervised weighted binary cross entropy; thresholds calibrated on validation |
| Historical JAX scalar | normalized scalar state stencils, both directions | reconstruct states, then physical scalar flux with LF stabilization | linear head takes three troubled probabilities; later rich head takes those plus five normalized stencil values |
| Historical JAX Euler | normalized characteristic positive/negative split flux stencils | Roe basis from neighboring averages; reconstruct `F(Ubar)` LF splits, then shared face flux | original linear/rich heads; historical phase objectives include MSE/TV/BVD/regularization/positivity, with settings varying by phase |
| Saved V2 Euler | same characteristic split-flux inference inputs | historical numerical path, deterministic TENO trouble gate, soft weights | backbone only, **no saved dedicated head**; fixed λ=.25, normalized K=15 MSE plus pre-repair positivity penalty |
| Saved V2 scalar | normalized state stencils | scalar FV reconstruction, deterministic TENO gate | backbone only; training λ=.75, validation-selected λ=.25; MSE plus TV-growth penalty |
| Existing state-flux prototype | characteristic **state averages**, both directions | periodic-only state reconstruction then local Rusanov | reconstruction policy supplied by caller; no retraining or shock validation in that prototype |

The saved V2 metadata omits the subsequently introduced `pretraining_inputs`
argument. The original/default branch of `training_pool` uses forward
characteristic states, whereas `get_euler_flux_jax` calls
`euler_split_flux_stencils`. This is a verified input mismatch, not a proof of
its causal effect. The extended feature vector is a separate NumPy module;
the saved architecture still has exactly five inputs and 3155 parameters.
Its isolated-stencil BVD descriptor is explicitly documented as an approximation.
No such descriptor or BVD penalty is added here.

The historical Euler source describes its inputs as cell averages and uses a
flux-difference update, but feeds `F(Ubar)` into cell-average reconstruction
coefficients. This has a finite-difference interpretation on suitable point
data; that does not resolve the inconsistency when the inputs are **finite-volume
cell averages**. In general `F(Ubar) != average(F(U))`, with an O(dx²) discrepancy.
The density-only entropy wave is a special case where the nonlinear-flux issue
is hidden. The coupled smooth diagnostic tests a different question.

Weights-only historical pickle files do not contain complete optimizer/RNG
states or exact per-update provenance. Consequently their complete training
histories cannot be recovered from weights or source defaults. Adjacent metadata,
archived phase reports and the baseline manifest retain the available evidence;
this is an explicit reproduction limit, not an inferred exact training recipe.
Previously examined validation and historical test cases are not fresh test data.

## Experimental finite-volume path

`src/differentiable/accuracy_upgrade.py` adds an opt-in `flux_variant` factory.
`variant='historical'` delegates directly to the existing factory; no baseline
source is edited. `variant='fv_state_flux'` reconstructs conservative averages
in a face-local Roe basis and back-projects left/right interface states before
evaluating physical fluxes. The high-order Rusanov speed is
`max(|v_L|+c_L, |v_R|+c_R)`. This is the usual two-state estimate; no universal
nonlinear shock-stability or positivity theorem is attributed to it.

`state_stencils` is the canonical source for **both directions** of the new
solver and warm-up C. Historical matched warm-up B calls the existing canonical
`euler_split_flux_stencils`. A deliberately retains the historical forward-state
warm-up. Both directions use the same face basis; one basis computation is
performed per call, including six extra ghost bases, without recomputing it
for the mirrored stencil. Periodic duplicate endpoints are made identical.
Transmissive boundaries use constant ghost extrapolation; no fifth-order
boundary-accuracy claim is made for a nonzero true boundary gradient. Reflecting
walls mirror momentum and explicitly impose zero mass and energy wall flux,
while retaining wall momentum flux. Wall momentum need not be conserved in
the domain because pressure exerts a force.

Averaged states must be physically admissible before a Roe basis is evaluated.
The checked step rejects invalid initial states without evaluating a physical
flux/eigenbasis on them. The raw flux factory is a low-level function with an
admissible-input contract; it must be used with a checked integrator on shocks.

## Conservative safeguard and its limits

The low-order fallback uses **global** LF with
`alpha=1.05 max_i(|v_i|+c_i)`. For positive ideal-gas states and `nu alpha <= 1`,
the FE update is a convex combination of `U_i` and neighboring split states
`U ± F(U)/alpha`. This is the low-order convexity argument, not a theorem for
unlimited high-order reconstructions. Code additionally requires a stricter
configured CFL <=.9 and checks density/pressure floors in floating point.

Finite nonphysical reconstructed face states are scaled toward their own cell
averages **before** evaluating sound speed or physical flux. This changes only
face fluxes. A global face coefficient is used for reliability; one difficult
face can suppress reconstruction throughout the domain. Nonfinite face
reconstruction is separately flagged and falls back to cell faces in guarded
mode. The averages themselves are never clipped, replaced or averaged.

For every FE-like update, the code first checks the low-order candidate. Given
an admissible low candidate, it bisects a **single global** coefficient on the
segment to the high-order candidate. Density is affine and pressure is concave
on positive density, so the admissible segment is an interval. Forty bisections
and a small safety factor select an admissible coefficient. The actual update
is recomputed from `F_low + theta (F_high-F_low)` with one shared flux at each
interface, then checked again. This also exposes finite precision failures.

SSPRK3 combines accepted FE-like updates. A failed attempt is discarded in its
entirety; up to eight timestep halvings are allowed. Accepted time is returned
explicitly. Exhaustion returns the original input, accepted time zero, and
`success=False`; callers must stop rather than report a completed solution.
Rejected fluxes contribute no boundary exchange. Budgets use accepted fluxes
with SSPRK weights (1/6, 1/6, 2/3). No historical nonconservative state repair
is called by the new path.

Limiter coefficients and discontinuous decisions are stop-gradient. Gradients
through smooth inactive regions are tested against finite differences. This is
not full differentiability through switching, and no global accuracy proof for
the limiter, entropy inequality, TVD theorem or universal positivity theorem is
claimed. `face_guard` and the full stage safeguard are separate options.

## Experimental and reference discipline

The paired pilot uses seeds 3101/3102, identical architecture, 150 teacher
steps, 60 physics updates, Adam 2e-4, gradient clipping at 1, batch 4 and K=15.
It trains at N=64 and evaluates all validation trajectories at N=64/128.
It is **smaller than saved V2 training** (300 updates, three grids). Final
checkpoints are selected at a fixed update, not by the final holdout. The
training objective is normalized MSE plus .1 pre-rejection positivity penalty.
Invalid updates would be rejected; no repair is applied. Training A/B retains
the historical flux and C uses state flux with face admissibility scaling.
The pilot rejects invalid FE-like sub-updates, a stricter check than the
original raw SSPRK stages; none activates in any of the six recorded runs.
These are controlled pilots, not exact recreations of the original guarded
training procedure.

Initial teacher probes use training trajectories only. Training and validation
IDs are checked for disjointness. Test datasets are hashed for preservation
but never opened as datasets. The final six Riemann specifications were written
and SHA-256 frozen before training; all candidates are frozen before confirmation.
No holdout result selects a strength, architecture or optimizer setting.
Intervals resample the six independent trajectories separately at each grid,
never time snapshots or two grids as independent cases. Such a small holdout
cannot support broad generalization claims.

Euler unscaled L1/L2 summaries average conserved components of different units;
component errors are saved and must be consulted. Scalar results are reported
separately. Runtime diagnostics in exploratory runs include host overhead and
sometimes concurrent workers. Only `performance.json` is the isolated production
cost comparison, with synchronized JAX execution and nine warmed repeats.

The first Euler wrapper checked every FE-like sub-update even for the raw
baseline. It was corrected to check actual SSPRK stages before final comparisons.
The initial results remain in `results/euler/`; use `results/euler_actual_stages/`
for scientific comparison. A regression compares the wrapper against the original
SSPRK3 arithmetic. Temporal and reference runs that completed their stricter
checks remain valid, with floating-point rearrangement differences only.

## Academic grounding

[Fu, Hu & Adams (2016)](https://doi.org/10.1016/j.jcp.2015.10.037) motivates
TENO selection and recovery of the linear reconstruction in regular smooth
regions. It does not validate a newly learned selector.

[Zhang & Shu (2010), author manuscript](https://www.math.purdue.edu/~zhan1966/research/paper/euler.pdf)
provides the convex admissible-set framework for FV/DG methods and the low-order
LF argument in Remark 2.4. Its high-order polynomial/quadrature limiter is not
copied here. Our global flux-segment correction has different retention and
accuracy properties that require separate measurements.

[Zhang & Shu (2012), author manuscript](https://www.math.purdue.edu/~zhan1966/research/paper/fd-weno.pdf)
concerns finite-difference WENO and cannot be applied to cell averages without
checking the discretization. [Hu, Adams & Shu, conservative flux limiting](https://arxiv.org/abs/1203.1540)
is additional methodological grounding for flux correction; this implementation
uses a global update segment rather than their local sufficient-condition
construction. Neither reference proves the entire neural method implemented here.
