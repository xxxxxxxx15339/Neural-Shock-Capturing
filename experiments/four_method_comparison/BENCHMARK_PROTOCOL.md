# Frozen matched protocol

The authoritative, hashed declaration is [benchmark_manifest.json](configs/benchmark_manifest.json),
created before matched evaluation. Models are selected by architecture, finite
load status and historical training/selection documentation, not these test errors.
All canonical benchmarks have historical exposure; this is not an untouched test.

Four authentic reconstruction identities are evaluated. The NENO6 classifier,
normalization, polynomials, candidate ordering, optimized coefficients and threshold
come from the paper-linked author repository. Its author scalar advection smoke
is reproduced before using the reconstruction in the common Euler/Burgers solver.
These latter runs are **matched ports**, not the author's published solver runs.
The author's PyTorch mask/weight arithmetic is float32 despite double network and
states; the adapter retains it. The original Keras classifier is also float32;
numerical solution arithmetic and historical JAX heads are float64.

Euler uses historical characteristic split-flux reconstruction, adjacent-state
spectral-radius LF splitting and one Roe basis per interface. Periodic/transmissive
boundary implementations are unchanged. Three mirrored ghost cells and explicit
zero wall mass/energy flux are used for reflecting-wall blast tests. Wall momentum
exchange is measured, not assumed zero. NENO6's asymmetric six-point offsets are
(-2,-1,0,1,2,3) and (3,2,1,0,-1,-2) at the same interface. No five-point checkpoint
is used for that method. Scalar uses the repository's two reconstructed states
and LF numerical flux with alpha=1. This nonlinear scalar formulation is not a
proof of fifth/sixth-order finite-difference accuracy.

All methods use point samples x=i*L/N, the historical repository convention.
We do not reinterpret these as exact finite-volume cell averages. Historical
Euler split-flux is not substituted with the experimental state-flux solver.
No observed method difference is attributed to a solver correction.

Every case has a common fixed timestep, calculated from its initial speed with
CFL=.25 and safety factor1.3, then adjusted to end exactly at T. This is an
experimental timestep choice, not an admissibility theorem. Smooth advection uses
dt≤.15*dx^(5/3), making SSPRK3 temporal error scale as O(dx^5), with a separate
finest-grid dt/2 sensitivity control. The measured full-solution order is reported; no fifth-order
temporal claim is made. Grids: main discontinuous problems128/256; smooth64/128/256;
stress64/128. Physical case definitions and final times are in the manifest.

Sod and Lax use the repository's exact gamma=1.4 Riemann solver. Two-rarefaction,
strong contrast and near-vacuum also use exact non-vacuum Riemann solutions.
Shu–Osher uses the repository's unshifted sin(5*x) initial condition, x∈[0,10],
shockx=1, T=1.8; oscillatory region[5.5,7.2]. An independently run classical
TENO5 ladder512/1024/2048 and a finest-grid dt/2 check measure reference sensitivity.
The dt/2 finest solution is linearly interpolated onto evaluation points. Both
reference differences and their effect on method rankings are reported; they
are sensitivity estimates rather than rigorous error bounds. Blast is stability
only: no trustworthy converged full-time reference is asserted.

Burgers sine u0=.5+.25sin(2πx) has shock time2/π. T=.5 is pre-shock; T=.9 is
post-shock. The exact entropy branch is solved in the frame y=x-.5t: the left
root on[0,.5] and its reflected right branch. The stationary moving-frame shock
lies at y=.5 and its exact-point value is defined as the average of its traces.
Top-hat u=1 on[.2,.6), zero elsewhere, T=.2 has a rarefaction and shock before
interaction, so its entropy reference is explicit. Its shock speed is1/2.

Raw actual SSPRK3 stages are checked for finite values, rho>0 and p>0. Evaluation
stops at the first invalid stage, saving its candidate, last admissible stage,
last complete state, step and time. No repair, retry or hidden fallback is used.
Failed rollouts receive no finite error or full-time runtime score. Runtime to
failure is retained separately. All-discard NENO6 NaNs remain visible.
Diagnostic arrays retain three slots for JAX's fixed shapes: after the first
invalid stage, later slots are unexecuted padding. Use first_invalid_stage and
the explicit attempted/invalid stage counters; padded slots are not extra failures.

Metrics: mean absolute point error (L1), RMS error, maximum absolute error;
Euler rho/momentum/energy L1 and their arithmetic mean; separate primitive
velocity/pressure errors; TV with periodic closing edge only for periodic cases;
overshoot/undershoot relative to declared reference extrema. Shock position is
the maximum adjacent density/scalar jump within ±.12L of the exact shock.
Resolution is the distance between interpolated 10/90% crossings nearest the
exact wave in that window; missing crossings remain unavailable.
For Euler Riemann cases, each search radius is also capped at45% of the nearest
exact neighboring-wave separation, preventing the contact diagnostic from
selecting a stronger shock. This is determined from the reference, not a model's
error. The stored primary L1 errors do not depend on the wave-search diagnostic.
Conservation is
dx*sum(U_final-U_initial)+integrated(F_right-F_left), with SSPRK3 weights
(1/6,1/6,2/3) for stage boundary fluxes.

Original NN-TENO5 and phase21/run6 have different backbone weights. The main
V2-vs-NN percentage describes the whole method. The causal head ablation uses
phase21/run6's exact same backbone for pure neural, fixed lambda=sigmoid(-3),
and learned lambda on N=128. Fixed lambda is predeclared from historical
initialization, not optimized on tests. Lambda statistics are descriptive; no
calibrated-uncertainty interpretation is made.

JAX CPU timing separates compilation, first execution, seven warmed single-step
repeats, and three warmed whole-device-loop rollouts. Median/IQR are saved.
Host diagnostic wall time is separate. Failed trajectories are not credited as
fast complete simulations. A cross-grid accuracy/cost comparison is descriptive
where error references permit it. No bootstrap of spatial cells or timesteps is
used: there are too few independent trajectories per family for reliable intervals.
