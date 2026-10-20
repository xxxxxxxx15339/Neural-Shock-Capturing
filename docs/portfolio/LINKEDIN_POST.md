# Suggested LinkedIn caption

Can a neural network improve a shock-capturing solver without breaking its
numerics? My research internship explored that question with WENO/TENO, 1D PDEs
and differentiable JAX. The useful result was a reproducible investigation:
measured gains, failed settings, convergence limits and clear attribution.

Code and evidence: https://github.com/xxxxxxxx15339/Neural-Shock-Capturing

# Main LinkedIn post

During my research internship, I worked on a question at the intersection of CFD
and scientific machine learning: can a small neural network improve how a
numerical solver captures shocks?

I extended a WENO/TENO research codebase and implemented a differentiable JAX
pipeline for linear advection, inviscid Burgers and the 1D compressible Euler
equations. Separate scalar and Euler networks learn candidate-selection behavior
through short PDE rollouts.

The hybrid uses a deterministic TENO5 smoothness gate: neural corrections are
allowed where the classical detector rejects a candidate stencil. I tested how
much correction to apply, how soft and hard decisions differ, and what happens
to conservation, stage positivity and smooth-grid accuracy.

Using all 922 validation trajectories, a modest soft correction reduced macro L1
error relative to TENO5 by about 2.3% for scalar problems and 5.2% for Euler at the
15-step horizon. These are validation-selected results, not a claim of universal
superiority. Larger corrections produced failures, longer rollouts showed
problem-dependent behavior, and a strong blast-wave smoke test remained unstable
at the selected setting.

One valuable finding was a numerical limitation: the historical Euler split-flux
path is only about second order for general smooth cell-average data. An optional
finite-volume formulation recovered approximately fifth-order spatial consistency
on that diagnostic; its shock behavior still needs validation.

The dataset contains analytical mathematical reference solutions—not measured
flow experiments. The repository includes checkpoint hashes, numerical tests,
uncertainty estimates, reproducible experiments and negative results.

Thank you to my supervisor, Radouan BOUKHARFANE, who proposed the unrolled
physics-loss research direction. The foundational WENO-NN work is by Ben Stevens
and Tim Colonius; WENO/TENO algorithms come from the numerical-analysis literature.

As a fourth-year AI engineering student, this project strengthened my interest in
research internships involving scientific ML, aerospace, CFD and research software.

Repository: https://github.com/xxxxxxxx15339/Neural-Shock-Capturing

#ScientificMachineLearning #ComputationalFluidDynamics #JAX #NumericalMethods #ResearchSoftware

# Longer technical version

My research internship investigated neural augmentation of high-order shock-
capturing methods for scalar conservation laws and the one-dimensional Euler
equations.

The engineering work covered a differentiable JAX solver, characteristic Euler
reconstruction, SSPRK3 time stepping, separate scalar/Euler models, analytic
cell-average training targets, strict checkpoint compatibility and a common
numerical evaluation path.

The neural architecture stayed fixed at 5 → 64 → 32 → 16 → 8 → 3. A deterministic
TENO5 gate suppresses neural corrections wherever all classical candidate
stencils are accepted. Candidate weights, gating and correction strength remain
separate, testable parts of the method.

I evaluated strengths 0, .1, .25, .5, .75 and 1 with both soft and hard candidate
policies on the full validation split: 615 scalar and 307 Euler trajectories.
Paired, stratified bootstrap intervals quantify trajectory sampling uncertainty.
A soft correction at λ=.25 yielded macro L1 ratios of .9770 for scalar
(95% interval .9742–.9788) and .9479 for Euler (.9436–.9523). Constants and nearly
zero baseline errors were excluded from ratios; their absolute errors remain
reported. Selection used validation only.

That short-horizon improvement did not settle the scientific question. Long-time
advection and shock problems behaved differently. Stronger Euler corrections
could violate admissibility, and the selected method failed a strong blast-wave
smoke test. A diagnostic of the historical positivity repair
measured mass and energy changes directly instead of calling it conservative.

Spatial tests also showed why a smoothness gate is not an accuracy proof. Regular
smooth profiles, smooth critical points and general nonlinear Euler cell averages
need separate checks. The historical Euler flux path approached second order on
the general nonlinear test; an explicitly separate finite-volume state-flux
variant approached fifth order on that spatial diagnostic. It is not yet a
validated replacement for shock calculations.

A second investigation compared Euler warm-up characteristic-state stencils with
the characteristic split-flux inputs encountered by the solver. Their normalized
distributions differ. A reusable split-flux warm-up option is available for a
future controlled ablation; no retraining benefit is claimed from distribution
plots alone.

The portfolio includes measured solution profiles, gate/correction diagnostics,
convergence studies and CPU accuracy/cost comparisons, with reference uncertainty
and failed configurations kept visible. All labels are mathematical reference
solutions, not experimental measurements. One original training seed per physics
and unresolved general stability questions remain limitations.

The unrolled physics-loss proposal and numerical clarifications are credited to
my supervisor, Radouan BOUKHARFANE. The original WENO-NN foundation is Stevens and
Colonius; classical TENO is Fu, Hu and Adams. My contribution is the implementation,
extension, evaluation and reproducible software around this research direction.

Project and complete evidence: https://github.com/xxxxxxxx15339/Neural-Shock-Capturing
