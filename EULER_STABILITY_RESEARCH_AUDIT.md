# WENO-NN Euler stability and generalization: research audit

Audit date: **29 October 2026**. Audited HEAD: **`f9d4c2a`** on `main`. No solver, checkpoint, dataset, historical result, or training configuration was changed for this investigation. No training was run.

Evidence labels used throughout:

- **[A] Archived measurement:** machine-readable result, state array, log, or directly inspectable figure in this repository.
- **[R] Reproduced:** lightweight numerical check performed during this audit; commands and measurements are recorded in §10.
- **[H] Historical account:** a committed written account or commit message, without a complete surviving experiment record. This is evidence of what was reported, with weaker reproducibility than [A]/[R].
- **[I] Implementation fact:** behavior established by reading the relevant implementation.
- **[T] Theory:** mathematical reasoning or a cited numerical-analysis result, with its assumptions.
- **[U] Unverified mechanism:** plausible explanation or future research direction; not an experimentally established cause.

The repository contains several generations of methods. Results for the original WENO coefficient-correction network, Keras NN-TENO5, historical JAX blends, and V2 gated NN-TENO5 must not be combined into one model's performance record. Paths and line numbers below refer to this HEAD unless a commit is specified. JSON records are additionally identified by case/policy keys.

## 1. Executive summary

1. **Euler admissibility failures were real, including invalid intermediate RK stages.** The committed oracle-trained NN-TENO5 reproduces `Non-positive Euler density at step 6` on Sod, N=120, CFL=.35 [R]. Later V2 validation records negative pressure on random Riemann case `euler-validation-000088-1023461505243801251` at stronger neural corrections [A]. A fresh trace first encounters negative pressure at **step 13, SSPRK3 stage 1**, for soft strength .75; strength .25 completes all 15 steps [R]. The selected V2 model also fails the closed-wall Woodward–Colella blast test at N=128/256, while classical TENO5 completes it [A]. These are numerical failures, distinct from physical shock formation.

2. **A stable, conservative trajectory can still be less accurate or more oscillatory.** Historical short-rollout successes did not consistently transfer to canonical long-time problems. In later prospective evaluation, the selected model improves several Riemann/Sod errors but worsens the N=192 contact error by 6.87% and long-time scalar advection error by 12.65% [A]. On N=192 Sod, mean conserved-state L1 improves from .00265136 to .00253802 while density TV increases and a small density undershoot appears [A]. Neither finiteness, conservation, nor smaller L1 alone establishes nonlinear stability.

3. **Some important limitations belong to implementation and discretization, not the neural model alone.** The old WENO7 coefficient layer produces 10 non-finite outputs on the stored 100,000-row normalized dataset with epsilon=1e-6, versus zero with 1e-4 [R]. Separately, reconstructing `F(cell-average U)` using cell-average reconstruction coefficients gives only about **second-order general smooth Euler consistency**, including with classical TENO5 [A/R]. An experimental characteristic-state reconstruction plus Rusanov flux recovers approximately fifth spatial order on that diagnostic; shock robustness and positivity remain unvalidated.

4. **The stabilization attempts helped, but did not create a conservative positivity guarantee.** Curriculum learning, per-stage guards, pre-repair penalties, finite/magnitude-aware repair, gradient clipping, and classical/neural blending were all attempted. Historical records describe progress from failed training to finite Euler checkpoints [H]; later artifacts show that repair can add mass and energy [A]. The repaired training dynamics differ from unguarded evaluation. Small convex blends and a deterministic TENO trouble gate are useful controls, but neither proves Euler positivity or entropy stability.

5. **Several earlier causal claims exceed their evidence.** A fixed-timestep dissipation sweep does not prove Roe-basis ill-conditioning caused the blowup. Near-neighbor stencil conflicts suggest insufficient information but do not prove a target is not a function of the inputs; the supplied audit also compares normalized inputs against unnormalized targets. Constant-lambda sweeps do not rule out all spatially varying gates. Historical Shu–Osher improvements remain limited by reference uncertainty. These qualifications are important for an accurate research interview.

The defensible project claim is: **a modular learned reconstruction framework with demonstrable, configuration-dependent gains and an unusually useful record of stability, consistency, and validation limitations**. A universally superior, generally fifth-order, positivity-preserving neural Euler method is not established.

## 2. Numerical architecture

### 2.1 Repository scope and method generations

The initial inventory contained 1,330 non-Git/non-cache files, including 130 Python files, 529 JSON files, 343 NPZ archives, 128 PNGs, 54 PDFs, 22 CSVs, 22 logs, 15 pickles, 15 HDF5 files, and two slide decks. All 130 Python files were indexed and parsed; none had a syntax error. No C/C++ implementation or notebook was found in the working tree or the searched Git filename history. Forty-one reachable commits were available. Source, experiment scripts, data generators, result records, archived validation, plots, documentation, and relevant historical diffs were investigated. This was not a rerun of every experiment or a byte-by-byte scientific verification of every artifact.

| Generation | Neural decision | Principal files | Evidence boundary |
|---|---|---|---|
| Classical ENO/WENO/TENO | Divided-difference/smoothness-based selection; WENO3/5/7 and TENO3/5/7 implementations | `src/schemes/`, `src/core/` | Implementation availability does not establish every advertised order on general nonlinear FV Euler data. The function named `ENO3` chooses between two linear candidates on three points; its name alone is not an order verification. |
| WENO-NN, Keras | Corrects coefficients of the classical reconstruction, then projects their sum to one | `src/networks/WenoNetworks.py`, `src/schemes/weno{3,5,7}.py` | `SeemsGood3.h5` is the original reference model; rejected WENO7 candidates are mostly absent. |
| NN-TENO, Keras | Independent troubled probabilities for candidate stencils, thresholded to keep/discard | `src/networks/TenoNetworks.py`, `src/schemes/teno{3,5,7}.py` | Stored TENO5 imitation and oracle models exist; availability of order-3/7 builders is not cross-order generalization evidence. |
| Historical differentiable JAX | Soft training weights; hard inference; raw, fixed, maximum-probability, or learned blends | `src/differentiable/`, `scripts/unrolled_physics_loss/` | P0 canonical results use point-sampled states/references and historical policies. |
| V2 | Separate scalar/Euler backbones; deterministic TENO gate; soft or hard candidates; explicitly controlled strength | `gated_teno5_jax.py`, `reconstruction.py`, `Train_Gated_NNTENO5_Dataset_V2.py` | Exact cell-average corpus, K=15 training; calibrated soft strength .25. This is a different method generation from P0. |

### 2.2 Conserved state, equation of state, and characteristic basis

The one-dimensional ideal-gas system is

\[
 U_t+F(U)_x=0,\quad U=(\rho,m,E)^T,\quad v=m/\rho,
\]
\[
 p=(\gamma-1)\left(E-\frac{m^2}{2\rho}\right),\qquad
 F(U)=\left(m,\frac{m^2}{\rho}+p,\frac m\rho(E+p)\right)^T,
 \qquad \gamma=1.4.
\]

**Pressure is recovered consistently in the inspected physical solver paths.** `src/core/eulerEquations.py:25–34,55–77`, `src/core/SimulationClasses.py:66–77`, `src/analysis/euler_regression.py:8–12`, and `src/differentiable/euler_jax.py:36–96` use this same relation. The expanded flux formulas are algebraically equivalent. A fresh 256-state check found flux disagreement with the primitive form below 2.85e-14 and eigenvalue disagreement below 3.56e-15 [R]. There is no evidence here of a missing gamma factor or a specific-energy/energy-density confusion in those physical flux routines.

At each face, Roe averages give

\[
 \widetilde v=\frac{\sqrt{\rho_L}v_L+\sqrt{\rho_R}v_R}{\sqrt{\rho_L}+\sqrt{\rho_R}},\quad
 \widetilde H=\frac{\sqrt{\rho_L}H_L+\sqrt{\rho_R}H_R}{\sqrt{\rho_L}+\sqrt{\rho_R}},\quad
 \widetilde c^2=(\gamma-1)(\widetilde H-\widetilde v^2/2).
\]

The right eigenvectors are columns `(1,v−c,H−vc)`, `(1,v,v²/2)`, `(1,v+c,H+vc)`; the left matrix is computed by inversion. The **same face basis** projects all stencil entries for both LF directions and transforms the reconstructed flux back. The current basis is internally consistent: max absolute `L R−I` was 8.91e-16 on the fresh moderate-state test [R]. This does not measure its conditioning along a failing trajectory.

The current Euler path is **characteristic reconstruction**, not independent reconstruction of rho, momentum, and energy in the physical basis. A network acts separately on each characteristic field. No controlled component-wise-versus-characteristic neural Euler experiment was found. The Roe basis is used for decomposition; the default numerical flux is not the Roe approximate Riemann flux.

### 2.3 Actual spatial pipeline

The default Euler algorithm reconstructs **characteristic split fluxes**:

\[
 a_{i+1/2}=\max\{\max_j|\lambda_j(U_i)|,\max_j|\lambda_j(U_{i+1})|\},
\]
\[
 g^{\pm}_{i+1/2,k}=\tfrac12 L_{i+1/2}\big(F(U_k)\pm a_{i+1/2}U_k\big),
 \quad
 \widehat F_{i+1/2}=R_{i+1/2}\left(\mathcal R(g^+)+\mathcal R(g^-)\right).
\]

For the negative direction the stencil is shifted one cell and reversed. The one-face dissipation coefficient is constant across that face's reconstruction stencil, but is **only bounded using the adjacent cells**, not all stencil cells. This is an inspectable potential limitation, not proof that it caused a particular failure.

`src/core/eulerEquations.py:79–122` and `src/differentiable/euler_jax.py:166–254` implement this pipeline. `SimulationClasses.py:212–231` and `euler_jax.py:109–123` gather stencils. Transmissive ghosts repeat the boundary state; periodic ghosts wrap. The left transmissive boundary uses the physical flux of the first state, while the right uses the reconstructed last face. Thus boundary accounting must use the actual implemented numerical fluxes.

**There are no separately reconstructed physical left/right interface states in this default split-flux path.** The pipeline in this implementation is

`initial conserved values → face Roe basis → characteristic LF flux stencils → neural/classical reconstruction → one conservative numerical face flux → flux difference → RK update`.

The optional `get_euler_state_flux_jax` (`euler_jax.py:259–280`) instead reconstructs characteristic **states**, transforms each side to physical conserved states, and evaluates a Rusanov flux. Its face states must themselves be admissible. It currently supports periodic boundaries only and is experimental.

### 2.4 Reconstruction and neural architecture

For WENO5/TENO5 the three candidate polynomials are

\[
 q_0=\tfrac13 f_{i-2}-\tfrac76 f_{i-1}+\tfrac{11}6 f_i,\quad
 q_1=-\tfrac16 f_{i-1}+\tfrac56 f_i+\tfrac13 f_{i+1},\quad
 q_2=\tfrac13 f_i+\tfrac56 f_{i+1}-\tfrac16 f_{i+2},
\]

with optimal weights `d=(.1,.6,.3)`. WENO-JS uses `alpha_k=d_k/(epsilon+beta_k)^2`; TENO uses scale-separation values chi and cutoff 1e-5. JAX computes chi in log space (`teno5_jax.py:81–96`).

The original WENO-NN correction MLP consumes the classical coefficient vector, produces bounded corrections `0.5*tanh(.)`, and applies

\[
 c_j^*=\widetilde c_j+\frac{1-\sum_\ell\widetilde c_\ell}{s},
 \qquad \widehat f=\sum_j c_j^* f_j.
\]

The coefficient sum is one (`WenoNetworks.py:21–49`), which supplies constant reproduction. It does not impose the higher moment conditions for order 5/7, a physical invariant domain, or an entropy inequality. Spatial conservation comes from the shared flux difference, not from this coefficient projection.

NN-TENO5 uses `5→64→32→16→8→3` (3,155 parameters), ReLU hidden layers and independent sigmoid troubled probabilities. Each stencil is normalized by its own min/range. Keras treats `probability >= threshold` as discard; thresholds may be calibrated per candidate. JAX soft mode uses keep weights `1−probability`; hard mode uses a .5 threshold. Candidate weights are `d_k*keep_k`, renormalized; all-discard falls back to the stencil center (`teno5_jax.py:99–144`, `projection_jax.py:39–63`). This fallback prevents an undefined zero-weight division on finite inputs. It does not guarantee admissibility of the Euler update.

V2 computes

\[
 \mathcal R_{V2}(s)=\mathcal R_T(s)+G(s)\lambda\,[\mathcal R_{NN}(s)-\mathcal R_T(s)],\qquad
 G(s)=\mathbf1\{\min_k\chi_k(s)<10^{-5}\}.
\]

`gated_teno5_jax.py:19–67` makes strength zero exactly classical TENO. The gate operates separately on each LF sign/characteristic field. The neural branch is still evaluated on the full array before `where`; gate activation is not evidence of sparse-compute acceleration. The expanded feature builder in `src/data/feature_vector.py` exists, but the stored production backbones have **five inputs**, not the full proposed enriched feature vector.

### 2.5 Time stepping, CFL, and training

All principal Euler comparisons use SSPRK3, with `D(U)=(Fhat_right−Fhat_left)/dx`:

\[
 U^{(1)}=U^n-\Delta tD(U^n),
\]
\[
 U^{(2)}=\tfrac34U^n+\tfrac14\big(U^{(1)}-\Delta tD(U^{(1)})\big),
\]
\[
 U^{n+1}=\tfrac13U^n+\tfrac23\big(U^{(2)}-\Delta tD(U^{(2)})\big).
\]

The legacy adaptive driver recomputes `dt=min(CFL*dx/max|v±c|, remaining time)` at full-step starts (`SimulationClasses.py:84–102`); its logged CFL is based on that input state. It does not certify all subsequent stage speeds. Fixed-step alternatives also exist. The historical JAX Euler trainer uses N=64, assumed maximum speed 2, CFL=.5, and hence dt=.00390625, **independent of the sampled problem's actual speeds** (`Train_NNTENO5euler_UnrolledJAX.py:62–69`). P0 benchmarks choose a fixed timestep from initial speed and a safety factor. V2 training uses each case's own stored dt, 15 complete SSPRK3 steps; V2 prospective benchmarks again use a fixed initial-speed margin and measure actual stage CFL.

Current NumPy validates each substage and raises on nonphysical states. JAX guards are optional; training applies repair after every substage, while official unguarded diagnostics record invalid raw stages. **Invalid intermediate stages are possible even if the initial or final state is positive.** SSPRK3 inherits a forward-Euler invariant property only if the spatial forward-Euler map has that property under the applicable timestep restriction; that premise has not been established here.

Historical unrolled losses combine final-state mismatch, TV/BVD penalties, parameter anchoring, and later pre-repair positivity penalties. V2 Euler instead uses scaled conserved-state MSE plus positivity penalty and clipped Adam gradients; its Euler branch does not use the scalar TV term (`Train_Gated_NNTENO5_Dataset_V2.py:208–234`). Training trajectories may be repaired; validation/test are deliberately unguarded (`:183–194`).

## 3. Confirmed Euler stability problems

### 3.1 Oracle-local labels destabilized a coupled Euler rollout

**Symptom/configuration.** Oracle-trained Keras NN-TENO5 fails Sod, N=120, length=1, T=.2, adaptive CFL=.35, using calibrated thresholds from its metadata. The fresh check reproduced **`FloatingPointError: Non-positive Euler density at step 6.`** [R]. The historical record also reports scalar step-advection L1≈5.34e9 versus TENO5≈.0133 [H]; this scalar number was not rerun.

**Evidence.** `Documentation/Results.md:407–449`; `models/nn_teno5_oracle_metadata.json:60–114` records exact-mask accuracy .830772 and macro-F1 .642433; the checkpoint actually exists as `models/nn_teno5_oracle.h5`. `src/schemes/teno5.py:847–888` integrates it into Euler. Reproduction is specified in §10.

**Mechanism.** The oracle minimizes isolated interface reconstruction error over keep/discard masks on synthetic functions. That objective contains no stage-level density/internal-energy constraint and no control of repeated nonlinear amplification. A lower local oracle error is therefore compatible with a destabilizing selection sequence. Misclassification, the chosen label objective, and rollout distribution shift are competing contributions; a recorded per-face decision at the first bad stage would distinguish them [U]. This failure is evidence about the learned model–solver interaction, not a proof that all oracle supervision is unsuitable.

**Attempted fix/outcome.** L2 anchoring to the TENO-imitation model, strengths 1–500, reportedly kept the tested Sod/advection runs bounded, but none improved the original model across the six-case suite; Shu–Osher remained worse [H]. The anchored sweep checkpoints/raw logs are absent, so this is a reported mitigation rather than a verified guarantee. **Confidence:** high on the Sod failure; medium on the historical anchoring outcomes; moderate on the broad mechanism, low on any unique first-face cause.

### 3.2 Historical unrolled Euler training: runaway states and non-finite parameters

**Symptom/configuration.** A scalar-pretrained reconstruction inserted into Euler produced runaway states under a clamp-only guard. The stored `models/euler_instability_trace.csv` gives:

| LF multiplier | Stored behavior; CSV step is zero-based |
|---|---|
| 1 | maximum density ≈4.81e3 at step 8, 1.94e37 at step 10; non-finite at step 11 |
| 1.5 | delayed growth, but maximum density reaches 6.21e18 at step 14; the saved 15-step trace ends there |
| 2, 3, 5 | non-finite at step 3; preceding maximum density ≈8.50e20, 1.88e18, 1.34e43 respectively |

**Evidence.** CSV lines 1–41 and `plots/04_euler_instability.png` [A]; `EulerInstability_TraceCapture.py:53–114` freezes the old clamp and varies alpha at the **same** dt/dx. `Documentation/Results.md:474–557` describes earlier failures and subsequent changes [H]. `models/nn_teno5euler_unrolled_jax_params.pkl` survives as a non-finite checkpoint; the strict loader now rejects it (`model_io.py:49–71`).

**Root cause/competing explanations.** Confirmed: the evolving neural/solver combination was unstable, and clamping positivity was insufficient. Unproven: Roe-basis ill-conditioning was the initiating cause. The CSV contains no eigenvector condition numbers, pressure minima, raw stage states, or effective CFL. The fixed training timestep is not a verified bound for the random pressure/velocity range. Multiplying LF alpha without reducing dt can itself violate an explicit stability restriction; it also changes the network's split-flux inputs. Thus the non-monotonic alpha experiment does **not** rule out a wave-speed/timestep contribution. The textual 5.7e144 maximum concerns another reported trace quantity; it is not the saved CSV's maximum-density datum.

**Attempted fixes.** Reject invalid sampled Riemann targets; guard every RK stage rather than just full steps; increase the clamp floor from tiny values to .05 and cap velocity; curriculum K=2→4→8→15; correct BVD to compare neighboring one-sided reconstructions; measure positivity before repair; detect non-finite/out-of-scale rows; cap penalty values; clip gradients; later blend with classical reconstruction. The committed records describe curriculum progress from K=4 failure, to K=8 failure, to completion of all stages [H]. Several mechanisms changed together, so their individual causal effects are incompletely isolated.

**Outcome.** Finite later checkpoints and successful unguarded canonical runs exist [A]. Training becoming finite does not demonstrate a stable conservative training scheme. The trace script imports the current sampler, whose smooth-family mixture changed after the original experiment; reproducing the *historical* trace exactly requires the corresponding sampler revision, not merely the frozen guard. **Confidence:** high on recorded runaway behavior; medium on historical training chronology; low on a uniquely identified eigenbasis cause.

### 3.3 V2: negative pressure at stronger neural corrections, including raw RK stages

**Observed configuration.** Euler checkpoint SHA-256 `dbae82f977bc5b07b0fb7671971696e69f44541dad450474fc575813f22bb7b5`; random Riemann case `euler-validation-000088-1023461505243801251`; N=128; dt=.0017428079661349484; T=.026142119492024225; 15 steps; transmissive boundaries. `full-validation/failure_inventory.json` records:

| Policy | Minimum recorded stage pressure | Invalid stage rows | Final non-finite? |
|---|---:|---:|---|
| soft, strength .75 | −.0570911 | 8 | no |
| soft, strength 1 | not recoverable from null/NaN aggregate | 22 | yes |
| hard, strength .5 | −.00872765 | 2 | no |
| hard, strength .75 | −.152506 | 11 | no |
| hard, strength 1 | not recoverable from null/NaN aggregate | 21 | yes |

The hard strength-1 policy also fails case `euler-validation-000242-5114025453019182078` at N=256, with a recorded minimum density −.0364831 [A]. Six failed method–trajectory combinations involve **two independent IVPs**, not six independent problems. In contrast, selected soft strength .25 has no invalid stages across the 307 Euler validation trajectories; minimum stage density .0954547, pressure .0835742, maximum measured CFL .334033 [A].

**Fresh backward trace.** A separate CPU float64 trace stopped immediately at the first invalid raw substage, before passing it into another flux calculation [R]:

| Mode/strength | First invalid point, one-based | Density minimum | Pressure minimum |
|---|---|---:|---:|
| soft/.25 | none over 15 steps | positive throughout | positive throughout |
| soft/.75 | step 13, stage 1 | 1.21304 | −.00380485 |
| soft/1 | step 11, stage 1 | .959311 | −.0304347 |
| hard/.5 | step 15, stage 1 | .718730 | −.00872765 |

The archived direct repair study likewise sees failure for soft/.75, but its unguarded final state is non-finite, unlike the batched full-validation record. **This numerical-path discrepancy is preserved, not averaged away.** Both reject the policy by raw-stage admissibility. Fused/batched execution and threshold sensitivity are possible explanations, requiring a controlled comparison; no unique cause has been established.

**Mechanism.** Reconstruction changes the conservative face flux, which changes the stage's energy/momentum balance until `E−m²/(2rho)` becomes negative while rho is still positive. The next speed calculation then leaves the physical domain. The first-invalid trace localizes the failure to a forward-Euler stage, but does not yet identify a single erroneous mask, spatial face, or characteristic field. The archived hard/.5 failure has maximum effective CFL≈.31187, and soft/.75≈.29367: calling the failure simply “CFL too large” is unsupported without a method-specific bound and a dt refinement experiment.

**Fix/outcome.** Strength calibration rejects failing policies and chooses soft/.25; `reconstruction.py:45–51` keeps soft/hard semantics explicit. This works on this validation corpus, not universally (§3.4). **Confidence:** high for loss of admissibility and correction-strength sensitivity; moderate for a structural lack of invariant-domain control; low for a unique classifier error cause.

### 3.4 Selected V2 fails the closed-wall blast stress test

**Symptom/configuration.** Woodward–Colella pressure states 1000/.01/100 with jumps at .1/.9, rho=1, v=0, N=128/256, T=.038; closed walls; fixed initial-speed factor 2 and nominal CFL=.15. The selected soft/.25 model becomes non-finite at both grids. The short T=.001, N=128 smoke already fails. Classical TENO5 completes both full-time grids with admissible raw stages; its minimum pressure remains .01 and maximum measured CFL≈.1076 [A]. The P0 learned blend completes N=128 but fails N=256, so it is also not a general remedy.

**Evidence.** `docs/research_release/results/wall-smoke/benchmark.json`, `wall-closure/benchmark.json`, and their per-method JSON/NPZ files; named rows `v2-selected-soft`, `teno5`, `p0-historical-learned`. Initial state and fixed time-grid definitions are `v2_benchmark.py:69–87`. These configurations assess **stability**, not a certified blast accuracy ranking: no full-time accuracy reference was supplied after the failed smoke. “Selected” and “trained” V2 Euler policies are identical here and are not independent corroborations.

**Root cause/competing explanations.** Confirmed: failure to transfer to a much stronger, reflective, interacting-wave configuration outside the V2 corpus. Strong shock ratios, boundary interaction, local LF speed choice, learned mask errors, and lack of positivity limiting are competing contributors [U]. The corrected wall contract does not fix this instability. No archived first-failure face/stage attribution was found, so it is premature to identify the wall or the backbone alone as the origin.

**Attempted fix/outcome.** Physical wall fluxes were corrected independently (§3.7). Smaller fixed CFL/speed safety was used for the declared stress test; no post-failure tuning is claimed. The selected method still fails. Applying the training repair with a universal magnitude cap of 1000 would itself alter legitimate blast states, whose initial E=2500, and cannot count as a physical repair. **Confidence:** high on the failed generalization/stability test; low on a unique mechanism.

### 3.5 Repair concealed violations and altered conserved quantities

**Symptoms/evidence.** Three distinct implementation problems must be separated:

1. **Historically inert penalty [H/I].** Positivity was originally penalized only on an already-repaired final state; violations removed by repair were invisible to that loss (`Documentation/Results.md:503–539`). Current `numerics_jax.py:179–233` accumulates penalties before each repair.
2. **Historical repair changed admissible low states [A].** P0's saved before/after study (`validation/p0-2026-10-14/repair-assessment/repair_assessment.json`; `Documentation/P0_Validation.md:122–153`) shows a low-positive entropy wave's error changed from ≈6.20e-10 unguarded to .0466453 under the old repair, adding ≈.0400 integrated mass and .0999 energy. Commit `dcb779b` restricts flooring to still-invalid rows. Current low-positive controls remain unchanged [R].
3. **The remaining repair is nonconservative when activated [A/I].** For the V2 failing N=128 case at soft/.75, one repaired stage row adds integrated mass **.001611149** and energy **.000888843**. Hard/1 repairs three rows, adding .004317693 mass and .002987223 energy (`results/repair/repair.json`). The physical balance error equals those additions, while accounting for repair as an artificial source restores the bookkeeping residual to roundoff. This is not physical conservation restored by repair.

**Mechanism.** Replacing an isolated cell by the average of valid neighbors can yield a valid state because the admissible Euler set is convex, but the replacement generally changes the sum of conserved variables. The current repair also uses periodic `roll` neighbors even for a transmissive training problem (`euler_jax.py:431–445`), and floors/caps residual invalid clusters. This is a computational survival device, not a conservative positivity-preserving discretization.

**Additional defect verified in this audit [R/I].** The current penalty is zero at exactly zero pressure, although the repair and diagnostic reject that state. It also gives zero for a positive out-of-scale state that repair changes: input `(rho,m,E)=(2000,0,2.5)` is clipped to `(1000,0,2.5)` with penalty zero. See `euler_jax.py:340–348` versus `:421–460`. This disproves a literal “every intervention is penalized” interpretation; there is **no evidence this particular mismatch caused an archived training failure**. Saturation/non-finite substitution also limits useful gradients; penalty finiteness alone is not gradient control.

**Outcome.** Pre-repair penalties and the P0 admissible-state fix address specific defects. Conservative, boundary-aware admissibility control and complete training intervention logs remain absent. **Confidence:** high for measured nonconservation and the current penalty mismatch; medium for their contribution to historical optimization.

### 3.6 WENO7 coefficient singularities: a reproducible software/numerical defect

**Symptom/configuration.** The historical account describes NaN coefficient outputs and failed Euler candidate evaluations within 1–3 steps across five seeds. Commit `753d526` changes `_weno7_coefficients` epsilon from 1e-6 to 1e-4 [H/I]. On the **stored** `data/eulerAvgsExact7.csv`, using local normalization and the legacy Keras/TensorFlow backend, a fresh forward evaluation found **10/100,000 non-finite coefficient rows** with the old epsilon and **0/100,000** with the new one [R]. No model training was needed.

**Evidence.** `src/networks/WenoNetworks.py:106–137`; `Documentation/Results.md:174–192`; the commit diff; §10 reproduction. The full historical five-seed failure logs and 200,000-pattern stress-test outputs are not retained, so those counts remain historical claims.

**Root cause qualification.** The failure and the epsilon mitigation are reproducible. The comment's explanation “square(epsilon+beta) underflows, then alpha/total=0/0” is not a reliable floating-point derivation: `(1e-6)^2≈1e-12`, far above float32's smallest normal ≈1.175e-38; zero denominators would tend to produce infinite alpha and potentially Inf/Inf. The expanded WENO7 beta polynomials contain large subtractive terms. An instrumented near-flat bad-row stencil yields a negative computed beta≈−9.54e-7, consistent with cancellation error. Exposing extra intermediates changes the TensorFlow graph and can change the arithmetic, so this probe does not isolate the exact operation that produced every original NaN. A cancellation-sensitive denominator/graph is a stronger candidate than blanket epsilon-squared underflow [U].

**Fix/outcome.** Epsilon increase removes the observed non-finite rows and historical reports say Euler candidate runs complete afterward. A stable sum-of-squares/shifted evaluation, nonnegative beta treatment, precision study, and accuracy impact of epsilon would be a stronger resolution. The larger epsilon was not shown to establish seventh-order neural accuracy or global Euler stability. **Confidence:** high on the defect and sampled mitigation; moderate on cancellation as mechanism; low on the original underflow narrative.

### 3.7 Mirror-only neural wall boundary violated the physical flux contract

**Symptom/evidence.** Mirroring ghost momentum did not force a learned reconstruction to yield zero wall mass/energy flux. With the selected V2 model and the exact random-state recipe in `test_v2.py:99–109`, this audit measured left wall mass/energy fluxes **−.00614857 / −.0453992**, and right **−.00144767 / −.00504035**, for `boundary='reflective'` [R]. This is a constructed boundary-contract check, not a measurement of leakage in the original unfinished blast run.

**Root cause.** The network need not respect the field/sign/reflection symmetries required for cancellation of independently reconstructed LF branches. Interior flux differences can conserve the implemented boundary budget while the boundary budget itself violates an impermeable wall [I/T].

**Fix/outcome.** `euler_jax.py:220–237` adds `reflective-wall`, explicitly setting wall mass and energy fluxes to zero and retaining momentum flux. The same probe gives exact zeros [R]; the dedicated regression checks three-step closed-wall mass/energy balance. Commit `279434d` records the correction. This fixes the physical boundary contract; it does not imply correct wall pressure or cure §3.4. **Confidence:** high.

### 3.8 Increased variation and overshoots; accuracy and oscillation control conflict

**Observed.** Original WENO5-NN Shu–Osher, N=300, T=2, has visibly higher density TV than WENO5, approximately 16.5 versus 14.2 at the end (`plots/shu_osher/total_variation.png`; `Documentation/Results.md:105–125`) [A/H]. The WENO7 rejected-candidate Sod TV plot rises transiently toward .881 around T=.13–.15 while WENO5 stays near .875–.876 (`plots/sod_weno7nn/total_variation.png`) [A]. Historical seed/fine-tune accounts report a tradeoff between lower regional L1, Sod TV, and Shu regional Linf [H].

**Stronger later evidence.** N=192 prospective Sod, selected V2 versus TENO5: mean-state L1 **.002538015 vs .002651361**, density TV **.87975619 vs .87746633**, density undershoot relative to the exact final range **.000350685 vs 0**. `results/independent/independent-sod-192/{v2-selected-soft,teno5}.json` and the corresponding NPZ profiles retain the measurements [A].

**Interpretation.** Sod's small undershoot is a concrete nonphysical range excursion. Shu–Osher contains genuine shock-generated entropy-wave oscillations: higher TV there may mean less damping, spurious amplification, phase error, or a combination. TV alone cannot distinguish these. A narrower monotone jump of the same amplitude does not by itself increase its total variation. The older description that sharper jumps necessarily explain greater TV is therefore incomplete. Neither TV plot alone establishes catastrophic instability.

**Attempts/outcomes.** Boundary-data enrichment, exact-target training, wider stencils, loss reweighting, anchoring, smooth-data mixing, and blend/gate variants changed these tradeoffs without a uniform improvement. Later soft/.25 gating improves several L1 results while retaining a measurable variation cost. **Confidence:** high for plotted/recorded variation differences; high for the later Sod undershoot; moderate/low for identifying which historical Shu oscillations were spurious.

### 3.9 General nonlinear FV Euler consistency was approximately second order

**Observed diagnostic.** For a smooth periodic state with variable rho, velocity, and pressure (`v2_nonlinear.py:11–23`), the historical split-flux path has last spatial L1 orders **2.01197 / 2.00831 / 2.01805** for classical TENO5 / gated-soft / gated-hard. The experimental state-flux path gives **4.99993** for all three on the recorded grid ladder [A]. Fresh N=40/80/160 tests reproduce last orders 2.06219 versus 4.99864 for the classical reconstructions [R].

**Mechanism.** Treating `F(Ubar_i)` as though it were the cell-average flux confuses nonlinear evaluation and averaging. For smooth U,

\[
 \overline{F(U)}-F(\bar U)
 =\frac{\Delta x^2}{24}F''(U)[U_x,U_x]+O(\Delta x^4)
\]

under the usual smoothness/Taylor assumptions. Applying cell-average face-reconstruction coefficients to the wrong averaged quantity introduces a lower-order term. This is not a pressure-formula bug and not uniquely a neural failure. A constant-pressure/velocity entropy wave has flux linear in density and can pass a fifth-order test while concealing the general nonlinear limitation.

**Fix/outcome.** Experimental state reconstruction evaluates physical flux at reconstructed states and reaches fifth order on this instantaneous diagnostic (`euler_jax.py:259–280`). It is not yet shock-/positivity-validated or retrained. This is specifically a **finite-volume cell-average consistency finding**: an appropriately formulated finite-difference WENO method can instead start from pointwise physical fluxes. Establishing that different interpretation would require its own data semantics and consistency derivation; it does not validate the present FV experiment. **Confidence:** high on the measured FV limitation and its leading mechanism.

### 3.10 Optimization defects and configuration changes: what is and is not confirmed

The historical record reports outlier characteristic-flux magnitudes up to 562 versus typical magnitudes near 23, loss stagnation, and subsequent restriction of Riemann samples and a magnitude bound 30 (`GenerateTrainingData_EulerExact.py:131–145,318–328`; `Results.md:150,175`) [H/I]. This is a reported data/optimization problem; the original row-level loss/gradient evidence is not preserved. Raw magnitude alone does not prove MSE domination after local normalization: small input range can also amplify a normalized target. Exact normalization and tail gradients should be checked before assigning a unique mechanism.

A later 50/50 entropy-wave/Riemann curriculum reportedly ended with loss spikes around 1.5e10–2.4e10 and a corrupted final checkpoint without clipping; global-norm clipping 1.0 retained a finite checkpoint (`Results.md:561–580`) [H]. Current trainers implement clipping. The spikes and per-update gradient norms are not preserved as a complete Euler training log. Thus “large destructive updates were reported and clipping was introduced” is supported; a measured exploding-gradient history is not available. The first richer-gate shape error was caught before training (`Results.md:705–707`), and two later smoke failures are JAX tracer/NumPy boolean errors, not Euler instability (`results/sweep-smoke/failure.log`, `input-mismatch/failure.log`).

**Confidence:** medium for the historical optimization episodes, high for the current sampling/clipping code and preserved smoke exceptions; low for a unique gradient-level explanation without the missing logs.

## 4. Generalization challenges

### 4.1 Demonstrated or reported transfer limitations

| Transfer axis | Evidence and limitation | Status |
|---|---|---|
| Initial-condition family / physics | WENO5 reference NN is favorable on periodic step advection but less favorable on Shu–Osher; original scalar unrolled model wins some short similar cases and loses canonical advection/Burgers. | Historical metrics/plots [A/H]; not evidence that all scalar-trained models fail every Euler case. |
| Longer horizon | Historical Euler Phase 13 reports short K/3K wins, then mean-state L1 ratios 1.29 on full Sod and 1.24 on Shu–Osher; smooth subregions were worse. | [H], with later checkpoints/benchmarks but no complete saved original short-vs-long matched table. More recent prospective runs give a stronger independent long-time test. |
| Smooth vs discontinuous | Historical smooth mixing reportedly narrows Sod's rarefaction error ratio 1.49→1.27; P0 ungated neural/blend convergence is near third order rather than classical fifth. | Smooth-order artifacts [A]; regional phase-to-phase values [H]. |
| Shock strength / interacting waves | Selected V2 soft/.25 is valid on all 307 short validation IVPs yet fails strong closed-wall blast. | Confirmed [A]; strength, walls, interaction, horizon shift together, so isolated cause unresolved. |
| Resolution | Prospective contact ratio .999272 at N=96 becomes 1.06874 at N=192; long-time scalar square ratio .997485→1.12655. Sod improves at both grids (.942486/.957250). | Confirmed [A]; no universal “finer grid helps NN relative to baseline.” |
| Numerical policy | Same V2 Euler backbone: soft/.25 macro relative L1 .947938, hard/.25 1.023218. Stronger soft/hard policies eventually fail. | Confirmed [A]; soft training and hard inference are different numerical schemes. |
| Smooth critical points | Gated regular entropy waves approach fifth spatial order; critical-sin4 approaches fourth on tested grids, including classical TENO5. Ungated neural-soft is around third order. | Confirmed [A]; not a unique neural critical-point failure or an asymptotic theorem. |
| Scalar→Euler reuse | Scalar warm start reportedly immediately failed Euler unrolled training; later V2 deliberately trains independent backbones. | Failure details partly [H]; independent scalar/Euler weights [I/A]. Normalization does not make the PDE dynamics interchangeable. |
| WENO order | Separate width-3/5/7 model builders and a separately trained wider WENO7 pipeline exist. | No controlled cross-order transfer result. Shape-incompatible checkpoints cannot simply be reused. Wider context did not guarantee accepted Euler improvement [H]. |
| CFL / time integrator | Nominal P0 CFL .4 yielded actual stage CFL≈.57355 on Sod; the old alpha sweep is confounded. | Configuration sensitivity observed [A/H]; no systematic NN-vs-classical CFL stability map or SSPRK2-vs-3 study found. |
| Shock location / boundaries | Historical boundary-clipped stencil discrepancies; later held-out P0 Sod locations/times and V2 exact corpus vary initial problems. | Some successful variation tests [A]; no clean position-only invariance experiment. Reflection remains outside V2 training. |

### 4.2 What the later full-validation and prospective results establish

`results/full-validation/per_case.json` contains **11,064** method–trajectory rollouts: 615 scalar + 307 Euler validation IVPs, six strengths, two policies. Selected soft/.25 macro L1 ratios are **.976976 scalar** and **.947938 Euler**, with zero failed policies at this selection. Euler soft/.5 is also zero-failure but has worse aggregate ratio .964660. These are useful empirical results on this finite distribution; they are not positivity guarantees or estimates of deployment failure probability.

Macro scores average family×resolution groups; weighted scores answer a different question. Constants and tiny-error groups are excluded from ratios. Intervals are bootstrap summaries conditional on the evaluated cases, do not incorporate new training seeds/reference uncertainty, and do not correct for selection. The same validation set was used for the strength choice. Original V2 checkpoint selection used only up to two validation trajectories per family/grid (30 Euler, 36 scalar), despite metadata also recording the full split sizes. Later full-validation calibration did not retrain the models.

The prospective non-blast suite includes four new seeded Riemann IVPs and canonical/stress cases at paired grids. The two grids of an IVP are not independent trajectories. Numerical-reference Shu–Osher and Burgers interaction rankings remain unresolved. Analytic Riemann/contact/advection references are cell averages. The selected Euler method improves independent Riemann and Sod errors yet is not superior on every contact/grid configuration. These mixed results carry more weight than repeated tuning against the same canonical benchmarks.

### 4.3 Training-distribution coverage and feature representation

**Known coverage.** V2 has 6,144 independent analytic PDE problems: 4,096 scalar, 2,048 Euler, with disjoint trajectory splits and N=64/128/256 buckets. Its Euler families are constants, contact, random Riemann, Sod-like, and two-rarefaction cases. The three grid buckets contain separately generated problems, not necessarily three realizations of one identical IVP. Known absences include Shu–Osher interactions, boundary reflections, generic post-interaction dynamics, viscosity, chemistry, and multidimensional flows (`data/nnteno_v2/README.md:21–62`). The trainer reads the initial snapshot and 15-step target, not all four stored snapshots as four independent training trajectories (`Train_Gated_NNTENO5_Dataset_V2.py:120–156`).

**Normalization [I/U].** Per-stencil min/range scaling supplies invariance to positive affine rescaling of the input shape and avoids fixed raw scales in the MLP. It also removes amplitude/offset information from the neural decision and supplies no explicit grid spacing, timestep, Mach number, field identity, or LF branch identity. Near-constant stencils have an exactly guarded zero range, but arbitrarily small nonzero ranges can amplify roundoff into a normalized shape. These are possible distribution/conditioning issues, not proven causes of the recorded PDE failures. Fixed epsilon in the classical gate also need not have the same scaling invariance as the normalized MLP.

**Measured warm-up/runtime mismatch [A/I].** Original V2 Euler warm-up uses characteristic **states**, while the solver consumes characteristic positive/negative **split fluxes** (`Train_Gated_NNTENO5_Dataset_V2.py:238–266`). The saved training-data-only audit measures normalized marginal Wasserstein distances about .04549/.02768 for positive/negative flux inputs; teacher BCE .67622 on warm-up states versus .69319/.68924 on fluxes. Hard teacher disagreement rises .43714→.49922/.49790 (`results/input-mismatch-r2/input_mismatch.json`; aggregation described in `docs/research_release/RESULTS.md:129–133`). This shows a measurable mismatch after normalization. Different snapshot times and the use of exact-reference rather than neural-evolved states prevent causal attribution. A `--pretraining-inputs split-flux` ablation exists but no improvement from retraining was measured.

**Boundary coverage [H/I].** The historical WENO-NN audit reports 7 of the 15 largest discrepancies at the left boundary and a periodic scalar training source. Adding 10% boundary examples did not improve the targeted Shu regional L1 criterion. Thus missing boundary data was a plausible contributor, **not the established complete explanation** (`Documentation/Report_Results.md:104–194`).

### 4.4 Non-identifiability: evidence is suggestive, the historical proof is overstated

The historical five-point exact-target study reports nearly equal input stencils with targets differing by 2.3–4.5, failed five-seed improvements, and separation of many such pairs after widening to seven points (`Results.md:142–192`) [H]. Finite local averages indeed cannot determine the exact face value of every unrestricted subcell function [T]. A practical learned closure therefore needs an approximation class/distribution and cannot promise exact inversion of coarse graining.

However, the available experiment does not prove the particular continuous target is mathematically multivalued for the actual normalized model input. Near-neighbors are not identical inputs; steep target dependence, amplitude normalization, finite-reference error, and under-resolution are competing explanations. The dedicated supplied audit has a concrete confound: `PDEDataset_NonIdentifiability_Audit.py:197–210` bins **normalized inputs** but calls its spread calculation with **raw physical targets**. Affine copies can legitimately have equal normalized shapes and different raw targets, because runtime reconstruction rescales the output. Targets should be normalized by each input's own min/range before making that argument; constant ranges need separate handling. Quantization at one/two decimal places also merges distinct inputs by construction.

Earlier exact-target “cell averages” use a uniform endpoint sample mean, not discontinuity-aligned exact integration (`GenerateTrainingData_EulerExact.py:149–156`); fine Shu targets are interpolated numerical data. V2 subsequently provides substantially better controlled cell-average references. Thus “input insufficiency is plausible and motivated wider context” is defensible; “the stored neighbor audit proves training is impossible” is not. End-to-end rollout supervision changes the objective but does not restore information absent from a fixed local architecture.

### 4.5 Teacher imitation, structural constraints, and attribution

Teacher imitation supplies no direct objective for outperforming the teacher, although beneficial deviations can occur; it is too strong to say an imitator can *never* improve a PDE metric. The historical WENO5→WENO7 target change affects fewer than 1% of stencils above .01 in the reported diagnostic, which helps explain why aggregate MSE gave little improvement [H]. Oracle labels introduce another objective with its own stability risk (§3.1). The transition to unrolled losses was scientifically motivated, but short-horizon optimization still provides no long-horizon invariant-domain guarantee.

P0 attribution controls show that a small **constant** classical/neural blend already captures much of the claimed gain. Canonical Euler fixed blend is slightly better than learned blend under the historical reference. Scalar raw-neural advection error becomes ≈475 after training through the blend, despite the blended method performing well (`P0_Validation.md:104–120`). Therefore “the learned mixing head caused all improvements” is unsupported, and the raw backbone should not be advertised as a stable stand-alone solver.

Checkpoint names are also imperfect experiment identities. Fresh byte comparisons confirm that scalar `run4_widehorizon.pkl` and `run5_mixtrained.pkl` are identical, and Euler `curriculum.pkl` is identical to `phase15_smoothmix.pkl`. Distinct historical descriptions do not establish distinct surviving model states. Later comparisons should identify file hashes and actual reconstruction policies, not infer provenance from filenames.

Historical fixed-lambda sweeps show worsening Sod error over the **tested constant** coefficients, while Shu improves over part of that range. This supports a limitation of that backbone with those tested policies. It does **not** prove no spatially varying gate could exploit locally favorable corrections. A richer-gate attempt failing to beat Sod is another finite negative experiment, not an impossibility theorem.

## 5. Mathematical explanation

### 5.1 Why Euler is harder than the successful scalar tests

For linear advection with prescribed speed, the characteristic basis is trivial and independent of the solution. For Euler, the learned flux changes rho, momentum, and energy; these in turn determine pressure, sound speed, Roe averages, and the next set of characteristic inputs. Errors can therefore feed back through both reconstruction and the nonlinear physical decomposition. The entropy/contact mode and two acoustic modes must together produce a physical conserved state; plausible scalar characteristic values do not independently imply admissibility.

For an ideal gas the admissible set is

\[
 \mathcal G=\{U:\rho>0,\quad e_{\mathrm{int}}(U)=E-m^2/(2\rho)>0\}.
\]

It is convex because rho is linear and internal-energy density is concave on rho>0. This helps design limiters and explains why averaging valid neighbor states can repair an isolated invalid cell. It does not make such replacement conservative. The first variation

\[
 \delta p=(\gamma-1)\left(\delta E-v\,\delta m+\tfrac12v^2\delta\rho\right)
\]

shows why component errors can destroy pressure through cancellation even with positive density. Near small internal energy or high kinetic energy the relative pressure error can be large. Once pressure becomes negative, `sqrt(gamma*p/rho)` is undefined over the reals; zero/negative density also compromises divisions and the Roe square roots. These are concrete failure paths in this code. No claim is made that the historical experiments specifically sampled an astrophysical high-Mach regime.

Large `cond(R)` could amplify characteristic errors. Differentiating `L=R^{-1}` gives `dL=−L(dR)L`, which can also amplify training sensitivities. This is a plausible mechanism to measure. An accurate inverse test `LR≈I` on moderate states is compatible with poor conditioning on a different trajectory. Neither inversion in the source nor the alpha sweep demonstrates that such conditioning actually initiated a recorded blowup.

### 5.2 Conservation, consistency, positivity, monotonicity, and stability are different

| Property | What it means here | What the evidence supports |
|---|---|---|
| Conservation | One common numerical flux per face gives telescoping updates, with appropriate boundary exchange. | Strong for unguarded interior flux differences. Some invalid policies conserve their implemented budgets; repairs and incorrect physical wall fluxes require separate treatment. |
| Constant-state consistency | Equal stencil entries reconstruct that constant; flux reduces to the physical constant flux. | Implemented coefficient/candidate constraints and passing controls. This does not enforce all high-order moments. |
| Spatial accuracy | Truncation error decays at the expected rate for a specified interpretation of data. | V2 regular entropy waves approach fifth order; critical profiles and general nonlinear FV Euler expose limitations. |
| Positivity | Every relevant state has positive rho and pressure. | Observed on selected finite suites; violated by other policies and by blast. No theorem for the learned solver. |
| Monotonicity/TVD | Appropriate scalar maximum-principle or variation bounds hold under stated hypotheses. | Neither WENO nomenclature nor positive candidate weights proves this. No componentwise Euler maximum principle is expected in general. |
| Entropy stability | A consistent discrete counterpart of an entropy inequality holds. | No enforced Euler discrete entropy inequality or measured comprehensive entropy budget found. |
| Nonlinear stability | Perturbations remain controlled in an appropriate norm/time regime. | Completion/finiteness and selected-case errors are limited evidence, not a uniform bound. |

For SSPRK3 the integrated conservative budget over one step, without repair, is

\[
 M^{n+1}-M^n=\Delta t\left[\tfrac16(F_L^0-F_R^0)+\tfrac16(F_L^1-F_R^1)
 +\tfrac23(F_L^2-F_R^2)\right],\quad M=\Delta x\sum_i U_i.
\]

If repairs add full stage increments `r1,r2,r3`, their final budget contribution is

\[
 \Delta x\sum_i\left(\tfrac16r_{1,i}+\tfrac23r_{2,i}+r_{3,i}\right).
\]

These weights are correctly implemented in `diagnostics.py:98–133`. For transmissive boundaries, the total mass/momentum/energy need not remain constant. For reflective walls, mass/energy flux vanish, but momentum changes through wall pressure forces. Subtracting neither boundary exchange nor repair sources can give misleading “conservation errors.” Conversely, counting repair as a source is accurate bookkeeping, not proof of conservation of the original PDE.

### 5.3 Why reconstruction constraints and blending were insufficient

The TENO neural weights can be nonnegative and normalized while each candidate polynomial contains negative coefficients. A convex combination of candidate **polynomial values** is not a convex combination of admissible physical Euler states. In the default solver the candidates are characteristic split fluxes, not positive density/energy states at all. A scalar-looking probability constraint therefore does not solve the physical invariant-domain problem.

Likewise, a convex blend of classical and neural face reconstructions does not inherit positivity or TVD when one endpoint is not known to satisfy it. Even a pointwise flux blend needs a compatibility proof for the whole update, including spatially varying coefficients, shared faces, all RK stages, and CFL. The historical wording that blending has “no risk of instability” is contradicted by the later stronger-correction and blast evidence. Blending can reduce exposure to bad corrections; it is an empirical trust mechanism until constrained appropriately.

High-order accuracy also requires more than a coefficient sum. On smooth stencils the optimal WENO combination cancels lower-order candidate errors. Learned O(1) changes to those weights can leave O(dx^3) terms. A fixed small blend retains a nonzero multiple of that lower-order term asymptotically. This explains why P0 blends approach third order despite a small neural contribution. The V2 deterministic gate recovers the classical reconstruction wherever it stays off, which can restore the observed regular-profile order; it supplies no universal critical-point theorem.

### 5.4 Error propagation, CFL, and training gradients

For the fully discrete solver map `Phi`, a local linearized error relation is

\[
 e^{n+1}\approx D\Phi(U^n)e^n+\delta\Phi(U^n),
\]

where the defect includes reconstruction/model/discretization errors. A small instantaneous reconstruction error does not bound a product of amplification operators over many steps. The solver also encounters its own erroneous states during deployment, rather than just exact training snapshots. A K=15 final-state loss samples a much shorter composition than some canonical/stress runs.

The same composition appears in backpropagation, together with derivatives through normalization, eigenbasis inversion, and soft selection. Finite loss values and capped penalties do not bound the derivative of the rollout. Global gradient clipping addresses destructive parameter updates, but cannot itself make the forward numerical map admissible. Hard thresholding changes the forward map and removes most useful selection gradients; soft training/hard inference therefore needs a direct scheme-level ablation, not merely a classifier-accuracy comparison.

### 5.5 Connection to reliable numerical methods

**Established theory, with assumptions.** Conservative positivity-preserving FV/DG constructions can limit a polynomial toward its own admissible cell average. With admissibility at the required quadrature/face points, a suitable positivity-preserving numerical flux, and a sufficient forward-Euler CFL bound, cell-average positivity is retained; SSP composition transfers that property stage by stage. The limiter leaves cell averages unchanged. This does not apply automatically to the present characteristic split-flux neural method. See the authors' [Zhang–Shu review, §3](https://www.math.purdue.edu/~zhan1966/research/paper/review.pdf).

**A posteriori estimates, with assumptions.** A continuous or sufficiently regular space–time reconstruction can be inserted into the PDE to compute

\[
 \mathscr R=\partial_t\widehat U+\partial_xF(\widehat U).
\]

Entropy-relative stability can connect this residual and initial error to an error bound, subject to admissible compact state ranges, reconstruction regularity, flux assumptions, and controlled stability constants. Near shocks, reconstruction gradients/stability constants can make such estimates ineffective. The [Giesselmann–Makridakis–Pryer analysis](https://arxiv.org/html/1405.7616v1), especially its introduction and Remark 5.7, explicitly discusses this limitation; it is not a ready-made robust bound for all post-shock Euler solutions.

**Direct connection to Professor Krumscheid's work.** The verified 2026 preprint by Giesselmann, Kwon, and Krumscheid uses temporal reconstruction and SIAC spatial filtering for fully discrete RKDG convection–diffusion error estimation. It considers periodic Cartesian settings and entropy-compatible systems; its model analyses/experiments include linear advection–diffusion, nonlinear scalar equations, and a diffusive p-system. Nonlinear constants involve reconstruction Lipschitz norms. It does not supply a trained neural filter or establish this repository's Euler shock reliability. A learned admissible reconstruction checked by an independently evaluated residual is a **proposed extension**, not a result of that paper. [Primary preprint](https://arxiv.org/html/2604.01200v1).

For this project, the practical reliability options are:

| Limitation | Structural control / classical fallback | Independently computed diagnostic | How to test accuracy together with stability |
|---|---|---|---|
| Negative stages / blast failure | Learn a correction to a conservative face flux, constrained against a positivity-preserving low-order flux; or use a proved FV/DG scaling construction. Reject/recompute a stage with a valid classical method under its CFL. | Pre-repair rho/internal-energy margins at every stage; first invalid cell/face; fallback count. | Same dt/grid/reference, no hidden repair, all stages admissible, budgets satisfied, and errors/overshoot reported. |
| Nonconservative repair | Replace state overwrites by shared-face correction limiting or conservative recomputation; boundary-aware stencil handling. | Physical boundary budget and repair contribution recorded separately. | Include low-positive entropy waves and blast so fixed .05/1000 thresholds cannot masquerade as a guarantee. |
| Smooth/order loss | Preserve polynomial moments; turn correction off with provable smooth asymptotics; correct FV/FD semantics. | Semidiscrete exact-divergence errors, moment residuals, and dt-halving controls. | Regular, critical, and genuinely nonlinear Euler states, not only linear entropy waves. |
| Wrong candidate / distribution shift | Treat neural output as a proposal; retain deterministic TENO or limited low-order fallback where admissibility/entropy tests reject it. | Classical–neural flux difference, normalized correction size, decision margins, candidate disagreements, reference-free residual indicators. | False acceptance rate and accuracy cost of rejection, across independent families and strengths. |
| Physical boundary symmetry | Enforce wall mass/energy contract explicitly; use one boundary flux in all budgets. | Wall flux components and symmetry tests on random admissible states. | Separate wall correctness from stage admissibility and wall-pressure accuracy. |
| Training instability | Pre-repair penalties, finite parameter/gradient checks, conservative training dynamics, saved best checkpoints; gradient clipping as optimization support. | Gradient norms, per-stage corrections, rejected updates, finite checks, raw vs guarded loss. | Compare unguarded deployment with training dynamics; finite weights are necessary but insufficient. |
| Unresolved reference margins | Accurate analytic cell averages or independently converged reference ladders. | Spatial/time Cauchy changes and independent-solver comparison. | Claim a ranking only when uncertainty is small relative to the method margin; keep unresolved outcomes. |

Indicators such as TENO chi, BVD, model disagreement, or an entropy-residual sensor can be useful alarms, but are not by themselves a rigorous error bound. In particular, the discrete residual of a numerical solution inserted back into its **own identical update formula** is essentially zero by construction. Error estimation needs a separate reconstruction/PDE residual, including temporal and face contributions, and a valid stability argument. An entropy budget must include physical boundary entropy fluxes; a simple global decrease test would be wrong for general transmissive data.

## 6. Unresolved problems

Rank combines practical severity, scientific importance, and available evidence; “unresolved” includes a confirmed failure whose initiating mechanism is not isolated.

| Priority | Question | Severity / significance | Evidence strength | What would resolve it |
|---|---|---|---|---|
| 1 | Can the selected learned Euler scheme preserve an admissible conservative update on strong/out-of-distribution flows? | Critical; central reliability limitation. | Very strong failed-stage/blast artifacts. | Constrained correction plus independent unguarded stress validation; E1/E3. |
| 2 | What initiates pressure loss: correction geometry, CFL, local alpha, basis conditioning, or their interaction? | Critical; determines the correct remedy. | Failure is strong; unique cause weak. | First-invalid-stage/face trace with matched dt/alpha/basis controls; E1/E2. |
| 3 | What discretization is genuinely high order for general nonlinear FV Euler data? | High scientific significance; affects every accuracy claim. | Strong convergence evidence and fresh reproduction. | Extend the experimental state-flux variant to shock/positivity tests and controlled training; E4. |
| 4 | How much do training repairs bias learned dynamics and conservation? | High; current training/evaluation mismatch. | Strong controlled repair budgets; original per-update events missing. | Log all interventions and compare conservative/unguarded training; E1/E6. |
| 5 | Can gains transfer over family, resolution, horizon, and CFL? | High; some positive transfer, explicit counterexamples. | Strong recent finite-suite evidence; single backbone seed. | Predeclared factorial tests with independent seeds and separate model/policy selection; E5/E6. |
| 6 | Is the warm-up state/flux mismatch causally harmful? | Medium–high; concrete representation mismatch. | Strong measured mismatch, no intervention result. | Matched state-vs-split-flux pretraining ablation; E6. |
| 7 | Are short stencil targets intrinsically ambiguous, or are the audits confounded? | High research significance. | Historical neighbors suggestive; supplied normalization audit flawed. | Correct normalized-target/constant controls and controlled input-pair study; E7. |
| 8 | Are the reported Shu–Osher improvements larger than reference uncertainty? | High for claims of superiority; not an admissibility issue. | Strong evidence that some rankings remain unresolved. | Finish adequate spatial/time/independent-solver reference studies; E8. |
| 9 | Is WENO7 fixed robustly beyond the stored corpus, without accuracy damage? | Medium; old NaNs reproduced, empirical mitigation works. | Strong sampled defect/fix; exact floating-point origin unresolved. | Beta/denominator precision analysis and order/critical-point checks; E4. |
| 10 | Can an independently computable estimator remain useful at Euler shocks? | Major prospective research question. | No implementation or validated guarantee here. | Start with smooth FV/DG cases, then study shock-aware residual/error control; E9. |

No confirmed current pressure-equation inconsistency, unique eigenvector-ordering bug, or quantified component-wise-versus-characteristic benefit was found. Commit `daa861f` refactors characteristic transforms and dissipation; it does not, without a before/after physical experiment, prove that every earlier Euler difficulty came from a basis defect. Resource-stopped reference/convergence jobs (`status.json`, exit 130) are incomplete experiments, not numerical divergence or successful validation.

## 7. Proposed experiments

These are minimal **future** experiments; this audit only ran the lightweight checks in §10. Proposed training ablations require separate authorization. Preserve the existing corpus/checkpoints and write each experiment to a new output directory with source/configuration/model hashes.

### E1 — Localize the first admissibility failure and test a conservative fallback

**Reproducer.** Load the N=128 validation case specified in §3.3 from `euler_validation.h5`; use its exact stored dt/initial state and 15-step target. Baselines: TENO5, selected soft/.25, soft/.75, hard/.5, then the explicitly labeled historical repair. Run at dt, dt/2, dt/4 to the **same physical final time** with 15/30/60 steps.

**Diagnostics.** At every raw RK substage record minimum rho and internal energy, first invalid cell, adjoining face fluxes, LF signs/fields, candidate values/keep weights/chi, correction norm, stage max speed/CFL, `cond(R)`, and `||LR−I||`. Check parameter/input finiteness before inference. Count repair/fallback changes and their weighted integrated contributions. Stop a diagnostic trace before the first invalid state is used in another flux call; do not replace this by a final-state check.

**Intervention.** Separately test a shared-face limited correction to an admissible low-order FV flux, with an appropriate CFL and a documented acceptance rule. A rejected face must use the same accepted flux in both adjacent updates. Recompute necessary neighbor stages rather than overwrite a cell after its update.

**Success criteria.** Zero invalid raw stages and physical conservation residual near roundoff at the problem's scale; no hidden state repair; report L1/Linf, extrema, TV and intervention rate. Reduced dt establishing a stable range diagnoses time sensitivity but is not proof of general positivity. A fallback is useful if it prevents known failures while quantifying its accuracy/runtime cost.

### E2 — Separate dissipation, timestep, and characteristic conditioning

**Reproducer.** Preserve the exact original failing random state from the historical revision/sampler, and also use E1's saved IVP. Test alpha multipliers 1/1.5/2/3/5 with (a) fixed dt, reproducing the historical confound, and (b) dt reduced inversely with the multiplier. Add adjacent-cell, full-stencil, and global speed bounds without changing other variables. Use TENO5 and the same fixed neural checkpoint in each row.

**Diagnostics.** First-invalid measurements from E1, plus normalized neural input distributions and conditional basis errors on identical states. A frozen-basis test may help attribution but must be labeled as a diagnostic altered method. A component-wise variant can be an additional controlled ablation; it is not already validated here.

**Success criteria.** Establish which intervention postpones/removes the **first** violation and whether large condition numbers precede it. If reducing dt stabilizes high-alpha runs, the old sweep cannot establish an eigenbasis cause. If basis conditioning stays moderate before failure, revise that hypothesis. No scheme is accepted based on a delayed NaN alone.

### E3 — Isolate the blast generalization shift

**Reproducer.** Use `configs/v2/wall-closure-smoke.json` / the saved wall-smoke manifest, first N=128, T=.001, corrected walls, selected frozen model. Compare TENO5, V2 soft/.25, and P0 learned blend; optionally an admissible low-order FV control. Start with the existing short horizon, not a costly finest reference.

**Ablations/diagnostics.** Reduce only pressure contrast while holding boundaries fixed; separately test pre-boundary-interaction data with different boundaries; halve dt at constant physical time. Capture the first-invalid stage/face. Keep wall mass/energy flux exact, and record momentum flux separately. Never apply the fixed training magnitude cap to a physically valid E=2500 blast state as an unreported “fix.”

**Success criteria.** Recover admissible raw stages over a declared contrast range with conservation and correct wall contract, then extend T to .038 and N to 256. Only after stability is established build a convergence-controlled accuracy reference. This identifies a stability domain, not a universal proof.

### E4 — Verify discretization semantics, moments, and WENO7 numerics

**Reproducer.** Reuse `v2_nonlinear.nonlinear_spatial` at N=40/80/160/320 for classical, ungated neural, and gated policies under split-flux and experimental state-flux paths. Add the existing regular entropy wave and critical-sin4 profiles. Compare exact **cell-average flux divergences**, without time integration; later use common-time dt∝dx^(5/3) and dt/2 to control SSPRK3 error.

**WENO7 control.** Evaluate the existing epsilon values on stored normalized stencils and near-flat/jump patterns in float32/64. Record betas, eps+beta, alpha, normalized weights, and coefficient moments. Compare a numerically stable beta expression to the expanded polynomial before changing a solver.

**Success criteria.** Reproduce the baseline second-order diagnostic and demonstrate claimed order on general nonlinear data if adopting a new FV formulation. No non-finite coefficients; preserve constants and required moments. Shock admissibility/conservation must then be checked separately. A good entropy-wave rate alone does not pass this experiment.

### E5 — Test soft/hard mismatch and repeated-input sensitivity

**Reproducer.** Frozen V2 model on E1 plus a small predeclared random-Riemann subset; soft/hard policies at strength 0/.25/.5/.75. Compare scalar execution, batched/vmapped execution, and float64 compiler paths on exactly the same states before rollouts. Perturb input states at a controlled relative scale, retaining admissibility.

**Diagnostics.** Probability-to-threshold margins, TENO chi-to-cutoff margins, changed candidate sets, reconstruction/flux differences, first invalid stage, final error differences and conservation. Include cases where every candidate is rejected.

**Success criteria.** Characterize a reproducible tolerance envelope and flag policies whose admissibility depends on batching. Confirm the zero-strength identity. Classifier agreement alone cannot pass a scheme-level stability test. This specifically resolves the full-validation versus repair-study discrepancy in §3.3.

### E6 — Controlled generalization and warm-up ablation

**Design.** Separate training, policy-validation, and final-test trajectory IDs. Use at least three declared training seeds when training is authorized. Compare identical warm-up/optimizer budgets for characteristic-state versus actual split-flux inputs. Separately compare no repair, historical repair with every event logged, and a physically constrained conservative training solver; do not change all three at once.

**Minimal final matrix.** Use independent Riemann/contact initial conditions at N=96/192/384, shock location shifts relative to cell edges, held-out pressure ratios, and K/3K/long common-time horizons. Add one smooth variable-velocity/pressure case, one critical profile, and one shock–entropy interaction. Run a small CFL grid with the same time integrator before expanding integrator comparisons. Match reference semantics and time exactly across methods.

**Baselines/diagnostics.** Classical TENO5, no-network fixed blend, trained backbone with fixed blend, learned blend/gate, and constrained alternative. Report failure rate, family/grid macro scores, all component norms, correction/repair rates, TV/range excursions, and runtime. Keep constant/tiny-error controls in absolute units, not unstable ratios. Bootstrap independent IVPs, not stencil rows or repeated grids of one IVP.

**Success criteria.** A predeclared aggregate improvement with zero admissibility/conservation failures on the claimed domain, no severe hidden family regression, and robustness across seeds. A split-flux warm-up change is causal evidence only if all other factors match and PDE outcomes improve, not merely teacher BCE.

### E7 — Repair the input-ambiguity experiment

**Reproducer.** Begin with the stored five-/seven-point exact-target CSVs; keep trajectory IDs, category, LF sign, field, grid, min/range, target provenance, and quadrature error. Compare raw input/raw target and normalized input/**normalized target**. Treat constant stencils separately. Include affine copies of the same smooth function as a negative control: they should not create a normalized-target “conflict.”

**Diagnostics.** Independent-group nearest-neighbor pairs, distances in the actual network representation, target differences versus reference/quadrature noise, variation with bin width, and local error/loss. Use discontinuity-aligned cell averages and a controlled reference for any newly generated data. Compare width 5, width 7, explicit physical/context features, and a short history feature under equal training budgets only after the audit is valid.

**Success criteria.** Demonstrate persistent conditional target spread after normalization/reference controls and beyond what mere quantization explains. This supports a statistical closure limitation; an impossibility proof would require explicit identical coarse inputs with distinct valid underlying solutions/targets, not a near-neighbor plot alone.

### E8 — Resolve accuracy rankings and oscillation interpretation

**Reproducer.** Start with the saved P0 and V2 Shu–Osher reference ladders; add an independently formulated sufficiently resolved solver and temporal refinement. Preserve original point-sampled results as historical comparisons; evaluate FV methods against cell averages. Existing resource-stopped finest-reference jobs remain incomplete until actually completed.

**Diagnostics.** Spatial/time Cauchy changes, error margin, shock-aligned density wave amplitudes/phases, overshoots/undershoots, and TV against a common reference. Inspect smooth-region spectra only with a declared sampling/window and converged reference; such spectra were not established by the present TV plots.

**Success criteria.** Stable ranking and reference uncertainty small relative to the margin; the existing proxy criterion uses twice the sum of latest spatial/time changes below 25% of the method margin. That remains an empirical criterion, not a rigorous bound. A slightly smaller L1 accompanied by larger undershoot must be reported as an accuracy/variation tradeoff.

### E9 — Prototype independent reconstruction-based error control

**Minimal start.** Use scalar advection and a smooth Euler entropy wave or smooth p-system in a small FV/DG setup before post-shock Euler. Baselines: classical reconstruction/SIAC, unconstrained learned reconstruction, moment-constrained learned reconstruction. Build a separate space–time reconstruction and evaluate its continuous PDE residual by independent quadrature/automatic differentiation, including face/time contributions where necessary.

**Diagnostics/success criteria.** Exact error, estimator size, effectivity, convergence rate, reconstruction admissibility, boundary budgets, and online evaluation cost. Require the estimator to remain meaningful across unseen profiles rather than merely correlate with training error. For any rigorous claim, identify and verify the applicable entropy/regularity/state-range assumptions. Shock tests should initially be reported as empirical estimator studies; adapting the theory to robust post-shock Euler control is a research objective.

## 8. Interview preparation

### 8.1 A three-minute explanation

*Approximately 415 words; deliver at a measured pace.*

“My project combined classical ENO/WENO reconstruction with learned decisions, and the main lesson was that local reconstruction quality did not automatically give a reliable Euler solver.

In the Euler implementation I project a stencil into a face-based Roe characteristic basis, split the flux with Lax–Friedrichs, reconstruct each characteristic split flux, transform back, and update conservative variables with SSPRK3. This preserves the shared-face conservative form. But it does not by itself preserve positive density or pressure.

I encountered several different problems. One was a concrete implementation defect: the WENO7 coefficient layer produced non-finite outputs on near-degenerate stencils. Increasing its regularization epsilon removed those failures on the stored dataset. Another was a learning–solver interaction: a TENO network trained on locally optimal oracle masks failed the Sod simulation with negative density at step six. So even a meaningful local label objective was insufficient for the coupled time evolution.

The unrolled Euler training also had runaway states. We tried shorter rollout curricula, guards at every Runge–Kutta stage, positivity penalties measured before repair, gradient clipping, and blending with classical reconstruction. Those changes eventually produced finite checkpoints. However, the neighbor repair changed conserved quantities, and I would not describe it as a conservative positivity-preserving method. Also, the old dissipation sweep did not isolate the root cause: we increased dissipation without compensating the timestep, so Roe-basis conditioning remains a hypothesis rather than a demonstrated explanation.

Generalization was similarly mixed. Short random-Riemann training did not reliably transfer to long canonical runs or smooth regions. The later gated model improved several independent Riemann and Sod errors, but had a contact regression at a finer grid and failed a much stronger closed-wall blast test. Even on Sod, lower L1 error came with slightly increased density variation and an undershoot. I therefore separate accuracy, conservation, positivity, and nonlinear stability.

The audit also exposed a classical discretization limitation: treating flux evaluated at the cell-average state as an averaged flux gives only about second-order consistency on a general nonlinear smooth Euler test. That issue affects the baseline too, so it should be corrected before attributing every accuracy limitation to machine learning.

What I would pursue next is a learned correction inside a structurally constrained conservative method, with stage admissibility checks and a reliable fallback. I would combine that with an independently computed space–time reconstruction residual. DG reconstruction and SIAC postprocessing are attractive settings, but I would start with smooth problems where the estimator assumptions can be checked, and treat useful post-shock Euler error control as an open research question.”

### 8.2 Five challenging questions and evidence-backed answers

**1. “What precisely failed, and how do you know it was numerical rather than a physical shock?”**

The oracle Sod run produces non-positive density and raises at step 6 [R]. The V2 Riemann trace produces negative pressure in stage 1 before another flux call [R]. Physical Euler shocks retain positive density/pressure for these admissible problems. Separately, higher Shu TV is not enough to diagnose instability because the physical solution contains oscillations. Cite §§3.1/3.3/3.8 and show the actual failure trace rather than only a contour plot.

**2. “Does your characteristic projection solve the system-stability problem? Was the Roe basis wrong?”**

The current common face basis is algebraically consistent and passes inverse/kernel regression tests. Characteristic reconstruction helps separate wave families but does not enforce the nonlinear invariant set after recombination. I have no controlled component-wise neural comparison or measured conditioning trace that proves a Roe defect initiated the blowup. The fixed-alpha/fixed-dt experiment is confounded. I would measure condition numbers and first-invalid-stage fluxes under a matched-CFL ablation, as in E2.

**3. “If the scheme is conservative and the weights are convex, why can pressure become negative?”**

Conservation is a telescoping balance. It does not restrict each updated state's internal energy. The weights combine characteristic flux polynomial values, not admissible Euler states; the polynomial coefficients can be negative. `E−m²/(2rho)` can therefore become negative at an RK stage. Convex blending with an unconstrained endpoint does not create an invariant-domain proof. A positivity-preserving FV/DG construction needs an admissible cell average, appropriate reconstruction constraints, flux, and CFL at each stage.

**4. “What evidence really demonstrates generalization, and did you select on your test cases?”**

Some early canonical cases were repeatedly used for acceptance/model selection and should be called validation, not independent test evidence. P0 introduced predeclared new configurations; V2's later prospective suite froze models/strengths before evaluation. The exact corpus has trajectory-disjoint splits. Still, the training backbone evidence is primarily one seed, numerical-reference rankings remain unresolved, and the prospective suite contains explicit regressions and blast failures. I would report both successful and failed transfers and distinguish policy-selection uncertainty from a new-distribution test.

**5. “Why would I trust a learned error estimator or postprocessor more than your network's confidence?”**

I would not treat network confidence or candidate agreement as an error certificate. I would compute a separate PDE residual of a reconstruction, add initial/boundary/time terms, and use a stability result with checkable assumptions. The repository already shows why this separation matters: conservative finite outputs can still be inaccurate, and repair can conceal invalid stages. Moment-constrained learned DG/SIAC reconstruction could improve approximation, but the residual and admissibility checks should remain independently computable. Near shocks, useful relative-entropy estimates are not automatic; I would explicitly test that limitation rather than claim a global guarantee.

### 8.3 Three research ideas grounded in these findings

**Idea 1 — Learn admissible conservative corrections, with a posteriori rejection.** Let the network propose a face correction to a known robust low-order flux. Choose a shared correction factor using density/internal-energy constraints at every relevant stage, with classical recomputation if a proposed stage is rejected. The research question is whether the learned proposal improves resolution at the same admissibility/conservation standard and acceptable fallback cost. This directly addresses §§3.1–3.5; the exact limiter and its CFL assumptions still require design/proof.

**Idea 2 — Learn DG reconstruction subject to moment constraints; certify its residual independently.** Parameterize a local polynomial correction in a zero-mean basis so the cell average is preserved, impose polynomial reproduction conditions, then use admissibility limiting as appropriate. Evaluate the continuous space–time residual independently of the network. Compare estimator effectivity and error on unseen smooth solutions before shocks. This targets the local-objective/global-stability gap and the actual order loss, while using DG's accessible polynomial structure.

**Idea 3 — Learn a SIAC-type postprocessor inside an explicitly constrained family.** Preserve kernel unit mass and the required reproduction moments, or learn corrections in the nullspace of those moment constraints. Compare classical and learned filters with the same independently evaluated residual and a conservative/admissibility-aware acceptance rule. Signed high-order kernels need not preserve positivity; classical SIAC generally preserves global integrals in appropriate periodic settings, not every cell average. Therefore shock gating, boundary treatment, and any limiting must be analyzed separately, and can alter superconvergence. This is motivated by the verified RKDG–SIAC estimator work, but no learned SIAC method or theorem is implemented in this repository.

## 9. Presentation material

### Slide A: Generalization Challenges

**Take-home statement:** Gains depend on the IVP, horizon, resolution, and numerical policy; local accuracy and short-rollout validation did not establish universal transfer.

**Observed results**

- Original scalar-like successes did not transfer uniformly to Euler shock–entropy interaction; historical short-rollout gains also weakened at canonical horizons.
- Selected V2 soft/.25: full-validation Euler macro L1 ratio **.94794**, zero invalid stages on **307 short validation trajectories**. This is a selected validation score.
- Prospective N=192: Sod ratio **.95725**, contact ratio **1.06874**. Selected model fails closed-wall blast at both tested grids. Include the unsuccessful outcomes.
- Ungated neural reconstruction approaches third order on regular smooth tests; the deterministic gate recovers classical regular-profile behavior. General nonlinear FV Euler exposes a separate baseline consistency issue.

**Hypothesized mechanisms — label explicitly**

- Short-horizon/family coverage, missing boundary reflections and shock interactions, warm-up state/split-flux mismatch, and normalization removing context.
- Local stencil ambiguity is plausible; the archived normalized-input/raw-target audit does not prove it.

**Next experiment:** Frozen models, independent trajectory IDs, matched-time/CFL matrix; state-vs-split-flux warm-up ablation with multiple seeds.

**Repository figures:** Use [`docs/portfolio/figures/final_benchmark.png`](docs/portfolio/figures/final_benchmark.png) as the main panel: it already shows regressions, unresolved rankings, and failed blast tests. Optional inset: [`smooth_convergence.png`](docs/portfolio/figures/smooth_convergence.png). The older [`plots/03_validation_matrix.png`](plots/03_validation_matrix.png) illustrates the historical short-to-canonical gap but must be labeled as a different checkpoint/policy. Do not use [`plots/05_non_identifiability_audit.png`](plots/05_non_identifiability_audit.png) as proof of impossibility without fixing the audit's normalization confound.

**Suggested speaker note:** “The positive results are conditional observations. The later validation and prospective suite make the limits clearer; they do not erase the historical negative results.”

### Slide B: Euler Stability Problems

**Take-home statement:** Shared conservative fluxes and normalized neural weights do not enforce positive Euler internal energy at every RK stage.

\[
 p=(\gamma-1)\left(E-\frac{m^2}{2\rho}\right)>0,
 \qquad U^{(1)}=U^n-\frac{\Delta t}{\Delta x}\Delta\widehat F.
\]

**Observed results**

- Oracle NN-TENO5, Sod N=120: **negative density at step 6**, reproduced.
- V2 random Riemann N=128, soft/.75: **first negative pressure at step 13, stage 1**, reproduced; soft/.25 completes the 15-step control.
- Historical repair prevents some crashes but changes mass/energy; one repaired V2 stage adds **.001611 mass** and **.000889 energy**.
- WENO7 coefficient NaNs: **10/100,000 → 0/100,000** under epsilon mitigation, reproduced. This is a separate implementation defect.

**Hypotheses / limitations**

- Wrong candidate choices, timestep/dissipation interaction, or ill-conditioned characteristic transforms may contribute. The old alpha sweep does not isolate them.
- Classical blending/gating helps empirically but is not a positivity/TVD/entropy proof. Even lower Sod L1 has a small undershoot/TV cost.

**Next experiment / reliability direction:** First-invalid face/stage diagnostics, conservative positivity-constrained correction, matched-CFL fallback evaluation, and independently computed reconstruction residual.

**Repository figures:** [`plots/04_euler_instability.png`](plots/04_euler_instability.png) documents runaway density, but add a caption “fixed timestep; alpha=1.5 remains finite but enormous at the end of the stored horizon; eigenbasis cause unproven.” Its existing title is too causal. For a measured accuracy/variation contrast, use [`docs/portfolio/figures/euler_sod.png`](docs/portfolio/figures/euler_sod.png) with the TV/undershoot numbers from §3.8. Optional small panel: [`plots/sod_weno7nn/total_variation.png`](plots/sod_weno7nn/total_variation.png), labeled as a rejected historical candidate, not current V2.

## 10. Evidence index

### 10.1 Primary implementation and experiment sources

| ID | File/function and locations | Used for |
|---|---|---|
| S1 | [`src/core/eulerEquations.py`](src/core/eulerEquations.py), `roe_eigenbasis` 16–45, physical flux/speeds 55–77, `getEulerFlux` 79–122 | Correct EOS/flux, common face basis, adjacent speed bound, conservative/boundary fluxes. |
| S2 | [`src/core/SimulationClasses.py`](src/core/SimulationClasses.py), validator 66–77, adaptive/fixed driver 79–120, `stepItEuler` 138–154, stencil gathering 202–241; [`TimeSteppingMethods.py`](src/core/TimeSteppingMethods.py):19–32 | CFL behavior, per-stage rejection, SSPRK coefficients, boundaries. |
| S3 | [`src/analysis/euler_regression.py`](src/analysis/euler_regression.py):8–28,47–164,183 onward; [`InitialConditions.py`](src/initial_conditions/InitialConditions.py), Sod/Shu functions | Pressure extraction, adaptive benchmark helper, exact Riemann reference, initial conditions. |
| S4 | [`src/schemes/weno5.py`](src/schemes/weno5.py):13–52,96–157; [`weno3.py`](src/schemes/weno3.py), [`weno7.py`](src/schemes/weno7.py), [`ENO3.py`](src/schemes/ENO3.py), [`Hook5.py`](src/schemes/Hook5.py) | Candidate reconstructions, normalization, neural wrappers, available orders. |
| S5 | [`src/networks/WenoNetworks.py`](src/networks/WenoNetworks.py):21–49,106–141 | Coefficient correction and consistency constraint; WENO7 epsilon defect. |
| S6 | [`src/networks/TenoNetworks.py`](src/networks/TenoNetworks.py):22–75; [`src/schemes/teno5.py`](src/schemes/teno5.py):575–732,847–888; [`teno_common.py`](src/schemes/teno_common.py) | Keras classifiers, thresholds, local normalization, center fallback. |
| S7 | [`src/differentiable/euler_jax.py`](src/differentiable/euler_jax.py):36–123,166–280,340–466 | Physical pipeline, walls, experimental FV flux, penalty/repair. |
| S8 | [`src/differentiable/numerics_jax.py`](src/differentiable/numerics_jax.py):42–115,179–233; [`diagnostics.py`](src/differentiable/diagnostics.py):15–31,83–139 | Neighbor BVD, raw substages, pre-repair penalty, RK-weighted conservation diagnostics. |
| S9 | [`src/differentiable/teno5_jax.py`](src/differentiable/teno5_jax.py):39–144; [`projection_jax.py`](src/differentiable/projection_jax.py):39–63,124–126; [`teno5_jax_toy.py`](src/differentiable/teno5_jax_toy.py):37–67,73–183 | Polynomial candidates, chi, normalized weights, soft/hard mismatch, blending heads. Some docstrings describe an earlier prototype; actual checkpoints have the five-layer architecture. |
| S10 | [`gated_teno5_jax.py`](src/differentiable/gated_teno5_jax.py):19–96; [`reconstruction.py`](src/differentiable/reconstruction.py):22–74; [`model_io.py`](src/differentiable/model_io.py):49–107 | Explicit V2/P0 policy identities, deterministic gate, strict finite checkpoint validation. |
| S11 | [`GenerateTrainingData_Euler.py`](scripts/legacy_imitation_training/GenerateTrainingData_Euler.py), `split_characteristic_stencils`; [`GenerateTrainingData_EulerExact.py`](scripts/legacy_imitation_training/GenerateTrainingData_EulerExact.py):131–156,180–255,257–328 | Training input representation, boundary categories, earlier approximate cell averages, reference interpolation, magnitude filtering. |
| S12 | [`Train_WENO7NN_Euler.py`](scripts/legacy_imitation_training/Train_WENO7NN_Euler.py):319–352,465–550,583–601,1366 onward; companion WENO5 and TENO trainers | CSV loading, split policy, physical acceptance checks, scaled regression/anchoring. |
| S13 | [`GenerateTrainingData_TENOExact.py`](scripts/legacy_imitation_training/GenerateTrainingData_TENOExact.py), `oracle_labels`; [`PDEDataset_NonIdentifiability_Audit.py`](scripts/unrolled_physics_loss/PDEDataset_NonIdentifiability_Audit.py):70–119,170–210 | Local oracle labels, tie handling, normalization/raw-target confound. |
| S14 | [`Train_NNTENO5euler_UnrolledJAX.py`](scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX.py):62–108,122–144,160–237,240 onward; curriculum/mix/learned-lambda companion scripts | Fixed training dt, sampler filtering, smooth data, guarded unrolled losses, subsequent attempts. |
| S15 | [`Train_Gated_NNTENO5_Dataset_V2.py`](scripts/unrolled_physics_loss/Train_Gated_NNTENO5_Dataset_V2.py):58–70,99–159,183–234,238–295 | Separate families/backbones, case-specific dt, unguarded evaluation, current warm-up representation. |
| S16 | [`data/nnteno_v2/README.md`](data/nnteno_v2/README.md):21–62; `build_dataset.py`, `verify_dataset.py`, `data/cases.jsonl`, `quality_report.json` in that package | Exact analytic cell-average corpus, trajectory IDs, coverage/absence, split/grid semantics. |
| S17 | [`v2_evaluate.py`](src/differentiable/v2_evaluate.py):66–106,110–169; [`v2_studies.py`](src/differentiable/v2_studies.py):106–165; [`v2_benchmark.py`](src/differentiable/v2_benchmark.py):50–92,114 onward | Full-validation computation, score definitions, input mismatch, repair comparison, fixed prospective settings/reference limits. |
| S18 | [`v2_nonlinear.py`](src/differentiable/v2_nonlinear.py):11–38; [`benchmark_cases.py`](src/differentiable/benchmark_cases.py):142–179; `smooth_study.py`, `reference_study.py`, `repair_study.py` | Nonlinear consistency, P0 point references/time rules, order/reference/repair studies. |

### 10.2 Measurement artifacts and historical accounts

| ID | Evidence | Important qualification |
|---|---|---|
| A1 | [`models/euler_instability_trace.csv`](models/euler_instability_trace.csv); [`EulerInstability_TraceCapture.py`](scripts/unrolled_physics_loss/EulerInstability_TraceCapture.py):53–114; `plots/04_euler_instability.png` | Stored density trace confirms instability, not a measured basis condition. Current sampler is not fully frozen. |
| A2 | [`models/nn_teno5_oracle_metadata.json`](models/nn_teno5_oracle_metadata.json):12–17,60–114; oracle HDF5/history/split | Classifier metrics plus checkpoint; PDE failure freshly reproduced. Contrary to a historical note, the oracle model was eventually committed (`df4986c`). |
| A3 | [`full-validation/per_case.json`](docs/research_release/results/full-validation/per_case.json), [`failure_inventory.json`](docs/research_release/results/full-validation/failure_inventory.json), [`summary.json`](docs/research_release/results/full-validation/summary.json) | Match case IDs, modes, strengths. Null extrema do not reveal the first violation. Failed rows do not qualify for accuracy claims. |
| A4 | [`repair/repair.json`](docs/research_release/results/repair/repair.json), its six NPZ histories; P0 [`repair_assessment.json`](src/differentiable/validation/p0-2026-10-14/repair-assessment/repair_assessment.json) and baseline `repair-before.log`/`repair-after.log` | Measured repaired-state sources and old low-positive-state defect; not a general conservative fix. |
| A5 | [`wall-smoke/benchmark.json`](docs/research_release/results/wall-smoke/benchmark.json), [`wall-closure/benchmark.json`](docs/research_release/results/wall-closure/benchmark.json) | Selected method fails; classical controls complete. Full-time blast has no accuracy reference. |
| A6 | [`independent/benchmark.json`](docs/research_release/results/independent/benchmark.json), especially `independent-sod-192`, `independent-contact-{96,192}`, and advection square rows | Cell-average prospective results; mixed outcomes. Selected/trained Euler labels duplicate the same policy. |
| A7 | [`nonlinear-euler/nonlinear_euler.json`](docs/research_release/results/nonlinear-euler/nonlinear_euler.json); [`temporal-cpu/convergence.json`](docs/research_release/results/temporal-cpu/convergence.json); [`selected-spatial/convergence.json`](docs/research_release/results/selected-spatial/convergence.json) | General nonlinear FV and special entropy-wave rates must be distinguished. Extended fine-grid temporal protocol remains incomplete. |
| A8 | [`input-mismatch-r2/input_mismatch.json`](docs/research_release/results/input-mismatch-r2/input_mismatch.json); `normalized_inputs.npz` | Only training snapshots; does not establish causal improvement from a new warm-up. |
| A9 | P0 [`reproduction/benchmarks/metrics.json`](src/differentiable/validation/p0-2026-10-14/reproduction/benchmarks/metrics.json), `reproduction/smooth/smooth_summary.json`, `warmstart-controls/metrics.json`, `reference-aggregate/reference_summary.json` | Older checkpoint/policy, historical point-sampled estimands. Learned-head attribution and Shu reference limits. |
| A10 | [`models/v2/euler_gated_dataset_v2_seed2026.json`](models/v2/euler_gated_dataset_v2_seed2026.json), scalar counterpart, histories/test records; `configs/v2/{full,selected,independent,wall-closure}.json` | One stored training seed per physics. Earlier test reports exist; later selection used validation, not those reports. Sparse history cannot recover every training repair. |
| H1 | [`Documentation/Results.md`](Documentation/Results.md):29–138,142–324,384–449,451–594,598–718 | Rich phase chronology; distinguish reported experiments from retained raw evidence and overstrong causal interpretations. |
| H2 | [`Documentation/Report_Results.md`](Documentation/Report_Results.md):104–194 | Boundary discrepancy study and teacher-target limitations, mainly written evidence. |
| H3 | [`Documentation/P0_Validation.md`](Documentation/P0_Validation.md):51–157,159–205; [`docs/research_release/RESULTS.md`](docs/research_release/RESULTS.md) | Later source-linked validation and explicit limitations; supporting JSONs checked directly. |
| H4 | `Documentation/UNROLLED_PHYSICS_LOSS_GUIDE.md`, `Literature_Report_Mapping.md`, `SOUTENANCE_SHOCKS_AND_WENO.md`; root slides/PDFs, `TENO_NN_Course.html`, presentation review | Context and proposed interpretation; not independent experiments. Duplicated literature PDFs were not counted as project evidence. The root `rapport_de_stage.pdf` concerns an unrelated application and supplies no CFD evidence. |

The PNGs named in §9 were visually inspected. The historical Shu TV and WENO7 Sod TV plots support variation differences; they do not contain raw pressure/density failure traces. The paper PDFs in `references/` and `Documentation/Papers/` describe others' experiments, which were not attributed to this internship. The user’s “SIAC/DG/error estimation” connection is future research context; no DG solver or learned SIAC implementation was found in this repository.

### 10.3 Git chronology used

| Commit | Relevance |
|---|---|
| `2c212c6`, `5443ca7` | Initial equations/basic framework. |
| `daa861f` (19 July) | Common Roe face basis refactor, characteristic transformation changes, dissipation change, Euler/spatial regression additions. No unique instability attribution inferred. |
| `c1a3567` (19 July) | Adaptive Euler timestep and validation of intermediate RK stages added. Current classical/legacy path retains these. |
| `753d526` (11 August) | Exact-target generator, WENO7 trainer, magnitude filtering/epsilon mitigation; old training log removed. That removed log shows an optimization plateau, not a verified Euler NaN trace. |
| `7164e23`, `5c5d07c`, `2308522` | Near-passing WENO7 candidate, seed searches, anchor sweeps, remaining TV/Linf tradeoffs documented. Candidate mostly not retained. |
| `c68be0d`, `59deb9e` | TENO pipeline audit and switch from teacher fidelity to real-reference acceptance checks. |
| `58d3edf`, `ce91a07`, `109326a` | Oracle labels, reported instability, reported anchor mitigation and performance ceiling. |
| `df4986c` (20 August) | JAX solver/unrolled training and stored instability trace, scalar histories, non-finite Euler checkpoint; oracle artifacts eventually committed. |
| `4e514b1` (8 September) | Later curriculum/mixing/head variants, backed-up finite checkpoints, model comparison JSONs and phase reports. Many phase details arrive together; commit time is not an individual experiment timestamp. |
| `dcb779b` (15 October) | P0 reproducibility, unguarded stage/budget diagnostics, repair correction, attribution/convergence/reference studies. |
| `2c7532b`, `9919414` | Gated V2/data-backed approach and subsequent integration context. |
| `fc00b77`, `61b0f70` (20 October) | Explicit historical/V2 policies, full-validation strength studies, input mismatch, general nonlinear Euler consistency. |
| `279434d` (20 October) | Prospective stress testing, wall correctness correction, reference limits and portfolio artifacts. |
| `26dd5d8` (22 October) | Compiled one-flux/one-step checks remain at 1e-12, full 80-step hard-TENO CPU regression tolerance set to 1e-7 after reported runner differences ≈4.9e-8. This is portability/threshold sensitivity evidence, not a recorded Euler blowup. |
| `8d20d92`, `989fceb`, `f9d4c2a` | Merge of research release and README presentation changes; audited final state. |

Git history was inspected locally with `git log --all`, path histories, `git show`, and filename inventories. No repository mutation, fetch, checkout, commit, or remote publication was performed.

### 10.4 Lightweight checks performed during this audit

**Environment.** Existing `teno-jax`: Python 3.11.15, NumPy 2.4.6, JAX 0.10.2, Optax .2.8, CPU float64 enabled. Existing `weno-nn`: legacy Python/Keras/TensorFlow environment. No dependency installation or GPU experiment. Temporary scripts/results were written only beneath `/tmp/euler-research-audit`; the measurements and reproducible recipes below make this report independent of those temporary files.

**Existing targeted regressions:**

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/euler-audit-mpl \
  /home/bazouil/miniconda3/envs/teno-jax/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_core.py \
  src/differentiable/tests/test_repair.py \
  src/differentiable/tests/test_v2.py \
  tests/test_teno5_jax_euler_regression.py
```

Result: **39 tests passed; 25 subtests passed; 43.14 s**. This covers current stage rejection, classical Euler controls, basis/parity, low-positive repair, wall conservation, and the experimental smooth-state flux contract. It does not validate all neural configurations. No new repository tests were added.

**Independent algebra/consistency/wall/penalty probes:**

- Seed 20261029, 256 states: rho uniform [.1,5], v [−3,3], p [.1,12]; compare expanded flux to primitive expression and eigenvalues to v±sqrt(1.4p/rho). Max errors: flux **2.84217e-14**, speeds **3.55271e-15**, `LR−I` **8.90194e-16**.
- `nonlinear_spatial(n,teno5_reconstruct_jax,factory)`, N=40/80/160: split-flux L1 **4.0836059e-4, 8.8758570e-5, 2.1253486e-5**, last order **2.06219**; state-flux **5.5540452e-5, 1.7441256e-6, 5.4555269e-8**, last order **4.99864**.
- Wall probe uses the exact N=24, RNG seed=1 state recipe in `test_v2.py:99–109`, the frozen V2 Euler model, soft/.25, and `get_euler_flux_jax(...,return_boundary_flux=True)`. Mirror-only versus closed-wall results are listed in §3.7; this isolates boundary correctness without a long rollout.
- `positivity_penalty_jax([[1,0,0]])=0` while repair yields E=.125; `positivity_penalty_jax([[2000,0,2.5]])=0` while repair yields rho=1000; `[[.01,0,.025]]` is unchanged. These are constructed implementation controls, not additional archived CFD failures.
- SHA-256 checks: **416 artifact-manifest entries** matched across full-validation, repair, wall-closure, wall-smoke, nonlinear-euler, input-mismatch-r2, independent, selected-spatial, and temporal-cpu directories. This checks retained file identity; it does not independently rerun the 11,064-case sweep.
- All **15 stored JAX pickle checkpoints** were inspected with the strict loader: 14 have finite, recognized parameter schemas; the original `nn_teno5euler_unrolled_jax_params.pkl` is rejected for non-finite parameters. The alias identities described in §4.5 were independently checked.

**Reproduce the oracle Sod failure** in the legacy environment, from repository root:

```python
import json
from keras.models import load_model
from src.analysis.euler_regression import run_euler_benchmark
from src.initial_conditions.InitialConditions import sod
from src.schemes import NN_TENO5_Euler
model = load_model('models/nn_teno5_oracle.h5', compile=False)
with open('models/nn_teno5_oracle_metadata.json') as stream:
    thresholds = json.load(stream)['calibrated_thresholds']
run_euler_benchmark(lambda: NN_TENO5_Euler(model, threshold=thresholds),
                    sod(), 120, 1.0, .2, .35)
# FloatingPointError: Non-positive Euler density at step 6.
```

**Reproduce the WENO7 forward-layer check**, without training or writing any checkpoint, in the legacy environment:

```python
import inspect
import numpy as np
from keras import backend as K
from src.networks.WenoNetworks import _weno7_coefficients
x = np.loadtxt('data/eulerAvgsExact7.csv', delimiter=',').T
lo = x.min(axis=1, keepdims=True)
span = np.ptp(x, axis=1, keepdims=True)
x = ((x-lo)/np.where(span > 0, span, 1.)).astype(np.float32)
source = inspect.getsource(_weno7_coefficients)
for eps in ['1e-6', '1e-4']:
    namespace = {'K': K}
    exec(source.replace('eps = 1e-4', 'eps = '+eps), namespace)
    inp = K.placeholder(shape=(None, 7))
    evaluate = K.function([inp], [namespace['_weno7_coefficients'](inp)])
    coefficients = evaluate([x])[0]
    print(eps, np.sum(~np.isfinite(coefficients).all(axis=1)))
# 1e-6 10
# 1e-4 0
```

The fresh old-epsilon bad-row zero-based indices were 9950, 40602, 62521, 63600, 65779, 67706, 76530, 98058, 99108, 99967. The direct expression was evaluated unchanged apart from epsilon. No claim is made that every backend/compiler yields identical bad-row counts. Adding intermediate outputs to the graph can change floating-point optimization.

**Reproduce the first-invalid V2 pressure trace**, in the JAX environment:

```python
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import h5py
from src.differentiable.model_io import load_checkpoint
from src.differentiable.reconstruction import make_reconstruction
from src.differentiable.euler_jax import get_euler_flux_jax

identifier = 'euler-validation-000088-1023461505243801251'
with h5py.File('data/nnteno_v2/data/euler_validation.h5') as f:
    g = f['N128']
    ids = [x.decode() if isinstance(x, bytes) else str(x)
           for x in g['case_id'][:]]
    index = ids.index(identifier)
    initial = np.asarray(g['states'][index, 0])
    dt = float(g['short_dt'][index])
model = load_checkpoint('models/v2/euler_gated_dataset_v2_seed2026.pkl')

for mode, strength in [('soft', .25), ('soft', .75),
                       ('soft', 1.), ('hard', .5)]:
    rec = make_reconstruction(model, 'gated-'+mode, strength)
    flux = jax.jit(get_euler_flux_jax(rec, 'transmissive'))
    state = jnp.asarray(initial)
    invalid = False
    for step in range(15):
        u0 = state
        for stage in [1, 2, 3]:
            if stage == 1:
                state = u0-dt*128*flux(u0)
            elif stage == 2:
                state = .75*u0+.25*state-.25*dt*128*flux(state)
            else:
                state = u0/3+2*state/3-2*dt*128*flux(state)/3
            value = np.asarray(state)
            pressure = .4*(value[:, 2]-.5*value[:, 1]**2/value[:, 0])
            invalid = not (np.isfinite(value).all()
                           and np.all(value[:, 0] > 0)
                           and np.all(pressure > 0))
            if invalid:
                print(mode, strength, step+1, stage,
                      value[:, 0].min(), pressure.min())
                break
        if invalid:
            break
    if not invalid:
        print(mode, strength, 'all 15 steps admissible')
```

This deliberately exposes/inspects each raw stage instead of reproducing the fused batched validation executable. It is a diagnostic trace; its exact continuation after the first invalid stage is neither run nor claimed. The first-invalid results are in §3.3.

### 10.5 Primary literature consulted for the reliability connection

- Zhang and Shu, *Maximum-principle-satisfying and positivity-preserving high order schemes for conservation laws: survey and new developments*, 2011, [author-hosted manuscript](https://www.math.purdue.edu/~zhan1966/research/paper/review.pdf): conditional FV/DG positivity framework. This is not a guarantee for the repository's neural flux reconstruction.
- Giesselmann, Makridakis, Pryer, *A posteriori analysis of discontinuous Galerkin schemes for systems of hyperbolic conservation laws*, 2014 preprint / 2015 journal article, [primary manuscript](https://arxiv.org/html/1405.7616v1): reconstruction/relative-entropy methodology and shock limitations.
- Giesselmann, Kwon, Krumscheid, *A Posteriori Error Analysis of Runge–Kutta Discontinuous Galerkin Schemes with SIAC Post-Processing for Nonlinear Convection–Diffusion Systems*, submitted 1 April 2026, [primary preprint](https://arxiv.org/html/2604.01200v1): actual RKDG–SIAC error-control connection, with the scope qualifications in §5.5. Its verified existence does not establish any learned-SIAC claim.

The immediate research priorities are conservative stage admissibility, isolation of the first failure mechanism, and correction of general nonlinear FV consistency. Broader training and learned postprocessing should be evaluated against those numerical contracts and independent error evidence.
