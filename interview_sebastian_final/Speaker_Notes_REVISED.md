# Speaker notes — KIT research interview

**Main talk: 9 slides, scheduled 13 minutes 15 seconds.** Backups A–C are outside that timing and hidden in the PowerPoint slideshow.

Use the spoken scripts below; preparation notes and Q&A are **not** part of the timed talk. Equations are explained in words rather than read symbol by symbol. The timing is a rehearsal estimate, not a measured human delivery.

| Slide | Duration | Cumulative | Spoken words |
|---|---:|---:|---:|
| 1. Original landing page | 0:25 | 0:25 | 49 |
| 2. Research background and objectives | 1:00 | 1:25 | 124 |
| 3. Where the network enters the Euler scheme | 1:35 | 3:00 | 168 |
| 4. Negative density and pressure | 2:15 | 5:15 | 233 |
| 5. Increasing dissipation: what the experiment did not establish | 1:30 | 6:45 | 159 |
| 6. Numerical repair and the conservative budget | 1:25 | 8:10 | 184 |
| 7. Generalization beyond short validation | 2:00 | 10:10 | 218 |
| 8. Finite-volume consistency and implementation limitations | 1:20 | 11:30 | 149 |
| 9. Lessons and reliable numerical research | 1:45 | 13:15 | 189 |

Total main spoken script: **1473 words**. The scheduled pace includes pauses for the equations and plots. Rehearse once aloud; do not add the Q&A or backup scripts to the main talk.

The final slide is the end of the planned presentation. Invite discussion there. For PDF delivery use the Main_Only PDF when you want a hard stop after slide 9; the complete PDF retains the three backups.

## Slide 1 — Original landing page

Estimated speaking duration: **0:25**.

### Spoken script

Thank you for meeting with me. I’m Yasser Baouzil, and I’ll present my research internship on neural shock capturing. I’ll explain what I built, then focus on the Euler failures: what the evidence establishes, why several fixes were insufficient, and how numerical structure could make learned methods more reliable.

### Equation guide — not spoken

The unchanged cover shows a scalar conservation law. It is context; the Euler vector equation is introduced on slide 3.

### Established observations — not spoken

No numerical result is claimed by the aircraft image. It is V4’s original illustration.

### Mechanisms, uncertainty and remedies — not spoken

The image is not an internship CFD experiment. Introduce the evidence-driven scope without promising a universally improved solver.

### Mathematical terms to remember

Conservation law; reconstruction; shock capturing; numerical reliability.

### Preparation and evidence boundaries — not spoken

The title slide is deliberately unchanged from V4, including the original schlieren image, author and supervisor. That photograph is visual motivation, not a simulation or experiment performed in this repository. The scalar conservation law on the original landing page is expanded to the Euler system on slide 3. Do not read the whole cover or promise a universally improved scheme. Advance after introducing the scientific question.

### Possible technical questions

**What was your own contribution?**

A modular reconstruction/solver framework, neural integration and training/evaluation experiments, and investigation of stability and generalization limits. Attribute the classical schemes and established limiters to their literature, and distinguish the later validation work and method generations.

**What is the main result you want me to remember?**

Useful accuracy gains were possible on some configurations, but independent conservation, stage admissibility, consistency and reference checks exposed limitations that a favorable error number alone would miss.


## Slide 2 — Research background and objectives

Estimated speaking duration: **1:00**.

### Spoken script

My technical work began with a modular conservation-law solver: an ENO baseline and WENO and TENO reconstructions of orders three, five and seven. I integrated neural reconstruction decisions, characteristic Euler reconstruction, convergence checks, and benchmarks including Sod and Shu–Osher. I also worked with differentiable JAX rollouts, classical–neural blending and investigations of numerical failures.

The conservation law says that local changes are balanced by physical fluxes. Keeping that structure visible made it possible to ask which numerical properties learning preserved.

The timeline separates four generations: original WENO coefficient corrections, neural TENO and oracle candidate selection, historical JAX rollout training, and later gated V2. I distinguish those checkpoints and later validation work throughout; the results do not all belong to one network or one internship-stage experiment.

### Equation guide — not spoken

∂tU+∂xF(U)=0 expresses local conservation. U is a scalar or conserved vector; F is its physical flux. A numerical shared-face balance is a separate discrete construction.

### Established observations — not spoken

Source and benchmark artifacts establish the modular framework, neural integration and separate method generations. Convergence and failure investigations do not imply all implementations attain their named formal orders.

### Mechanisms, uncertainty and remedies — not spoken

Early scalar results do not establish system-level stability. Distinguish personal framework/integration work from established classical schemes, later generations and the subsequent audit. The ENO baseline is not claimed to have verified orders 3/5/7.

### Mathematical terms to remember

U: conserved state; F: physical flux; ENO/WENO/TENO; reconstruction order; benchmark; method generation.

### Preparation and evidence boundaries — not spoken

Equation: ∂tU+∂xF(U)=0 is a local balance, not a statement that every numerical state is physically admissible. In Euler, U has three components; scalar cases have one. WENO/TENO order labels describe their reconstruction design, not a verified fifth/seventh-order result for every implementation path.

The function named ENO3 selects between two linear candidates on a three-point stencil, so do not claim that the whole ENO/WENO framework has uniformly demonstrated orders 3/5/7. Use the on-slide wording, which separates the ENO baseline from WENO/TENO orders. The original WENO-NN reference checkpoint is SeemsGood3.h5. Keras NN-TENO5 imitation/oracle, historical JAX blends, and V2 scalar/Euler models are distinct. V2 uses separate scalar and Euler backbones. The original aircraft image remains contextual rather than numerical evidence.

Early successes on scalar steps did not establish a reliable Euler method. The main task was to insert learning into an existing numerical structure and then evaluate the complete solver, not just a classifier.

### Possible technical questions

**Why learn only part of the solver?**

It keeps the known physical flux and conservative update visible. Learning can propose local reconstruction decisions while numerical constraints and diagnostics can remain independently defined.

**Does higher reconstruction order imply a more accurate shock?**

No. Formal smooth-solution order, resolution of a discontinuity, nonlinear stability and errors at a fixed grid are different properties. The general nonlinear FV consistency test later shows a limitation shared with the classical baseline.


## Slide 3 — Where the network enters the Euler scheme

Estimated speaking duration: **1:35**.

### Spoken script

The Euler state contains density, momentum and total energy. Velocity is momentum divided by density. Pressure couples all three entries in the physical flux.

The update on the right is the forward-Euler building block. A common face flux leaves one cell and enters its neighbor, so interior contributions cancel in the discrete budget. SSPRK3 composes several such evaluations.

At each face, the actual code uses a common Roe characteristic basis, forms Lax–Friedrichs split fluxes, and reconstructs each characteristic field. The outlined block is where the neural decision enters. The combined flux is then transformed back into conserved variables. The default solver reconstructs characteristic split fluxes, rather than first constructing separate physical left and right states.

Later V2 starts with classical TENO and adds a gated, scaled neural correction. G is a classical trouble gate and lambda is the correction strength. Zero lambda recovers classical TENO. This retains conservative form, but it does not establish positive Euler states. Reconstruction and wave speeds are evaluated again at each Runge–Kutta stage.

### Equation guide — not spoken

U=(ρ,m,E) contains density, momentum density and total energy density; v=m/ρ. F is the physical Euler flux. The flux-difference equation is one forward-Euler step. In R_V2=R_T+Gλ(R_NN−R_T), R_T is classical reconstruction, G the classical trouble gate and λ the correction strength. G is not a neural confidence certificate.

### Established observations — not spoken

The default code reconstructs characteristic split fluxes in a common Roe face basis. At zero V2 correction strength it recovers classical TENO. No controlled neural component-wise comparison or first-failure basis-conditioning trace establishes a unique basis defect.

### Mechanisms, uncertainty and remedies — not spoken

Conservative form and characteristic separation do not constrain the Euler invariant set. The nearby-cell LF speed bound is an implementation choice; its adequacy on every failing stencil remains unmeasured. Structural face/stage control is the relevant proposed addition.

### Mathematical terms to remember

U=(ρ,m,E); v=m/ρ; total energy density; Roe averages; left/right eigenvector matrices; characteristic field; LF split; shared-face flux; SSPRK3; G and λ.

### Preparation and evidence boundaries — not spoken

Explain the update as a forward-Euler building block, not the entire third-order RK formula. D(U) is defined in Backup A with the 1/Δx factor. For periodic data interior flux cancellation gives conservation; transmissive data require boundary exchange, and walls exert momentum forces. Consistency of constants is separate from conservation and high-order accuracy.

Current default: euler_jax.py:166–254. At a face the same Roe L and R are used for all stencil entries and both LF signs. The negative branch is shifted one cell and reversed. The adjacent-cell speed maximum is not automatically a bound on every state in the reconstruction stencil. The current basis inverse passed moderate-state checks; its conditioning on a failing trajectory has not been measured.

The V2 gate is classical, not a learned confidence certificate. Each LF sign/characteristic field can receive its own gate decision. The backbone is 5→64→32→16→8→3, with per-stencil min/range normalization. Soft candidate weights in V2 differ from hard mask inference in older JAX policies. The experimental state-reconstruction/Rusanov path on slide 8 is a separate formulation.

### Possible technical questions

**What exactly does the neural network see?**

Five neighboring values of one characteristic split flux after local min/range normalization, not the three conserved variables presented jointly. The same small classifier is applied over fields and LF directions.

**Is using a Roe basis the same as using a Roe Riemann solver?**

No. Here Roe averages define the projection basis. LF splitting and reconstruction define the numerical flux; it is not the Roe approximate Riemann flux.


## Slide 4 — Negative density and pressure

Estimated speaking duration: **2:15**.

### Spoken script

These were physical-admissibility failures, not merely less accurate shock profiles. The oracle Keras NN-TENO5 failed Sod with non-positive density at step six. Later V2, at soft strength point seven five, produced negative pressure at step thirteen, stage one. Strength point two five completed the same fifteen-step case admissibly.

Pressure is proportional to total energy minus kinetic energy. The boxed expression linearizes pressure around an admissible reference state. An energy error enters directly; a momentum error is multiplied by minus velocity; a density error enters with half the velocity squared. This is a local sensitivity formula, not a proof of the unique error responsible for the recorded failure.

A neural face-flux correction changes density, momentum and energy in the next stage. Their coupled error can make internal energy negative. Then the sound speed, which involves the square root of pressure over density, and characteristic calculations can break down.

We tried reduced corrections, blending, guards, repair and penalties. These were empirical safeguards. Normalized neural weights combine characteristic flux polynomials, not admissible Euler states. Conservation balances transfers but does not bound each cell’s pressure. SSPRK3 can transfer a suitable forward-Euler positivity property under its timestep restriction; it cannot create that missing property.

An established remedy is conservative positivity-preserving flux limiting toward an admissible low-order update. It needs appropriate face factors, stage constraints and CFL assumptions. That remedy was not implemented or validated in this neural solver.

### Equation guide — not spoken

For γ>1 and ρ>0, p>0 is equivalent to E−m²/(2ρ)>0. Differentiation gives ∂p/∂E=γ−1, ∂p/∂m=−(γ−1)v and ∂p/∂ρ=(γ−1)v²/2. Therefore δp≈(γ−1)(δE−vδm+v²δρ/2), with all coefficients evaluated at the admissible reference state. The neglected remainder is second order in small state perturbations. The formula is a sensitivity explanation, not an exact finite-perturbation test. Sound speed c=sqrt(γp/ρ) requires valid pressure and density.

### Established observations — not spoken

Oracle Keras NN-TENO5: Sod N120, non-positive density at step 6. V2 diagnostic: soft/.75 first invalid pressure at one-based step 13, stage 1; soft/.25 completes the same 15 steps. These belong to different checkpoints/generations.

### Mechanisms, uncertainty and remedies — not spoken

The local sensitivity equation does not identify a unique failing face or classifier decision. Reduced strength, guards, repair, blending and penalties helped some configurations but imposed no invariant-domain theorem. Known conservative positivity limiting is unimplemented/unvalidated here and requires a suitable low-order baseline and CFL.

### Mathematical terms to remember

Admissible set; internal energy E−m²/(2ρ); positivity versus conservation; forward Euler; SSP coefficient; invariant domain; low-order admissible baseline.; first-order pressure variation; admissible reference state; pressure gradient in conserved variables.

### Preparation and evidence boundaries — not spoken

Oracle configuration: models/nn_teno5_oracle.h5, calibrated thresholds from its metadata, Sod N=120, L=1, T=.2, adaptive CFL=.35. The audit independently reproduced the exact density exception. The oracle supervised isolated point-value accuracy over masks on synthetic functions, not stable Euler evolution. Misclassification, objective mismatch and evolving-input shift have not been uniquely separated at the first failing face.

V2 case ID: euler-validation-000088-1023461505243801251; N=128; dt=.0017428079661349484; fifteen full SSPRK3 steps; transmissive boundaries. The separate first-invalid trace gives p_min=−.0038048537160435105 with ρ_min≈1.21304 for soft/.75. The archived batched full-validation record has a different negative minimum and a finite final state, while the direct repair study continues to a non-finite final state. Both reject the policy by raw-stage admissibility. Do not claim their complete trajectories are bit-identical.

For γ=1.4, positive pressure is equivalent to positive internal-energy density when ρ>0. The admissible set is convex, but the present characteristic flux candidates are not admissible state endpoints. Positivity-preserving flux limiting needs a correct admissible low-order update, coupled face/stage constraints, and the appropriate CFL. It is not achieved by picking an arbitrary convex coefficient.

The historical penalty-after-repair defect and its subsequent correction are discussed on the next two slides. No theorem of positivity or entropy stability exists for the present learned reconstruction.

### Possible technical questions

**How can a conservative method produce negative pressure?**

Conservation is a telescoping shared-face balance; it does not constrain each updated internal energy. A flux correction can change E, m and ρ so that E−m²/(2ρ) becomes negative. Normalized characteristic-flux weights are not convex combinations of admissible Euler states.

**Why does SSPRK3 not automatically preserve positivity?**

Its convex-stage representation can inherit an invariant-set property of the forward-Euler spatial map under a suitable step bound. If that map is not positivity preserving, the SSP formula alone supplies no such guarantee.


## Slide 5 — Increasing dissipation: what the experiment did not establish

Estimated speaking duration: **1:30**.

### Spoken script

Here the LF multiplier increased while the timestep stayed fixed. Multiplier one ran away and became non-finite. One point five delayed the growth but still reached enormous density. Multipliers two, three and five diverged earlier.

This does not prove that additional dissipation inherently destabilizes Euler. The experiment was confounded: dissipation, the effective timestep restriction, and neural inputs changed simultaneously. The first equation shows that alpha changes the split flux. The second shows the indicative stability scale, alpha times delta t over delta x. Increasing alpha without reducing the timestep can move an explicit method outside its stable regime, and it also changes the network’s inputs.

Poor Roe-basis conditioning was suspected, but no first-failure condition-number trace established it as the initiating cause.

The next diagnostic is a matched-CFL comparison at a common final time, with a classical TENO5 control, first-invalid-RK-stage tracing and characteristic condition numbers. This removes specific confounds; it still would not identify a unique cause without controlled ablations.

### Equation guide — not spoken

F±=(F±αU)/2 defines LF splitting; the superscript sign selects the same sign in the bracket. CFL_LF≈αΔt/Δx is an indicative stability scale, not a proven global CFL limit for the arbitrary learned flux. At fixed Δt, increasing α changes both this scale and network inputs.

### Established observations — not spoken

The stored fixed-dt sweep has non-finite zero-based indices 11 for 1× and 3 for 2×/3×/5×. The 1.5× run remains finite through index 14 but density is about 6.21e18. It is not a stabilized success.

### Mechanisms, uncertainty and remedies — not spoken

The sweep changes three coupled factors, so it cannot isolate dissipation or Roe conditioning. A matched-CFL/classical-control study is proposed; it has not been run for this revision. More diffusion is not inherently unstable.

### Mathematical terms to remember

LF coefficient α; αΔt/Δx; explicit diffusion restriction; confounding variable; characteristic condition number κ(R); first-invalid stage.

### Preparation and evidence boundaries — not spoken

The re-created plot uses the exact CSV records. Its horizontal axis is zero-based, unlike the one-based step number in the fresh V2 trace. At multiplier 1 the stored density is ≈1.94e37 at index 10 and becomes non-finite at 11. At 1.5 the final index 14 is ≈6.21e18 but still finite. At multipliers 2,3,5 it becomes non-finite at index 3. A cross is placed at the next non-finite step at the preceding finite height, matching its diagnostic meaning; it is not a measured density at NaN.

The old trainer had N=64, nominal assumed maximum wave speed 2, CFL=.5, fixed dt=.00390625, while random cases had varying physical speeds. The alpha multiplier experiment used the frozen clamp-only guard. It does not describe the later V2 selected-model validation. The plot’s old title overstated causality, so this deck replaces its title/caption without altering the underlying values.

Basis inversion can amplify perturbations if R is poorly conditioned, but algebraic inverse accuracy on moderate states and conditioning are different tests. A diagnostic frozen-basis or component-wise control would itself be an altered method and should be labeled accordingly. Step halving must retain the same physical final time.

### Possible technical questions

**Why did more LF dissipation fail to stabilize the experiment?**

Alpha increased at fixed Δt, changing the explicit stability scale and the neural split-flux inputs simultaneously. Earlier divergence is observed, but it is not proof that dissipation itself is harmful. Match CFL and add a classical control before attributing the cause.


## Slide 6 — Numerical repair and the conservative budget

Estimated speaking duration: **1:25**.

### Spoken script

Repair helped some training trajectories continue, but it changed the numerical budget. In the conservative update, a shared face transfer cancels between neighboring cells. Overwriting an invalid cell afterward generally has no compensating transfer elsewhere.

The diagnostic on the right measures the integrated mass added by one intervention. The quoted numbers are more specific: the saved fifteen-step comparison has one repaired stage, and its final RK-weighted budget source is about point zero zero one six one one mass and point zero zero zero eight eight nine energy. These are not the raw unweighted stage increments or solution-error norms.

Historical penalties measured after repair could miss the violation just removed. Pre-repair penalties corrected that signal, but guarded training still differed from unguarded validation. Numerical survival was therefore not physical reliability.

Conservative positivity limiting instead constrains a shared face flux before updating both cells. Conservation follows from that common flux; positivity additionally requires an admissible low-order baseline, suitable correction factors and timestep bounds at each stage. This established alternative was not implemented or validated here. Existing state repair does not make the learned method positivity preserving.

### Equation guide — not spoken

The flux-difference update telescopes over interior faces. ΔM_repair=Δx Σ(ρ_after−ρ_before) defines a single intervention’s mass source. The quoted record instead sums RK-weighted repair exchanges over the 15-step rollout: per-step weights are (1/6,2/3,1) for stage-1/2/3 repair increments. Only one stage row is repaired in this case. Conservation_residual subtracts repair as an artificial source; physical_balance_error does not.

### Established observations — not spoken

repair.json, soft/.75 historical-repair: repaired_stage_rows=1; integrated_repair_exchange=[0.0016111493654163188,−0.00005706669322759772,0.0008888429518660344]. The source code proves this field is accumulated RK-weighted exchange, not an unweighted stage diagnostic. Physical balance error equals this source up to roundoff.

### Mechanisms, uncertainty and remedies — not spoken

Repair made some trajectories finite and pre-repair penalties fixed a historical signal defect. Neither restores conservative Euler dynamics or enforces a positivity-preserving spatial map. Shared-face positivity limiting is established literature but was not implemented or validated here.

### Mathematical terms to remember

Physical balance; repair source; integrated mass versus density; shared numerical face flux; RK-weighted repair budget; guarded training/deployment mismatch.

### Preparation and evidence boundaries — not spoken

The displayed ΔM formula is a per-intervention diagnostic. The quoted .001611149 mass and .000888843 energy are the final RK-weighted integrated repair contributions in a fifteen-step V2 diagnostic with soft/.75, historical-repair policy, the N=128 failing case, and one changed stage row. Exact SSPRK3 final repair weights are (1/6,2/3,1) for raw increments added after stages 1,2,3. For this particular first-stage repair, this distinction matters; do not equate the quoted final contribution with an unweighted density sum at that stage.

The current historical repair averages isolated invalid cells using periodic neighbors, even when the solver boundary is transmissive. Residual invalid clusters use floors/caps. The P0 fix prevents flooring already-admissible low-positive states, but it does not make repair globally conservative. Original per-update training intervention logs are incomplete.

The penalty-after-repair defect is historical and was corrected with a pre-repair penalty accumulator. A remaining implementation mismatch exists: the current penalty is zero at exactly zero pressure and at some positive out-of-scale states that repair changes. That is not established as the cause of any archived failure; mention it only if asked. A mathematically meaningful low-order flux baseline and sufficient CFL are prerequisites of positivity limiting.

### Possible technical questions

**Why did direct repair break conservation?**

Overwriting a cell adds an uncompensated increment after the conservative face-flux update. The recorded final RK-weighted source changes mass and energy; subtracting it in bookkeeping does not restore conservation of the original PDE.

**How does positivity-preserving conservative flux limiting work?**

Start from a low-order update known to be admissible under a sufficient CFL, then limit the shared-face high-order correction using coupled density/internal-energy constraints at each relevant stage. The shared face preserves conservative form; carefully chosen factors and assumptions provide admissibility.


## Slide 7 — Generalization beyond short validation

Estimated speaking duration: **2:00**.

### Spoken script

Selected V2 soft strength point two five completed three hundred and seven short Euler validation trajectories without invalid stages. Its macro L1 ratio was point nine four seven nine four. That score helped select the policy, so it is validation evidence.

Independent results were mixed. Sod at one hundred and ninety-two cells improved mean-state L1 by about four point three percent, but density total variation increased and a small undershoot appeared. The inset now explicitly zooms just right of the shock to show that excursion; it uses the same saved arrays. The contact regressed. Shu–Osher’s reference ranking is unresolved, so I show no accuracy point for it.

The stronger Woodward–Colella blast is the central failure: selected V2 became non-finite at both grids, while classical TENO5 completed both controls. There is no invented accuracy score for a failed run.

Potential contributors include stronger shocks, interacting waves and reflective walls absent from training. A reconstruction error also changes the next state, pressure and wave speeds; that changes the next characteristic projection and neural input. Warm-up used characteristic states rather than deployed split fluxes. These are measured mismatches or plausible feedback mechanisms, not isolated causes of the blast failure.

Better data can improve coverage, but cannot alone guarantee admissibility. Good short-horizon validation did not establish stability under stronger out-of-distribution Euler dynamics.

### Equation guide — not spoken

r_L1=E_L1(NN)/E_L1(TENO5); values below one improve that defined norm. The validation 0.94794 is a macro family×resolution ratio, not a pooled norm or universal success rate. TV(ρ)=Σ|ρ_(i+1)−ρ_i| measures density variation, not entropy stability. Sod’s explicit undershoot is a range excursion; it does not make its density negative.

### Established observations — not spoken

307 short validation trajectories pass. Independent Sod improves mean-state L1 but worsens TV/undershoot; contact N192 regresses; selected V2 blast N128/N256 becomes non-finite while TENO5 completes. Shu’s uncertain reference does not resolve a ranking.

### Mechanisms, uncertainty and remedies — not spoken

Training strength, horizon, wave interactions, walls and input mismatch are candidate transfer mechanisms, not proven unique blast causes. Added coverage can improve empirical transfer; admissibility still needs numerical constraints and an independently validated fallback.

### Mathematical terms to remember

Macro L1 ratio; family×resolution averaging; validation selection; prospective IVP; distribution shift; total variation; undershoot; short rollout; soft/hard policy.

### Preparation and evidence boundaries — not spoken

The aggregate Euler ratio .9479383302719695 averages family×resolution error ratios, excluding constants/tiny-error controls as declared. It is not a single relative norm over all entries. The 307 validation trajectories use 15 SSPRK3 steps each. The later full validation chose among strengths, conditioning uncertainty on a fixed backbone, not on multiple training seeds. Two grids of one prospective IVP are not independent trajectories.

Sod N=192: TENO5 mean-state L1=.002651360802799551; selected V2=.0025380153091719496, ratio=.9572500681506942; density TV .8774663254486642→.8797561899875133; density undershoot 0→.0003506854567733031. These are against an analytic cell-average reference. Contact N=192: .0022149671223096416→.0023672257990848457, ratio≈1.06874. At N=96 the contact is approximately a tie. So do not claim that refinement uniformly helps the learned method.

Closed-wall Woodward–Colella: pressure 1000/.01/100, rho=1, v=0; N=128/256; T=.038; corrected reflective-wall fluxes; nominal CFL=.15 with initial-speed safety 2. The selected model already fails the N=128 T=.001 smoke. Classical TENO5 completes both full-time grids with admissible raw stages. No full-time blast accuracy reference was computed after the failed smoke, so failed rows get no invented error ratio.

The original full final_benchmark figure is preserved in assets. The revised main slide uses six resolved Euler ratios from the same JSON and a separate native blast-failure callout; the enlarged shock-tail inset uses the same N=192 arrays as euler_sod. The revised Shu row contains no accuracy point or ratio because its numerical-reference ranking remains unresolved. The revised inset explicitly zooms x=.776–.805 and ρ=.12445–.12615, without modifying any saved array. Physical Shu entropy-wave oscillations must not be confused with numerical instability.

### Possible technical questions

**Why did the neural model fail the blast problem?**

The failure is established, but a unique cause is not. Stronger pressure contrasts, interactions, reflective walls, short training horizons and input mismatch are plausible contributors. First-invalid-stage traces and controlled distribution/solver ablations are needed to separate them.

**Why is more training data not a positivity guarantee?**

Coverage can reduce typical prediction errors, but a finite sample does not enforce the Euler invariant set for every future state. Errors change pressure and wave speeds, feeding back into later inputs. A guarantee needs an admissibility-preserving numerical acceptance mechanism under stated assumptions.


## Slide 8 — Finite-volume consistency and implementation limitations

Estimated speaking duration: **1:20**.

### Spoken script

This limitation also affected classical TENO5. On general nonlinear smooth Euler data, the historical finite-volume split-flux path approached spatial order two point zero one. Experimental state reconstruction with a Rusanov flux approached five on the same diagnostic.

The equation explains the mismatch: averaging a nonlinear physical flux is generally different from evaluating it at the average state. This particular historical path supplied fluxes of cell-average states to reconstruction with averaged-data semantics. A constant-pressure, constant-velocity entropy wave could hide the problem because its flux is linear in density.

This is not a statement that every LF split-flux WENO method is second order. Proper finite-difference formulations have a different interpretation.

The experimental state-flux remedy was checked for smooth spatial consistency, not full time-dependent shock robustness or positivity. A consistency improvement still needs separate Euler stability validation. The WENO7 and wall-contract issues remain separate implementation findings, covered in the notes and backups.

### Equation guide — not spoken

Overline(F(U)) generally differs from F(Ubar) because F is nonlinear. For sufficiently smooth U, the leading difference is (Δx²/24)F″(U)[U_x,U_x]+O(Δx⁴). The plotted errors compare semidiscrete spatial divergence with exact cell-average divergence; time truncation error is absent. These measured orders concern this formulation/test.

### Established observations — not spoken

Archived classical-TENO5 last spatial L1 orders are 2.0119685376 and 4.9999252974. The same general nonlinear smooth diagnostic and cell-average reference are used; this is not a temporal order experiment.

### Mechanisms, uncertainty and remedies — not spoken

Consistent FV state/Rusanov reconstruction fixes this observed smooth averaging-semantic mismatch; shock robustness, positivity and neural retraining remain unvalidated. Properly derived FD LF split-flux WENO is not ruled out by this result.

### Mathematical terms to remember

Cell average; point value; nonlinear consistency; semidiscrete divergence; truncation error; Rusanov flux; entropy wave; finite volume versus finite difference.

### Preparation and evidence boundaries — not spoken

The plot uses exact archived spatial-error values at N=40/80/160/320/640 and no time integration. For classical TENO5, last observed L1 order is 2.0119685376 for split flux versus 4.9999252974 for the experimental state flux. Gated policies give similar conclusions in this diagnostic. These are observed rates on a smooth test, not a universal fifth-order neural theorem.

For smooth U, a Taylor expansion gives overline(F(U))−F(Ubar)=(dx²/24)F''(U)[Ux,Ux]+O(dx⁴), under sufficient smoothness. Reconstructing face states first and evaluating a numerical flux avoids this particular mismatch. But those face states must then be admissible, so a consistency fix is not automatically a positivity fix. A proper finite-difference WENO method can start with pointwise physical fluxes; its semantics and flux-difference accuracy need to be established separately.

WENO7 backup: old eps=1e-6 yielded 10 non-finite coefficient rows in the stored 100k normalized stencils; eps=1e-4 yielded zero. The epsilon-squared-underflow explanation is doubtful because 1e-12 is far above float32 underflow. Cancellation/graph sensitivity remains under investigation. The reflective-wall correction explicitly imposes zero mass/energy wall flux; selected V2 blast failure persists after it. All these must be identified as distinct issues.

### Possible technical questions

**What exactly was the finite-volume consistency problem?**

The historical cell-average path used F(Ubar) as reconstruction data with averaged-flux semantics, although generally F(Ubar) differs from averaged F(U). Classical TENO5 showed the same O(Δx²) limitation. The state/Rusanov alternative reached order five only on this smooth spatial diagnostic.


## Slide 9 — Lessons and reliable numerical research

Estimated speaking duration: **1:45**.

### Spoken script

I would separate the learned proposal from the mathematical conditions used to accept it. We observed negative stages, artificial repair sources, limited transfer and accuracy–oscillation tradeoffs. Established numerical methods address these properties explicitly.

The first equation limits a neural face flux toward a suitable low-order flux. Shared-face use preserves conservative form. Positivity also needs an admissible low-order update, appropriate limiter factors and the relevant CFL. An arbitrary theta between zero and one is insufficient. Conditional SSP integration and structure-preserving reconstruction supply other parts of the method.

The second equation is the continuous PDE residual of a space–time reconstruction. It can support independent a posteriori assessment, but a residual alone is not a reliable error bound: reconstruction assumptions and stability estimates are essential.

This connects to Giesselmann, Kwon and Krumscheid’s RKDG–SIAC work on convection–diffusion systems. It does not certify this neural Euler solver. A concrete extension would begin with smooth admissible states and checkable reconstruction moments, then examine estimator effectivity and what changes near shocks. Those learned DG or SIAC extensions remain proposals.

Can learned high-order reconstruction improve numerical accuracy while retaining provable structural constraints and independently verifiable error control?

### Equation guide — not spoken

F_limited=F_LO+θ(F_NN−F_LO) is a proposed conservative correction architecture. θ is constrained by simultaneous face/stage admissibility conditions, not only 0≤θ≤1. R=∂t Uhat+∂xF(Uhat) is the inviscid continuous residual. A convection–diffusion residual has extra diffusion terms. Its relation to error requires independent reconstruction and a valid stability/error analysis.

### Established observations — not spoken

No learned DG/SIAC method or positivity-limited neural Euler implementation has been validated in this project. The proposed constrained architecture and independent assessment are research directions, not additional experimental results.

### Mechanisms, uncertainty and remedies — not spoken

Unit-mass SIAC convolution need not preserve each cell mean or positivity. A residual sensor is not a certificate; stability constants can deteriorate near shocks. Start where reconstruction, state bounds and moment assumptions can be checked before proposing broader guarantees.

### Mathematical terms to remember

Invariant domain; convex limiting; admissible low-order update; polynomial moments; space–time reconstruction; PDE residual; relative entropy; estimator effectivity; SIAC.

### Preparation and evidence boundaries — not spoken

The limited-flux equation on the slide is a proposed architecture, not code already validated here. Shared-face limiting preserves conservative form. A mathematical positivity argument also needs the low-order invariant-domain property, a sufficient CFL bound, compatible simultaneous face choices, and treatment of each stage/boundary. The Hu–Adams–Shu limiter and invariant-domain/convex-limiting literature supply foundations, not an automatic theorem for an arbitrary network.

For a posteriori assessment, do not compute only the residual of the numerical solution in its own identical discrete update, which is essentially zero by construction. Use a separate sufficiently regular reconstruction and continuous PDE residual, including relevant initial/boundary/time terms. In an actual convection–diffusion application the residual includes diffusion terms as well. The slide’s residual is for inviscid Euler.

The verified Giesselmann–Kwon–Krumscheid preprint is arXiv:2604.01200v1, submitted 1 April 2026. It treats periodic Cartesian RKDG settings, SIAC spatial reconstruction and time reconstruction for entropy-compatible convection–diffusion systems; worked model analyses include linear advection–diffusion, nonlinear scalar problems, and a diffusive p-system. Nonlinear stability constants depend on regularity/Lipschitz information. It contains no trained neural filter and is not a general post-shock neural Euler guarantee.

Possible research: constrain learned DG corrections in a zero-mean basis; constrain learned SIAC kernels to reproduction moments; evaluate residuals independently. Unit-mass signed convolution does not ensure local positivity or preserve every cell mean. Shock gating/limiting can alter superconvergence. Conclude here, invite discussion, and use the hidden backup slides only when relevant.

### Possible technical questions

**How does this connect to DG, SIAC and a posteriori estimation?**

DG makes polynomial moments explicit; constrained learned corrections or SIAC-type filters could preserve selected reproduction conditions. A sufficiently regular space–time reconstruction allows an independent continuous residual. Turning it into a reliable bound needs the appropriate stability, regularity and boundary assumptions; the 2026 paper does not certify this neural Euler method.


## Backup A — Detailed Euler formulation

Estimated speaking duration: **Backup; outside main timing**.

### Spoken script

The characteristic speeds are velocity minus sound speed, velocity, and velocity plus sound speed. A common Roe face basis projects every stencil entry. The LF splitting uses one face coefficient across the reconstruction stencil, and the negative direction is shifted and mirrored. The two reconstructed characteristic flux branches are added and transformed back. D denotes the conservative flux difference divided by cell width. SSPRK3 evaluates this operator at its intermediate states, so every state entering a new flux calculation must be physically admissible.

### Mathematical terms to remember

Roe enthalpy average; characteristic projection; LF stencil coefficient; D(U); SSPRK3 stages; ρ>0 and internal energy>0.

### Preparation and evidence boundaries — not spoken

The adjacent-cell LF maximum is an implementation choice, not a measured full-stencil bound. The common face basis is algebraically consistent, but the failing-trajectory condition number remains unmeasured. Do not call this a Roe Riemann flux. Forward-Euler conservation does not imply a positivity bound. The specific SSPRK3 convex coefficients transfer only a property that the underlying admissible forward-Euler maps satisfy under a sufficient step bound.

### Possible technical questions

**Why use characteristic rather than component-wise reconstruction?**

It separates local wave families and is a standard shock-capturing strategy. This repository has no controlled neural component-wise-versus-characteristic comparison proving a quantified benefit or isolating the failure cause.


## Backup B — WENO7 numerical conditioning

Estimated speaking duration: **Backup; outside main timing**.

### Spoken script

The classical WENO coefficients divide optimal weights by squared regularized smoothness indicators, then normalize. In the legacy backend, the stored hundred-thousand-row test produced ten non-finite coefficient rows with epsilon ten to the minus six and none with ten to the minus four. That supports the sampled mitigation. However, squaring ten to the minus six does not by itself underflow float32. Expanded smoothness polynomials can suffer cancellation, and a diagnostic near-flat row had a negative computed indicator. Exposing intermediate graph outputs can itself change arithmetic, so I have not isolated the exact NaN operation or established a globally stable seventh-order method.

### Mathematical terms to remember

WENO-JS α and ω; smoothness β; epsilon regularization; cancellation; float32 normal range; finite-output test versus accuracy/stability.

### Preparation and evidence boundaries — not spoken

Full recipe and ten bad-row indices are in EULER_STABILITY_RESEARCH_AUDIT.md §10.4. Commit 753d526 raises epsilon. Rejected candidate Euler failures within 1–3 steps and five seeds are historical descriptions without complete surviving logs. The fresh coefficient test is stronger direct evidence for the layer defect, but is not a reproduction of all historical training runs. A stable beta formula, precision comparison, nonnegative treatment and an order/critical-point study are worthwhile follow-ups. The instrumented first beta example near −9.54e-7 is cancellation evidence, not a fully isolated cause of the original full-graph NaN.

### Possible technical questions

**Does the epsilon change affect accuracy?**

It can change the smoothness weighting and sensitivity. The zero-NaN count establishes sampled finiteness only; a separate asymptotic/critical-point accuracy study is needed.


## Backup C — DG, SIAC and error estimates

Estimated speaking duration: **Backup; outside main timing**.

### Spoken script

DG makes the local polynomial representation explicit. A SIAC-type convolution can supply a smoother reconstruction, with polynomial reproduction governed by kernel moments. I would constrain any learned correction to preserve the required moments, and evaluate the PDE residual independently. A unit-mass kernel preserves a global integral in an appropriate periodic setting; it need not preserve each cell average, and signed kernels need not preserve positivity. The 2026 RKDG–SIAC paper supplies an error-analysis connection under its own hypotheses, not a trained neural filter or a general Euler shock certificate. Near shocks, large reconstruction gradients can make stability constants and estimates much less useful.

### Mathematical terms to remember

DG modal coefficients; kernel convolution; reproduction moments; unit mass; relative entropy; reconstruction regularity; effectivity.

### Preparation and evidence boundaries — not spoken

For the moment formula, r=1,…,s is a finite selected reproduction order, not every positive integer. A learned local DG correction can preserve each cell average by having zero cell mean; convolution with a unit-mass kernel is a different construction. The first-moment condition is sensitive to kernel convention and shift; specify the desired polynomial reproduction under the chosen coordinates.

Primary references: Hu/Adams/Shu JCP 242 (2013) 169–180, DOI10.1016/j.jcp.2013.01.024; Zhang/Shu JCP231(2012)2245–2258, DOI10.1016/j.jcp.2011.11.020; Gottlieb/Shu/Tadmor SIAMReview43(2001)89–112, DOI10.1137/S003614450036757X; Guermond/Nazarov/Popov/Tomas SISC40(2018)A3211–A3239, DOI10.1137/17M1149961; Giesselmann/Kwon/Krumscheid arXiv2604.01200v1. Full verified links and scope are in Sources_and_Evidence.md.

### Possible technical questions

**Could a learned SIAC filter simply smooth the shock?**

It might, and small residual or visual smoothness alone is insufficient. Accuracy, physical admissibility, shock location, boundary treatment and the estimator’s assumptions must be checked. Restricting/gating the filter near shocks may also change its superconvergence behavior.

## Delivery checklist

- Point to the first invalid RK stage, not only the final profile.
- Say “observed,” “hypothesized,” and “not yet implemented” explicitly when switching evidence levels.
- Keep Keras oracle, historical JAX and later V2 results separate.
- Preserve negative results: contact regression, undershoot, failed blast, unresolved Shu reference.
- Distinguish admissibility, conservation, entropy stability, consistency and error control.
- End at the research question; open a backup only in response to a relevant question.
