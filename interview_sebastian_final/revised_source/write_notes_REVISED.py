"""Editable content source for Markdown and embedded PowerPoint speaker notes."""
from pathlib import Path
import json,re
HERE=Path(__file__).resolve().parent
notes={}

def add(n,title,seconds,script,terms,detail,qa):
    notes[str(n)]={'title':title,'seconds':seconds,'duration':f'{seconds//60}:{seconds%60:02d}' if seconds else 'Backup; outside main timing',
                  'script':script.strip(),'terms':terms,'preparation':detail.strip(),
                  'qa':[{'q':q,'a':a} for q,a in qa]}

add(1,'Original landing page',25,"""
Thank you for meeting with me. I’m Yasser Baouzil, and I’ll present my research internship on neural shock capturing. I’ll focus on what the Euler experiments taught me: which failures were real, why some fixes were insufficient, and how numerical structure could make learned methods more reliable.
""",'Conservation law; reconstruction; shock capturing; numerical reliability.',"""
The title slide is deliberately unchanged from V4, including the original schlieren image, author and supervisor. That photograph is visual motivation, not a simulation or experiment performed in this repository. The scalar conservation law on the original landing page is expanded to the Euler system on slide 3. Do not read the whole cover or promise a universally improved scheme. Advance after introducing the scientific question.
""",[
('What was your own contribution?','A modular reconstruction/solver framework, neural integration and training/evaluation experiments, and investigation of stability and generalization limits. Attribute the classical schemes and established limiters to their literature, and distinguish the later validation work and method generations.'),
('What is the main result you want me to remember?','Useful accuracy gains were possible on some configurations, but independent conservation, stage admissibility, consistency and reference checks exposed limitations that a favorable error number alone would miss.')])

add(2,'Research background and objectives',60,"""
The starting point was a classical solver for conservation laws. The equation says that a cell changes because a physical flux enters and leaves it. Near a shock, high-order reconstruction must resolve sharp changes without introducing unacceptable numerical oscillations.

During the internship I worked with an ENO baseline and WENO and TENO implementations of orders three, five and seven. I tested scalar advection and Burgers, then compressible Euler problems such as Sod and Shu–Osher.

The project went through four generations. The original WENO network corrected reconstruction coefficients. Neural TENO selected candidate stencils. The JAX work trained through solver rollouts and tested blending. Later V2 retained a classical trouble gate around a learned correction. These are different methods and checkpoints, so I will identify which experiment belongs to which generation.
""",'U: conserved state; F: physical flux; ENO/WENO/TENO; reconstruction order; benchmark; method generation.',"""
Equation: ∂tU+∂xF(U)=0 is a local balance, not a statement that every numerical state is physically admissible. In Euler, U has three components; scalar cases have one. WENO/TENO order labels describe their reconstruction design, not a verified fifth/seventh-order result for every implementation path.

The function named ENO3 selects between two linear candidates on a three-point stencil, so do not claim that the whole ENO/WENO framework has uniformly demonstrated orders 3/5/7. Use the on-slide wording, which separates the ENO baseline from WENO/TENO orders. The original WENO-NN reference checkpoint is SeemsGood3.h5. Keras NN-TENO5 imitation/oracle, historical JAX blends, and V2 scalar/Euler models are distinct. V2 uses separate scalar and Euler backbones. The original aircraft image remains contextual rather than numerical evidence.

Early successes on scalar steps did not establish a reliable Euler method. The main task was to insert learning into an existing numerical structure and then evaluate the complete solver, not just a classifier.
""",[
('Why learn only part of the solver?','It keeps the known physical flux and conservative update visible. Learning can propose local reconstruction decisions while numerical constraints and diagnostics can remain independently defined.'),
('Does higher reconstruction order imply a more accurate shock?','No. Formal smooth-solution order, resolution of a discontinuity, nonlinear stability and errors at a fixed grid are different properties. The general nonlinear FV consistency test later shows a limitation shared with the classical baseline.')])

add(3,'Where the network enters the Euler scheme',105,"""
For Euler, the conserved state contains density, momentum and total energy. The physical flux contains momentum, momentum transport plus pressure, and energy transport. Velocity is momentum divided by density.

The update on the right is the forward-Euler building block: the difference of two numerical face fluxes changes the cell average. When neighboring cells use the same face flux, their interior contributions cancel in the global budget. SSPRK3 combines several such evaluations.

The actual reconstruction path matters. At each face I compute a Roe characteristic basis. I project a local stencil, form positive and negative Lax–Friedrichs split fluxes, and reconstruct each characteristic field. I then transform the combined numerical flux back to conserved variables. The default method therefore reconstructs characteristic split fluxes; it does not first reconstruct two separate physical Euler interface states.

In later V2, the classical TENO reconstruction remains the baseline. The gate G activates where classical TENO rejects a candidate. Lambda scales the neural correction. A zero lambda recovers classical TENO, but a nonzero correction still changes the entire coupled Euler evolution. The reconstruction and the physical wave speeds are evaluated again at every Runge–Kutta stage.
""",'U=(ρ,m,E); v=m/ρ; total energy density; Roe averages; left/right eigenvector matrices; characteristic field; LF split; shared-face flux; SSPRK3; G and λ.',"""
Explain the update as a forward-Euler building block, not the entire third-order RK formula. D(U) is defined in Backup A with the 1/Δx factor. For periodic data interior flux cancellation gives conservation; transmissive data require boundary exchange, and walls exert momentum forces. Consistency of constants is separate from conservation and high-order accuracy.

Current default: euler_jax.py:166–254. At a face the same Roe L and R are used for all stencil entries and both LF signs. The negative branch is shifted one cell and reversed. The adjacent-cell speed maximum is not automatically a bound on every state in the reconstruction stencil. The current basis inverse passed moderate-state checks; its conditioning on a failing trajectory has not been measured.

The V2 gate is classical, not a learned confidence certificate. Each LF sign/characteristic field can receive its own gate decision. The backbone is 5→64→32→16→8→3, with per-stencil min/range normalization. Soft candidate weights in V2 differ from hard mask inference in older JAX policies. The experimental state-reconstruction/Rusanov path on slide 8 is a separate formulation.
""",[
('What exactly does the neural network see?','Five neighboring values of one characteristic split flux after local min/range normalization, not the three conserved variables presented jointly. The same small classifier is applied over fields and LF directions.'),
('Is using a Roe basis the same as using a Roe Riemann solver?','No. Here Roe averages define the projection basis. LF splitting and reconstruction define the numerical flux; it is not the Roe approximate Riemann flux.')])

add(4,'Negative density and pressure',120,"""
This was a concrete failure, not simply a less accurate shock profile. The oracle-trained Keras NN-TENO5 failed Sod with non-positive density at step six. In a later V2 random Riemann case, soft strength point seven five first produced negative pressure at step thirteen, stage one. On the same short case, strength point two five completed all fifteen steps with admissible raw stages.

The pressure equation explains why positive density alone is not enough. Internal energy is total energy minus kinetic energy. The learned face correction changes density, momentum and energy together, and the subtraction can become negative. The following sound-speed calculation then leaves the real physical domain.

We tried reduced correction strength, classical blending, stage guards, repair and training penalties. These helped empirically, but they did not enforce the invariant domain of the Euler update. Nonnegative normalized candidate weights combine characteristic flux polynomials, not admissible physical states. Conservation only balances shared fluxes; it does not bound each cell’s pressure.

SSPRK3 also does not create a property that its forward-Euler building block lacks. A suitable positivity property and timestep restriction must hold at every relevant stage.

An established alternative is conservative flux limiting toward an admissible low-order flux, as in Hu, Adams and Shu. That was not implemented and validated in this neural solver. It is the kind of structural remedy I would investigate next.
""",'Admissible set; internal energy E−m²/(2ρ); positivity versus conservation; forward Euler; SSP coefficient; invariant domain; low-order admissible baseline.',"""
Oracle configuration: models/nn_teno5_oracle.h5, calibrated thresholds from its metadata, Sod N=120, L=1, T=.2, adaptive CFL=.35. The audit independently reproduced the exact density exception. The oracle supervised isolated point-value accuracy over masks on synthetic functions, not stable Euler evolution. Misclassification, objective mismatch and evolving-input shift have not been uniquely separated at the first failing face.

V2 case ID: euler-validation-000088-1023461505243801251; N=128; dt=.0017428079661349484; fifteen full SSPRK3 steps; transmissive boundaries. The separate first-invalid trace gives p_min=−.0038048537160435105 with ρ_min≈1.21304 for soft/.75. The archived batched full-validation record has a different negative minimum and a finite final state, while the direct repair study continues to a non-finite final state. Both reject the policy by raw-stage admissibility. Do not claim their complete trajectories are bit-identical.

For γ=1.4, positive pressure is equivalent to positive internal-energy density when ρ>0. The admissible set is convex, but the present characteristic flux candidates are not admissible state endpoints. Positivity-preserving flux limiting needs a correct admissible low-order update, coupled face/stage constraints, and the appropriate CFL. It is not achieved by picking an arbitrary convex coefficient.

The historical penalty-after-repair defect and its subsequent correction are discussed on the next two slides. No theorem of positivity or entropy stability exists for the present learned reconstruction.
""",[
('Can conservation and positivity fail independently?','Yes. A conservative flux difference can produce a cell with negative internal energy while conserving the implemented global budget. Conversely, overwriting a state can restore positive pressure but change mass and energy.'),
('What does SSPRK3 actually guarantee?','It can inherit a forward-Euler convex stability or invariant-set property under an appropriate step bound, if the spatial forward-Euler map has that property. Its three-stage form alone is not a positivity proof.')])

add(5,'Increasing dissipation: what the experiment did not establish',85,"""
This historical JAX trace originally encouraged a strong diagnosis. At multiplier one, density ran away and became non-finite. At one point five, growth was delayed, but density was already enormous when the stored trace ended. Multipliers two, three and five became non-finite earlier.

The numerical observation is real, but the causal interpretation was incomplete. In this experiment the timestep stayed fixed. Increasing the LF coefficient also changes the timestep restriction, roughly through alpha times delta t over delta x. In an explicit method, more diffusion is not automatically safer at the same timestep.

There was another changed variable: alpha enters the split flux itself, so the neural model received different stencil inputs. Roe eigenbasis conditioning was suspected, but the saved trace contains no condition-number history establishing it as the initiating cause.

A proper follow-up should vary alpha with a compensating timestep change, compare classical TENO5 on identical states, and record the first invalid stage, effective CFL and basis conditioning. I therefore treat the old result as a failed uncontrolled stabilization attempt, not proof that dissipation itself causes instability.
""",'LF coefficient α; αΔt/Δx; explicit diffusion restriction; confounding variable; characteristic condition number κ(R); first-invalid stage.',"""
The re-created plot uses the exact CSV records. Its horizontal axis is zero-based, unlike the one-based step number in the fresh V2 trace. At multiplier 1 the stored density is ≈1.94e37 at index 10 and becomes non-finite at 11. At 1.5 the final index 14 is ≈6.21e18 but still finite. At multipliers 2,3,5 it becomes non-finite at index 3. A cross is placed at the next non-finite step at the preceding finite height, matching its diagnostic meaning; it is not a measured density at NaN.

The old trainer had N=64, nominal assumed maximum wave speed 2, CFL=.5, fixed dt=.00390625, while random cases had varying physical speeds. The alpha multiplier experiment used the frozen clamp-only guard. It does not describe the later V2 selected-model validation. The plot’s old title overstated causality, so this deck replaces its title/caption without altering the underlying values.

Basis inversion can amplify perturbations if R is poorly conditioned, but algebraic inverse accuracy on moderate states and conditioning are different tests. A diagnostic frozen-basis or component-wise control would itself be an altered method and should be labeled accordingly. Step halving must retain the same physical final time.
""",[
('Why can more diffusion destabilize an explicit scheme?','A larger dissipative spatial eigenvalue can require a smaller timestep to stay within the time integrator’s stability region. The fixed-dt sweep changes that product instead of holding a matched stability scale.'),
('Would matched CFL prove the Roe basis is responsible if the run still fails?','No. It removes one confound. Candidate choices, input shift and nonlinear admissibility can still matter. A first-failure trace with condition numbers and controlled solver ablations is needed for attribution.')])

add(6,'Numerical repair and the conservative budget',85,"""
Repair solved a computational problem, but it changed the numerical problem. The flux-difference update cancels interior exchanges between neighboring cells. If I subsequently replace a cell’s state by its neighbors’ average, that replacement generally does not have a compensating change elsewhere.

The repair diagnostic measures the resulting integrated mass change. In a saved V2 comparison, one repaired stage row contributed about point zero zero one six one one mass and point zero zero zero eight eight nine energy. The accounting can include those artificial sources, but that is not conservation of the original Euler equations restored.

There was also a training-loss issue. A positivity penalty applied after repair could miss the violation that had just been removed. Moving that measurement before every repair corrected the signal, while gradient clipping helped contain destructive updates. Yet guarded training and unguarded validation still followed different dynamics.

The established alternative on the right is to limit a common face flux before updating adjacent cells, or recompute an admissible stage conservatively. That alternative remains to be tested in this neural implementation. A trajectory surviving artificial intervention is not enough to certify it.
""",'Physical balance; repair source; integrated mass versus density; shared numerical face flux; RK-weighted repair budget; guarded training/deployment mismatch.',"""
The displayed ΔM formula is a per-intervention diagnostic. The quoted .001611149 mass and .000888843 energy are the final RK-weighted integrated repair contributions in a fifteen-step V2 diagnostic with soft/.75, historical-repair policy, the N=128 failing case, and one changed stage row. Exact SSPRK3 final repair weights are (1/6,2/3,1) for raw increments added after stages 1,2,3. For this particular first-stage repair, this distinction matters; do not equate the quoted final contribution with an unweighted density sum at that stage.

The current historical repair averages isolated invalid cells using periodic neighbors, even when the solver boundary is transmissive. Residual invalid clusters use floors/caps. The P0 fix prevents flooring already-admissible low-positive states, but it does not make repair globally conservative. Original per-update training intervention logs are incomplete.

The penalty-after-repair defect is historical and was corrected with a pre-repair penalty accumulator. A remaining implementation mismatch exists: the current penalty is zero at exactly zero pressure and at some positive out-of-scale states that repair changes. That is not established as the cause of any archived failure; mention it only if asked. A mathematically meaningful low-order flux baseline and sufficient CFL are prerequisites of positivity limiting.
""",[
('Is a neighbor average guaranteed to be physically valid?','For admissible neighbors, convexity of the ideal-gas Euler admissible set supports positivity of their average. That does not preserve the global conserved sum when used to overwrite a different state, and the actual fallback has additional thresholds.'),
('How did you account for the repair during SSPRK3?','The final budget includes boundary flux exchanges with weights 1/6,1/6,2/3 and repair increments with weights 1/6,2/3,1. Subtracting these gives a bookkeeping residual; the repair contribution still represents a nonphysical source for the original conservative PDE.')])

add(7,'Generalization beyond short validation',120,"""
Later V2 gave encouraging validation results: the selected soft strength point two five had no invalid raw stages on three hundred and seven short Euler trajectories. Its macro L1 ratio relative to TENO5 was about point nine four eight. This score helped select the correction policy, so it is validation evidence, not a universal deployment guarantee.

The independent experiments give a mixed picture. Sod at one hundred and ninety-two cells had about four point three percent lower mean conserved-state L1 error. But density total variation increased, and a small density undershoot appeared. A lower error therefore did not mean every oscillation measure improved. The contact at the same resolution became about six point nine percent worse. The selected model failed the much stronger closed-wall blast at both tested grids, while the classical control completed it. Shu–Osher’s numerical-reference ranking remained unresolved.

Several mechanisms could explain the transfer gap. Training used short, mainly single-Riemann trajectories; the stress case had stronger pressure contrasts, interacting waves and reflective boundaries absent from the corpus. Errors also alter future Euler states and future neural inputs. We measured another mismatch: warm-up used characteristic states, whereas deployment used characteristic split fluxes. Soft and hard masks also define different numerical methods.

These are measured mismatches or candidate mechanisms, not isolated causes of the blast failure. Better deployment-aligned training and out-of-distribution tests are useful, but more data alone does not replace admissibility constraints and a reliable fallback.
""",'Macro L1 ratio; family×resolution averaging; validation selection; prospective IVP; distribution shift; total variation; undershoot; short rollout; soft/hard policy.',"""
The aggregate Euler ratio .9479383302719695 averages family×resolution error ratios, excluding constants/tiny-error controls as declared. It is not a single relative norm over all entries. The 307 independent validation IVPs use 15 SSPRK3 steps each. The later full validation chose among strengths, conditioning uncertainty on a fixed backbone, not on multiple training seeds. Two grids of one prospective IVP are not independent trajectories.

Sod N=192: TENO5 mean-state L1=.002651360802799551; selected V2=.0025380153091719496, ratio=.9572500681506942; density TV .8774663254486642→.8797561899875133; density undershoot 0→.0003506854567733031. These are against an analytic cell-average reference. Contact N=192: .0022149671223096416→.0023672257990848457, ratio≈1.06874. At N=96 the contact is approximately a tie. So do not claim that refinement uniformly helps the learned method.

Closed-wall Woodward–Colella: pressure 1000/.01/100, rho=1, v=0; N=128/256; T=.038; corrected reflective-wall fluxes; nominal CFL=.15 with initial-speed safety 2. The selected model already fails the N=128 T=.001 smoke. Classical TENO5 completes both full-time grids with admissible raw stages. No full-time blast accuracy reference was computed after the failed smoke, so failed rows get no invented error ratio.

The original full final_benchmark figure is preserved in assets. The main slide uses a readable Euler-only excerpt recreated from the same JSON; the small density inset uses the same N=192 arrays as euler_sod. The asterisk marks unresolved numerical-reference Shu ranking, not a certified win. Physical Shu entropy-wave oscillations must not be confused with numerical instability.
""",[
('Why does increased TV not necessarily mean instability?','Shu–Osher contains physical oscillations, and sharper or less damped structures can change variation. On Sod, the explicit small undershoot is stronger evidence of a nonphysical range excursion. Report norms, extrema and a common reference instead of equating TV with stability.'),
('What would make the generalization experiment more convincing?','Freeze training/policy choices before a separate family-, strength-, location-, grid- and horizon-varied test suite. Use multiple seeds, exact or controlled references, all-stage admissibility, conservation and variation diagnostics, with failures retained rather than excluded from the scientific conclusion.')])

add(8,'Finite-volume consistency and implementation limitations',80,"""
An important finding was not specific to the network. On general nonlinear smooth Euler data, the historical finite-volume split-flux path approached only second-order spatial consistency, including classical TENO5. An experimental characteristic-state reconstruction followed by a Rusanov flux approached fifth order on the same diagnostic.

The reason is the noncommutation shown here. Flux evaluated at a cell-average state is generally not the cell average of the physical flux. The historical path supplied the former to reconstruction coefficients intended for averaged data. A special constant-pressure, constant-velocity entropy wave could hide this because its flux is linear in density.

The state-flux formulation is an implemented experimental remedy, but only its smooth consistency and basic contracts were checked; shock robustness and positivity remain unvalidated. A correctly formulated finite-difference interpretation is another possibility, not a reason to ignore the finite-volume test.

There were also implementation issues: WENO7 epsilon sensitivity, and a neural wall-flux contract. Fixing those specific defects did not establish general Euler robustness.
""",'Cell average; point value; nonlinear consistency; semidiscrete divergence; truncation error; Rusanov flux; entropy wave; finite volume versus finite difference.',"""
The plot uses exact archived spatial-error values at N=40/80/160/320/640 and no time integration. For classical TENO5, last observed L1 order is 2.0119685376 for split flux versus 4.9999252974 for the experimental state flux. Gated policies give similar conclusions in this diagnostic. These are observed rates on a smooth test, not a universal fifth-order neural theorem.

For smooth U, a Taylor expansion gives overline(F(U))−F(Ubar)=(dx²/24)F''(U)[Ux,Ux]+O(dx⁴), under sufficient smoothness. Reconstructing face states first and evaluating a numerical flux avoids this particular mismatch. But those face states must then be admissible, so a consistency fix is not automatically a positivity fix. A proper finite-difference WENO method can start with pointwise physical fluxes; its semantics and flux-difference accuracy need to be established separately.

WENO7 backup: old eps=1e-6 yielded 10 non-finite coefficient rows in the stored 100k normalized stencils; eps=1e-4 yielded zero. The epsilon-squared-underflow explanation is doubtful because 1e-12 is far above float32 underflow. Cancellation/graph sensitivity remains under investigation. The reflective-wall correction explicitly imposes zero mass/energy wall flux; selected V2 blast failure persists after it. All these must be identified as distinct issues.
""",[
('Why did the fifth-order entropy-wave test not reveal this?','With constant pressure and velocity, the conserved flux is linear in the varying density. Nonlinear averaging errors vanish in that special subfamily. General variable-velocity/pressure smooth data exercise terms it does not.'),
('Did you demonstrate a fifth-order robust neural Euler solver with the new flux?','No. The optional state-flux path passes a smooth instantaneous consistency diagnostic and basic conservation/constant controls. It has not been shock-/positivity-validated or retrained as a production neural solver.')])

add(9,'Lessons and reliable numerical research',115,"""
My main lesson is to separate a learned proposal from the numerical contracts used to accept it. The project exposed negative stages, nonconservative repairs, transfer gaps and accuracy–robustness tradeoffs. Classical numerical analysis offers tools for each of these, rather than requiring the network to discover every constraint from data.

The first equation illustrates a possible direction: blend a neural face flux toward an admissible low-order flux using a shared limiter. The limiter must satisfy the coupled stage constraints and the appropriate timestep condition. An arbitrary theta between zero and one is not a positivity proof. Moment-preserving reconstruction and SSP time integration would provide other parts of the structure.

The second equation concerns independent error assessment. Given a sufficiently regular space–time reconstruction, its continuous PDE residual measures how far it is from satisfying the equation. Turning that quantity into an error bound still needs reconstruction assumptions and a suitable stability estimate; a residual sensor or network confidence alone is not a certificate.

This connects naturally to the work of Giesselmann, Kwon and Krumscheid on RKDG reconstruction and SIAC postprocessing for convection–diffusion systems. I would start with smooth settings where the assumptions can be checked, and then study what remains useful near shocks. Their result does not certify my neural Euler solver.

The research question I would like to pursue is whether learned high-order reconstruction can improve numerical accuracy while retaining provable structural constraints and independently verifiable error control.
""",'Invariant domain; convex limiting; admissible low-order update; polynomial moments; space–time reconstruction; PDE residual; relative entropy; estimator effectivity; SIAC.',"""
The limited-flux equation on the slide is a proposed architecture, not code already validated here. Shared-face limiting preserves conservative form. A mathematical positivity argument also needs the low-order invariant-domain property, a sufficient CFL bound, compatible simultaneous face choices, and treatment of each stage/boundary. The Hu–Adams–Shu limiter and invariant-domain/convex-limiting literature supply foundations, not an automatic theorem for an arbitrary network.

For a posteriori assessment, do not compute only the residual of the numerical solution in its own identical discrete update, which is essentially zero by construction. Use a separate sufficiently regular reconstruction and continuous PDE residual, including relevant initial/boundary/time terms. In an actual convection–diffusion application the residual includes diffusion terms as well. The slide’s residual is for inviscid Euler.

The verified Giesselmann–Kwon–Krumscheid preprint is arXiv:2604.01200v1, submitted 1 April 2026. It treats periodic Cartesian RKDG settings, SIAC spatial reconstruction and time reconstruction for entropy-compatible convection–diffusion systems; worked model analyses include linear advection–diffusion, nonlinear scalar problems, and a diffusive p-system. Nonlinear stability constants depend on regularity/Lipschitz information. It contains no trained neural filter and is not a general post-shock neural Euler guarantee.

Possible research: constrain learned DG corrections in a zero-mean basis; constrain learned SIAC kernels to reproduction moments; evaluate residuals independently. Unit-mass signed convolution does not ensure local positivity or preserve every cell mean. Shock gating/limiting can alter superconvergence. Conclude here, invite discussion, and use the hidden backup slides only when relevant.
""",[
('What is one concrete first research experiment?','Use a smooth scalar or Euler/p-system case with exact reference, compare classical and moment-constrained learned reconstruction, and independently compute the space–time residual. Measure actual error, estimator effectivity, order, admissibility and cost before attempting post-shock guarantees.'),
('What remains genuinely open rather than already solved by the literature?','An efficient learned high-order correction whose accepted Euler updates satisfy the appropriate structural conditions, together with useful independent error control on the intended flow class. Extending practically sharp estimates through shocks and preserving accuracy under constraints are substantive questions, not consequences of the existing code or preprint.')])

add(10,'Backup A — Detailed Euler formulation',0,"""
The characteristic speeds are velocity minus sound speed, velocity, and velocity plus sound speed. A common Roe face basis projects every stencil entry. The LF splitting uses one face coefficient across the reconstruction stencil, and the negative direction is shifted and mirrored. The two reconstructed characteristic flux branches are added and transformed back. D denotes the conservative flux difference divided by cell width. SSPRK3 evaluates this operator at its intermediate states, so every state entering a new flux calculation must be physically admissible.
""",'Roe enthalpy average; characteristic projection; LF stencil coefficient; D(U); SSPRK3 stages; ρ>0 and internal energy>0.',"""
The adjacent-cell LF maximum is an implementation choice, not a measured full-stencil bound. The common face basis is algebraically consistent, but the failing-trajectory condition number remains unmeasured. Do not call this a Roe Riemann flux. Forward-Euler conservation does not imply a positivity bound. The specific SSPRK3 convex coefficients transfer only a property that the underlying admissible forward-Euler maps satisfy under a sufficient step bound.
""",[('Why use characteristic rather than component-wise reconstruction?','It separates local wave families and is a standard shock-capturing strategy. This repository has no controlled neural component-wise-versus-characteristic comparison proving a quantified benefit or isolating the failure cause.')])

add(11,'Backup B — WENO7 numerical conditioning',0,"""
The classical WENO coefficients divide optimal weights by squared regularized smoothness indicators, then normalize. In the legacy backend, the stored hundred-thousand-row test produced ten non-finite coefficient rows with epsilon ten to the minus six and none with ten to the minus four. That supports the sampled mitigation. However, squaring ten to the minus six does not by itself underflow float32. Expanded smoothness polynomials can suffer cancellation, and a diagnostic near-flat row had a negative computed indicator. Exposing intermediate graph outputs can itself change arithmetic, so I have not isolated the exact NaN operation or established a globally stable seventh-order method.
""",'WENO-JS α and ω; smoothness β; epsilon regularization; cancellation; float32 normal range; finite-output test versus accuracy/stability.',"""
Full recipe and ten bad-row indices are in EULER_STABILITY_RESEARCH_AUDIT.md §10.4. Commit 753d526 raises epsilon. Rejected candidate Euler failures within 1–3 steps and five seeds are historical descriptions without complete surviving logs. The fresh coefficient test is stronger direct evidence for the layer defect, but is not a reproduction of all historical training runs. A stable beta formula, precision comparison, nonnegative treatment and an order/critical-point study are worthwhile follow-ups. The instrumented first beta example near −9.54e-7 is cancellation evidence, not a fully isolated cause of the original full-graph NaN.
""",[('Does the epsilon change affect accuracy?','It can change the smoothness weighting and sensitivity. The zero-NaN count establishes sampled finiteness only; a separate asymptotic/critical-point accuracy study is needed.')])

add(12,'Backup C — DG, SIAC and error estimates',0,"""
DG makes the local polynomial representation explicit. A SIAC-type convolution can supply a smoother reconstruction, with polynomial reproduction governed by kernel moments. I would constrain any learned correction to preserve the required moments, and evaluate the PDE residual independently. A unit-mass kernel preserves a global integral in an appropriate periodic setting; it need not preserve each cell average, and signed kernels need not preserve positivity. The 2026 RKDG–SIAC paper supplies an error-analysis connection under its own hypotheses, not a trained neural filter or a general Euler shock certificate. Near shocks, large reconstruction gradients can make stability constants and estimates much less useful.
""",'DG modal coefficients; kernel convolution; reproduction moments; unit mass; relative entropy; reconstruction regularity; effectivity.',"""
For the moment formula, r=1,…,s is a finite selected reproduction order, not every positive integer. A learned local DG correction can preserve each cell average by having zero cell mean; convolution with a unit-mass kernel is a different construction. The first-moment condition is sensitive to kernel convention and shift; specify the desired polynomial reproduction under the chosen coordinates.

Primary references: Hu/Adams/Shu JCP 242 (2013) 169–180, DOI10.1016/j.jcp.2013.01.024; Zhang/Shu JCP231(2012)2245–2258, DOI10.1016/j.jcp.2011.11.020; Gottlieb/Shu/Tadmor SIAMReview43(2001)89–112, DOI10.1137/S003614450036757X; Guermond/Nazarov/Popov/Tomas SISC40(2018)A3211–A3239, DOI10.1137/17M1149961; Giesselmann/Kwon/Krumscheid arXiv2604.01200v1. Full verified links and scope are in Sources_and_Evidence.md.
""",[('Could a learned SIAC filter simply smooth the shock?','It might, and small residual or visual smoothness alone is insufficient. Accuracy, physical admissibility, shock location, boundary treatment and the estimator’s assumptions must be checked. Restricting/gating the filter near shocks may also change its superconvergence behavior.')])

from notes_revisions import revise_notes
revise_notes(notes)

def main():
    (HERE/'notes_content.json').write_text(json.dumps(notes,indent=2,ensure_ascii=False)+'\n')
    total=sum(n['seconds'] for n in notes.values())
    words={k:len(re.findall(r"\S+",n['script'])) for k,n in notes.items()}
    lines=['# Speaker notes — KIT research interview','',
       '**Main talk: 9 slides, scheduled 13 minutes 15 seconds.** Backups A–C are outside that timing and hidden in the PowerPoint slideshow.',
       '', 'Use the spoken scripts below; preparation notes and Q&A are **not** part of the timed talk. Equations are explained in words rather than read symbol by symbol. The timing is a rehearsal estimate, not a measured human delivery.',
       '', '| Slide | Duration | Cumulative | Spoken words |','|---|---:|---:|---:|']
    elapsed=0
    for k,n in notes.items():
        if int(k)>9:continue
        elapsed+=n['seconds'];lines.append(f"| {k}. {n['title']} | {n['duration']} | {elapsed//60}:{elapsed%60:02d} | {words[k]} |")
    lines+=['',f"Total main spoken script: **{sum(words[str(k)] for k in range(1,10))} words**. The scheduled pace includes pauses for the equations and plots. Rehearse once aloud; do not add the Q&A or backup scripts to the main talk.",
       '', 'The final slide is the end of the planned presentation. Invite discussion there. For PDF delivery use the Main_Only PDF when you want a hard stop after slide 9; the complete PDF retains the three backups.']
    for k,n in notes.items():
        heading=f"Slide {k} — {n['title']}" if int(k)<=9 else n['title']
        lines += ['',f"## {heading}",'',f"Estimated speaking duration: **{n['duration']}**.",'','### Spoken script','',n['script']]
        if int(k)<=9:
            lines += ['','### Equation guide — not spoken','',n['equation_guide'],'','### Established observations — not spoken','',n['observed'],'','### Mechanisms, uncertainty and remedies — not spoken','',n['limits_and_remedies']]
        lines += ['','### Mathematical terms to remember','',n['terms'],'','### Preparation and evidence boundaries — not spoken','',n['preparation'],'','### Possible technical questions','']
        for q in n['qa']:lines.extend([f"**{q['q']}**",'',q['a'],''])
    lines += ['## Delivery checklist','',
        '- Point to the first invalid RK stage, not only the final profile.',
        '- Say “observed,” “hypothesized,” and “not yet implemented” explicitly when switching evidence levels.',
        '- Keep Keras oracle, historical JAX and later V2 results separate.',
        '- Preserve negative results: contact regression, undershoot, failed blast, unresolved Shu reference.',
        '- Distinguish admissibility, conservation, entropy stability, consistency and error control.',
        '- End at the research question; open a backup only in response to a relevant question.']
    (HERE.parent/'Speaker_Notes_REVISED.md').write_text('\n'.join(lines)+'\n')
    print('Main duration',total,'seconds; words',sum(words[str(k)] for k in range(1,10)),'; per slide',words)

if __name__=='__main__':main()
