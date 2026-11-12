"""Targeted notes changes; numerical evidence is inherited from the frozen audit.

Imported by write_notes_REVISED.py. Scripts alone comprise the timed talk.
Equation guides, preparation, evidence boundaries and Q&A are rehearsal material.
"""
import textwrap

def clean(value):
    return textwrap.dedent(value).strip()

def revise_notes(notes):
    seconds=[25,60,95,135,90,85,120,80,105]
    scripts={
    1:"""
    Thank you for meeting with me. I’m Yasser Baouzil, and I’ll present my research internship on neural shock capturing. I’ll explain what I built, then focus on the Euler failures: what the evidence establishes, why several fixes were insufficient, and how numerical structure could make learned methods more reliable.
    """,
    2:"""
    My technical work began with a modular conservation-law solver: an ENO baseline and WENO and TENO reconstructions of orders three, five and seven. I integrated neural reconstruction decisions, characteristic Euler reconstruction, convergence checks, and benchmarks including Sod and Shu–Osher. I also worked with differentiable JAX rollouts, classical–neural blending and investigations of numerical failures.

    The conservation law says that local changes are balanced by physical fluxes. Keeping that structure visible made it possible to ask which numerical properties learning preserved.

    The timeline separates four generations: original WENO coefficient corrections, neural TENO and oracle candidate selection, historical JAX rollout training, and later gated V2. I distinguish those checkpoints and later validation work throughout; the results do not all belong to one network or one internship-stage experiment.
    """,
    3:"""
    The Euler state contains density, momentum and total energy. Velocity is momentum divided by density. Pressure couples all three entries in the physical flux.

    The update on the right is the forward-Euler building block. A common face flux leaves one cell and enters its neighbor, so interior contributions cancel in the discrete budget. SSPRK3 composes several such evaluations.

    At each face, the actual code uses a common Roe characteristic basis, forms Lax–Friedrichs split fluxes, and reconstructs each characteristic field. The outlined block is where the neural decision enters. The combined flux is then transformed back into conserved variables. The default solver reconstructs characteristic split fluxes, rather than first constructing separate physical left and right states.

    Later V2 starts with classical TENO and adds a gated, scaled neural correction. G is a classical trouble gate and lambda is the correction strength. Zero lambda recovers classical TENO. This retains conservative form, but it does not establish positive Euler states. Reconstruction and wave speeds are evaluated again at each Runge–Kutta stage.
    """,
    4:"""
    These were physical-admissibility failures, not merely less accurate shock profiles. The oracle Keras NN-TENO5 failed Sod with non-positive density at step six. Later V2, at soft strength point seven five, produced negative pressure at step thirteen, stage one. Strength point two five completed the same fifteen-step case admissibly.

    Pressure is proportional to total energy minus kinetic energy. The boxed expression linearizes pressure around an admissible reference state. An energy error enters directly; a momentum error is multiplied by minus velocity; a density error enters with half the velocity squared. This is a local sensitivity formula, not a proof of the unique error responsible for the recorded failure.

    A neural face-flux correction changes density, momentum and energy in the next stage. Their coupled error can make internal energy negative. Then the sound speed, which involves the square root of pressure over density, and characteristic calculations can break down.

    We tried reduced corrections, blending, guards, repair and penalties. These were empirical safeguards. Normalized neural weights combine characteristic flux polynomials, not admissible Euler states. Conservation balances transfers but does not bound each cell’s pressure. SSPRK3 can transfer a suitable forward-Euler positivity property under its timestep restriction; it cannot create that missing property.

    An established remedy is conservative positivity-preserving flux limiting toward an admissible low-order update. It needs appropriate face factors, stage constraints and CFL assumptions. That remedy was not implemented or validated in this neural solver.
    """,
    5:"""
    Here the LF multiplier increased while the timestep stayed fixed. Multiplier one ran away and became non-finite. One point five delayed the growth but still reached enormous density. Multipliers two, three and five diverged earlier.

    This does not prove that additional dissipation inherently destabilizes Euler. The experiment was confounded: dissipation, the effective timestep restriction, and neural inputs changed simultaneously. The first equation shows that alpha changes the split flux. The second shows the indicative stability scale, alpha times delta t over delta x. Increasing alpha without reducing the timestep can move an explicit method outside its stable regime, and it also changes the network’s inputs.

    Poor Roe-basis conditioning was suspected, but no first-failure condition-number trace established it as the initiating cause.

    The next diagnostic is a matched-CFL comparison at a common final time, with a classical TENO5 control, first-invalid-RK-stage tracing and characteristic condition numbers. This removes specific confounds; it still would not identify a unique cause without controlled ablations.
    """,
    6:"""
    Repair helped some training trajectories continue, but it changed the numerical budget. In the conservative update, a shared face transfer cancels between neighboring cells. Overwriting an invalid cell afterward generally has no compensating transfer elsewhere.

    The diagnostic on the right measures the integrated mass added by one intervention. The quoted numbers are more specific: the saved fifteen-step comparison has one repaired stage, and its final RK-weighted budget source is about point zero zero one six one one mass and point zero zero zero eight eight nine energy. These are not the raw unweighted stage increments or solution-error norms.

    Historical penalties measured after repair could miss the violation just removed. Pre-repair penalties corrected that signal, but guarded training still differed from unguarded validation. Numerical survival was therefore not physical reliability.

    Conservative positivity limiting instead constrains a shared face flux before updating both cells. Conservation follows from that common flux; positivity additionally requires an admissible low-order baseline, suitable correction factors and timestep bounds at each stage. This established alternative was not implemented or validated here. Existing state repair does not make the learned method positivity preserving.
    """,
    7:"""
    Selected V2 soft strength point two five completed three hundred and seven short Euler validation trajectories without invalid stages. Its macro L1 ratio was point nine four seven nine four. That score helped select the policy, so it is validation evidence.

    Independent results were mixed. Sod at one hundred and ninety-two cells improved mean-state L1 by about four point three percent, but density total variation increased and a small undershoot appeared. The inset now explicitly zooms just right of the shock to show that excursion; it uses the same saved arrays. The contact regressed. Shu–Osher’s reference ranking is unresolved, so I show no accuracy point for it.

    The stronger Woodward–Colella blast is the central failure: selected V2 became non-finite at both grids, while classical TENO5 completed both controls. There is no invented accuracy score for a failed run.

    Potential contributors include stronger shocks, interacting waves and reflective walls absent from training. A reconstruction error also changes the next state, pressure and wave speeds; that changes the next characteristic projection and neural input. Warm-up used characteristic states rather than deployed split fluxes. These are measured mismatches or plausible feedback mechanisms, not isolated causes of the blast failure.

    Better data can improve coverage, but cannot alone guarantee admissibility. Good short-horizon validation did not establish stability under stronger out-of-distribution Euler dynamics.
    """,
    8:"""
    This limitation also affected classical TENO5. On general nonlinear smooth Euler data, the historical finite-volume split-flux path approached spatial order two point zero one. Experimental state reconstruction with a Rusanov flux approached five on the same diagnostic.

    The equation explains the mismatch: averaging a nonlinear physical flux is generally different from evaluating it at the average state. This particular historical path supplied fluxes of cell-average states to reconstruction with averaged-data semantics. A constant-pressure, constant-velocity entropy wave could hide the problem because its flux is linear in density.

    This is not a statement that every LF split-flux WENO method is second order. Proper finite-difference formulations have a different interpretation.

    The experimental state-flux remedy was checked for smooth spatial consistency, not full time-dependent shock robustness or positivity. A consistency improvement still needs separate Euler stability validation. The WENO7 and wall-contract issues remain separate implementation findings, covered in the notes and backups.
    """,
    9:"""
    I would separate the learned proposal from the mathematical conditions used to accept it. We observed negative stages, artificial repair sources, limited transfer and accuracy–oscillation tradeoffs. Established numerical methods address these properties explicitly.

    The first equation limits a neural face flux toward a suitable low-order flux. Shared-face use preserves conservative form. Positivity also needs an admissible low-order update, appropriate limiter factors and the relevant CFL. An arbitrary theta between zero and one is insufficient. Conditional SSP integration and structure-preserving reconstruction supply other parts of the method.

    The second equation is the continuous PDE residual of a space–time reconstruction. It can support independent a posteriori assessment, but a residual alone is not a reliable error bound: reconstruction assumptions and stability estimates are essential.

    This connects to Giesselmann, Kwon and Krumscheid’s RKDG–SIAC work on convection–diffusion systems. It does not certify this neural Euler solver. A concrete extension would begin with smooth admissible states and checkable reconstruction moments, then examine estimator effectivity and what changes near shocks. Those learned DG or SIAC extensions remain proposals.

    Can learned high-order reconstruction improve numerical accuracy while retaining provable structural constraints and independently verifiable error control?
    """}
    guides={
    1:'The unchanged cover shows a scalar conservation law. It is context; the Euler vector equation is introduced on slide 3.',
    2:'∂tU+∂xF(U)=0 expresses local conservation. U is a scalar or conserved vector; F is its physical flux. A numerical shared-face balance is a separate discrete construction.',
    3:'U=(ρ,m,E) contains density, momentum density and total energy density; v=m/ρ. F is the physical Euler flux. The flux-difference equation is one forward-Euler step. In R_V2=R_T+Gλ(R_NN−R_T), R_T is classical reconstruction, G the classical trouble gate and λ the correction strength. G is not a neural confidence certificate.',
    4:'For γ>1 and ρ>0, p>0 is equivalent to E−m²/(2ρ)>0. Differentiation gives ∂p/∂E=γ−1, ∂p/∂m=−(γ−1)v and ∂p/∂ρ=(γ−1)v²/2. Therefore δp≈(γ−1)(δE−vδm+v²δρ/2), with all coefficients evaluated at the admissible reference state. The neglected remainder is second order in small state perturbations. The formula is a sensitivity explanation, not an exact finite-perturbation test. Sound speed c=sqrt(γp/ρ) requires valid pressure and density.',
    5:'F±=(F±αU)/2 defines LF splitting; the superscript sign selects the same sign in the bracket. CFL_LF≈αΔt/Δx is an indicative stability scale, not a proven global CFL limit for the arbitrary learned flux. At fixed Δt, increasing α changes both this scale and network inputs.',
    6:'The flux-difference update telescopes over interior faces. ΔM_repair=Δx Σ(ρ_after−ρ_before) defines a single intervention’s mass source. The quoted record instead sums RK-weighted repair exchanges over the 15-step rollout: per-step weights are (1/6,2/3,1) for stage-1/2/3 repair increments. Only one stage row is repaired in this case. Conservation_residual subtracts repair as an artificial source; physical_balance_error does not.',
    7:'r_L1=E_L1(NN)/E_L1(TENO5); values below one improve that defined norm. The validation 0.94794 is a macro family×resolution ratio, not a pooled norm or universal success rate. TV(ρ)=Σ|ρ_(i+1)−ρ_i| measures density variation, not entropy stability. Sod’s explicit undershoot is a range excursion; it does not make its density negative.',
    8:'Overline(F(U)) generally differs from F(Ubar) because F is nonlinear. For sufficiently smooth U, the leading difference is (Δx²/24)F″(U)[U_x,U_x]+O(Δx⁴). The plotted errors compare semidiscrete spatial divergence with exact cell-average divergence; time truncation error is absent. These measured orders concern this formulation/test.',
    9:'F_limited=F_LO+θ(F_NN−F_LO) is a proposed conservative correction architecture. θ is constrained by simultaneous face/stage admissibility conditions, not only 0≤θ≤1. R=∂t Uhat+∂xF(Uhat) is the inviscid continuous residual. A convection–diffusion residual has extra diffusion terms. Its relation to error requires independent reconstruction and a valid stability/error analysis.'}
    observations={
    1:'No numerical result is claimed by the aircraft image. It is V4’s original illustration.',
    2:'Source and benchmark artifacts establish the modular framework, neural integration and separate method generations. Convergence and failure investigations do not imply all implementations attain their named formal orders.',
    3:'The default code reconstructs characteristic split fluxes in a common Roe face basis. At zero V2 correction strength it recovers classical TENO. No controlled neural component-wise comparison or first-failure basis-conditioning trace establishes a unique basis defect.',
    4:'Oracle Keras NN-TENO5: Sod N120, non-positive density at step 6. V2 diagnostic: soft/.75 first invalid pressure at one-based step 13, stage 1; soft/.25 completes the same 15 steps. These belong to different checkpoints/generations.',
    5:'The stored fixed-dt sweep has non-finite zero-based indices 11 for 1× and 3 for 2×/3×/5×. The 1.5× run remains finite through index 14 but density is about 6.21e18. It is not a stabilized success.',
    6:'repair.json, soft/.75 historical-repair: repaired_stage_rows=1; integrated_repair_exchange=[0.0016111493654163188,−0.00005706669322759772,0.0008888429518660344]. The source code proves this field is accumulated RK-weighted exchange, not an unweighted stage diagnostic. Physical balance error equals this source up to roundoff.',
    7:'307 short validation trajectories pass. Independent Sod improves mean-state L1 but worsens TV/undershoot; contact N192 regresses; selected V2 blast N128/N256 becomes non-finite while TENO5 completes. Shu’s uncertain reference does not resolve a ranking.',
    8:'Archived classical-TENO5 last spatial L1 orders are 2.0119685376 and 4.9999252974. The same general nonlinear smooth diagnostic and cell-average reference are used; this is not a temporal order experiment.',
    9:'No learned DG/SIAC method or positivity-limited neural Euler implementation has been validated in this project. The proposed constrained architecture and independent assessment are research directions, not additional experimental results.'}
    limits={
    1:'The image is not an internship CFD experiment. Introduce the evidence-driven scope without promising a universally improved solver.',
    2:'Early scalar results do not establish system-level stability. Distinguish personal framework/integration work from established classical schemes, later generations and the subsequent audit. The ENO baseline is not claimed to have verified orders 3/5/7.',
    3:'Conservative form and characteristic separation do not constrain the Euler invariant set. The nearby-cell LF speed bound is an implementation choice; its adequacy on every failing stencil remains unmeasured. Structural face/stage control is the relevant proposed addition.',
    4:'The local sensitivity equation does not identify a unique failing face or classifier decision. Reduced strength, guards, repair, blending and penalties helped some configurations but imposed no invariant-domain theorem. Known conservative positivity limiting is unimplemented/unvalidated here and requires a suitable low-order baseline and CFL.',
    5:'The sweep changes three coupled factors, so it cannot isolate dissipation or Roe conditioning. A matched-CFL/classical-control study is proposed; it has not been run for this revision. More diffusion is not inherently unstable.',
    6:'Repair made some trajectories finite and pre-repair penalties fixed a historical signal defect. Neither restores conservative Euler dynamics or enforces a positivity-preserving spatial map. Shared-face positivity limiting is established literature but was not implemented or validated here.',
    7:'Training strength, horizon, wave interactions, walls and input mismatch are candidate transfer mechanisms, not proven unique blast causes. Added coverage can improve empirical transfer; admissibility still needs numerical constraints and an independently validated fallback.',
    8:'Consistent FV state/Rusanov reconstruction fixes this observed smooth averaging-semantic mismatch; shock robustness, positivity and neural retraining remain unvalidated. Properly derived FD LF split-flux WENO is not ruled out by this result.',
    9:'Unit-mass SIAC convolution need not preserve each cell mean or positivity. A residual sensor is not a certificate; stability constants can deteriorate near shocks. Start where reconstruction, state bounds and moment assumptions can be checked before proposing broader guarantees.'}
    qa={
    4:[('How can a conservative method produce negative pressure?',
        'Conservation is a telescoping shared-face balance; it does not constrain each updated internal energy. A flux correction can change E, m and ρ so that E−m²/(2ρ) becomes negative. Normalized characteristic-flux weights are not convex combinations of admissible Euler states.'),
       ('Why does SSPRK3 not automatically preserve positivity?',
        'Its convex-stage representation can inherit an invariant-set property of the forward-Euler spatial map under a suitable step bound. If that map is not positivity preserving, the SSP formula alone supplies no such guarantee.')],
    5:[('Why did more LF dissipation fail to stabilize the experiment?',
        'Alpha increased at fixed Δt, changing the explicit stability scale and the neural split-flux inputs simultaneously. Earlier divergence is observed, but it is not proof that dissipation itself is harmful. Match CFL and add a classical control before attributing the cause.')],
    6:[('Why did direct repair break conservation?',
        'Overwriting a cell adds an uncompensated increment after the conservative face-flux update. The recorded final RK-weighted source changes mass and energy; subtracting it in bookkeeping does not restore conservation of the original PDE.'),
       ('How does positivity-preserving conservative flux limiting work?',
        'Start from a low-order update known to be admissible under a sufficient CFL, then limit the shared-face high-order correction using coupled density/internal-energy constraints at each relevant stage. The shared face preserves conservative form; carefully chosen factors and assumptions provide admissibility.')],
    7:[('Why did the neural model fail the blast problem?',
        'The failure is established, but a unique cause is not. Stronger pressure contrasts, interactions, reflective walls, short training horizons and input mismatch are plausible contributors. First-invalid-stage traces and controlled distribution/solver ablations are needed to separate them.'),
       ('Why is more training data not a positivity guarantee?',
        'Coverage can reduce typical prediction errors, but a finite sample does not enforce the Euler invariant set for every future state. Errors change pressure and wave speeds, feeding back into later inputs. A guarantee needs an admissibility-preserving numerical acceptance mechanism under stated assumptions.')],
    8:[('What exactly was the finite-volume consistency problem?',
        'The historical cell-average path used F(Ubar) as reconstruction data with averaged-flux semantics, although generally F(Ubar) differs from averaged F(U). Classical TENO5 showed the same O(Δx²) limitation. The state/Rusanov alternative reached order five only on this smooth spatial diagnostic.')],
    9:[('How does this connect to DG, SIAC and a posteriori estimation?',
        'DG makes polynomial moments explicit; constrained learned corrections or SIAC-type filters could preserve selected reproduction conditions. A sufficiently regular space–time reconstruction allows an independent continuous residual. Turning it into a reliable bound needs the appropriate stability, regularity and boundary assumptions; the 2026 paper does not certify this neural Euler method.')]
    }
    for i in range(1,10):
        n=notes[str(i)];n['seconds']=seconds[i-1]
        n['duration']=f"{n['seconds']//60}:{n['seconds']%60:02d}"
        n['script']=clean(scripts[i]);n['equation_guide']=guides[i]
        n['observed']=observations[i];n['limits_and_remedies']=limits[i]
        if i in qa:n['qa']=[{'q':q,'a':a} for q,a in qa[i]]
    notes['4']['terms']+='; first-order pressure variation; admissible reference state; pressure gradient in conserved variables.'
    notes['7']['preparation']=notes['7']['preparation'].replace('307 independent validation IVPs','307 validation trajectories')
    notes['7']['preparation']=notes['7']['preparation'].replace('The asterisk marks unresolved numerical-reference Shu ranking, not a certified win.',
        'The revised Shu row contains no accuracy point or ratio because its numerical-reference ranking remains unresolved. The revised inset explicitly zooms x=.776–.805 and ρ=.12445–.12615, without modifying any saved array.')
    notes['7']['preparation']=notes['7']['preparation'].replace('The main slide uses a readable Euler-only excerpt recreated from the same JSON; the small density inset uses the same N=192 arrays as euler_sod.',
        'The revised main slide uses six resolved Euler ratios from the same JSON and a separate native blast-failure callout; the enlarged shock-tail inset uses the same N=192 arrays as euler_sod.')
