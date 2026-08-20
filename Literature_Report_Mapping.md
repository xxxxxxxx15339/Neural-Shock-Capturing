# Literature Mapping — Which Papers Support Which Report Chapter

This maps the report structure already agreed on to the literature review
(7 papers read in full text, 1 read via search-level summary only — marked
below). For each chapter: which paper(s) apply, and *why* — not just that
they're topically related, but what specific claim, mechanism, or finding
in this project's own work each paper backs up or contextualizes. One gap
is flagged explicitly rather than papered over: the classical numerics
background (WENO/TENO themselves) isn't covered by this search at all — it
was scoped to the ML-training-methodology question, not the underlying
schemes — and still needs its own references (Jiang & Shu 1996 for WENO,
Fu et al. for TENO) before Chapter 2 is complete.

**Added after the professor's own answers to the 5 open questions
(Results.md Phase 12)**: Sun, Inaba & Xiao (2016), the BVD paper the
professor pointed to directly for Q4 (`L_BVD`). Unlike the other 6, this
one was read *because* it was assigned, not found by an independent
search — and it directly corrected a real implementation mistake (see
Chapter 5/6 entries below), not just added context.

Full citations are collected once at the bottom; inline references use
`[Author, Year]`.

---

## Front matter (Abstract)

No paper is cited directly in an abstract, but its one-sentence framing of
the contribution should reflect what the review clarified: this is not a
claim of a novel training paradigm (unrolled/differentiable-physics
training is established — `[Um et al., 2020]`, `[Kochkov et al., 2021]`,
`[Bar-Sinai et al., 2019]`), it's a specific application of that paradigm
to neural TENO/ENO candidate selection (where the closest prior work,
`[Wang et al. (NENO6), 2022]`, never attempted it), plus a documented
obstacle extending it to systems of conservation laws.

---

## Chapter 1 — Introduction

No direct citation needed here beyond a forward reference to Chapter 2's
gap and Chapter 4's motivation. Keep this chapter free of literature —
it's scene-setting, not argument.

---

## Chapter 2 — Theoretical Background

**Gap, stated plainly**: none of the 7 papers in this review are the right
citation for WENO/TENO numerics themselves — the search was scoped to
"how is a neural correction trained," not "what is a WENO/TENO scheme."
This chapter still needs the classical references this project's own code
comments already point to implicitly (Jiang & Shu's original WENO paper;
Fu, Hu & Adams's TENO papers) — not sourced in this review, source them
separately before finalizing.

**What this review does contribute here**: `[Wang et al. (NENO6), 2022]`
is worth introducing in this chapter, not just later — it's the paper
`src/networks/TenoNetworks.py`'s architecture is explicitly modeled on
(6→64→32→16→8 MLP for TENO6; this project's 5→64→32→16→8→3 for TENO5 is
the same design one candidate narrower), so it belongs wherever the report
first explains what "NN-TENO5" even is.

---

## Chapter 3 — State of the Project at the Start of the Internship

This is where `[Wang et al. (NENO6), 2022]` does its heaviest lifting.
Three specific facts from that paper support three specific claims about
the pre-existing NN-TENO5 baseline:

- **Architecture provenance**: confirms NN-TENO5's architecture is a
  documented, published design, not an ad hoc choice.
- **Training paradigm**: NENO6's own labels are generated exactly the way
  this project's original `Train_NNTENO5.py` does it — "what would the
  classical scheme itself decide" — confirming the imitation-training
  baseline this project inherited is the same paradigm as the closest
  published prior work, not a simplified/naive version of it.
- **Reported result shape**: NENO6's own numbers (2nd-order accuracy on
  smooth problems vs. the teacher's 4th order, roughly-parity on shocks)
  match this project's own finding for the imitation-trained baseline
  (5/6 "wins," but only noise-level margins, per `Results.md`) — this is
  independent, published confirmation that imitation training's ceiling
  isn't specific to this codebase.

`[Bar-Sinai et al., 2019]` also belongs here for one specific, quotable
claim relevant to *why* the project moved to oracle/exact-solution
training in its own Phase 2 (well before proj.pdf): *"a low value of...
training loss is a necessary but not sufficient condition for accurate and
stable numerical integration over time."* That's the general form of the
non-identifiability problem this project's own Phase 2 (WENO7-NN) and
Phase 7-9 (NN-TENO5 oracle) independently rediscovered.

---

## Chapter 4 — The Professor's Proposal (proj.pdf)

`[Um et al. (Solver-in-the-Loop), 2020]` and `[Kochkov et al., 2021]` are
the chapter's core citations — they establish that proj.pdf's central idea
(train by running the network through the actual solver and judging the
outcome, not by imitating a fixed target) is a real, working, published
methodology in the broader differentiable-physics literature, with
Kochkov et al. as the strongest existing proof that it can work at real
scale (2D turbulent Navier-Stokes, 40-80x speedup at equivalent accuracy).
This is the right place to note explicitly: proj.pdf is asking this
project to apply an established *general* idea to a *specific*, previously
untried case (TENO candidate selection) — that framing matters for how
ambitious the plan actually is.

`[Ross, Gordon & Bagnell (DAgger), 2011]` belongs here as the theoretical
grounding underneath Solver-in-the-Loop's empirical finding — the
imitation-vs-rollout distinction proj.pdf is proposing is a specific
instance of a decade-plus-old, formally analyzed problem in sequential
prediction generally, not a numerics-specific insight. Citing it shows the
plan's core mechanism has a rigorous foundation outside CFD.

---

## Chapter 5 — Methodology (the differentiable JAX implementation)

`[Um et al., 2020]` and `[Kochkov et al., 2021]` again, but for a
different reason here: both built differentiable solvers as infrastructure
before training anything, exactly as this project's `src/differentiable/`
rewrite had to. Worth citing as precedent for *why* a from-scratch
differentiable rewrite was necessary rather than a training-script change
— this is a recognized, unavoidable prerequisite in this line of work, not
over-engineering.

`[Singha, 2025, "Learning Under Laws"]` belongs here too, specifically
next to the `euler_positivity_repair` design (the professor's neighbor-
averaging repair, which replaced the earlier simple clamp — see Results.md
Phase 12): it's the paper that would justify a *more* rigorous version of
what was actually built — cite it as the documented target design, while
being explicit in the text that only the neighbor-average repair plus a
last-resort floor was implemented and tested, not the full
constraint-projection scheme.

`[Sun, Inaba & Xiao, 2016]` belongs here too, and specifically: this is
the one citation in the whole review that isn't context or precedent, it's
a *correction*. `L_BVD`'s first implementation (reconstructing a stencil
forwards and reversed in place) was a plausible-looking but wrong reading
of "boundary variation" — this paper's actual definition (each cell's own
reconstruction compared against its *neighbor's* own reconstruction, not
against a mirror of itself) is what `numerics_jax.boundary_variation_jax`
and `euler_jax.euler_boundary_variation_jax` now implement. Worth stating
in the text plainly: this fix came from the professor naming the source
directly, not from further self-directed literature search — a useful,
honest data point on where independent reading gets you and where it
doesn't.

---

## Chapter 6 — Experimental Results

This is where the parallels are the most direct, and where each citation
should sit next to the specific number or plot it explains:

- Next to the scalar run 1→3 table: `[Um et al., 2020]`'s quote on
  pre-training with small look-aheads before longer rollouts — the
  independent discovery of the same pretrain-then-anchor pattern belongs
  as a footnote right where that result is reported, not just in the
  discussion chapter.
- Next to the Euler density-blowup plot (`04_euler_instability.png`):
  `[Um et al., 2020]`'s near-verbatim description of the same failure
  ("exponential increases of existing oscillations and a diverging
  calculation") — cite directly under the figure.
- Next to the Phase 11 curriculum result: `[Um et al., 2020]` again (the
  fix that was tried) and `[Singha, 2025, "Learning Under Laws"]` (why
  it wasn't enough).
- Next to the validation-matrix "narrow, not general" result
  (`03_validation_matrix.png`): `[McGreivy & Hakim, 2024]` — this is the
  right place to note explicitly that this result is being reported
  *because* of that paper's critique, not despite it: a positive result
  scoped honestly to where it was actually tested, rather than presented
  as unconditional.

---

## Chapter 7 — Critical Discussion

`[McGreivy & Hakim, 2024]` is the anchor citation for this whole chapter —
its finding (79% of ML-for-fluid-PDE papers compare against a weak
baseline; heavy publication bias toward positive results) is the explicit
standard this project's own reporting should be checked against: TENO5 is
a real, non-trivial baseline, and the Euler failure and validation-matrix
loss are reported alongside the scalar win, not omitted.

`[Bar-Sinai et al., 2019]` and `[Singha, 2025, "Learning Under Laws"]`
both belong here for the same reason: each names, independently, a
specific limitation this project also hit (training-loss/stability
disconnect; clamping-insufficient-for-multi-constraint-systems) — citing
both together makes the point that neither obstacle is unique to this
implementation, which matters for how the Euler result should be
interpreted (a known category of difficulty, not a coding failure).

`[Wang et al. (NENO6), 2022]` returns here too, for the comparison this
project is best positioned to make and the literature doesn't: NENO6 never
attempted rollout-based training or reported systems-of-equations training
stability at all. Noting that gap directly is a real, citable
contribution of this project's own critical reading, not just a
literature summary.

---

## Chapter 8 — Conclusion and Perspectives

`[Singha, 2025, "Learning Under Laws"]` is the primary forward-looking
citation — its constraint-projection mechanism is the concrete, specific,
published next step for the Euler obstacle, more precise than a generic
"more work is needed."

`[Um et al., 2020]`'s curriculum result, now that Phase 11 tested it
directly and found it insufficient alone, supports a specific claim in
this chapter: the obstacle is structural, not a training-schedule
tuning problem — say this with the Phase 11 evidence in hand, not as a
hedge. Worth noting explicitly: `[Singha, 2025]`'s own method doesn't use
constraint projection *instead of* a rollout curriculum — it uses both
together ("a rollout curriculum that enforces consistency over long
prediction horizons," alongside constraint projection). That matches
Phase 11's own implication: curriculum learning is a real, worthwhile
piece (it measurably delayed the blowup) but was only ever going to be
half the fix — the next experiment is projection *added to* the existing
curriculum script, not a replacement for it.

---

## Not deeply read — flagged, not silently omitted

`[Anonymous, "Learning second-order TVD flux limiters using differentiable
solvers", 2025]` (arXiv:2503.09625) came up in the search as directly
relevant to the `L_TV` loss term, but its PDF would not parse through the
available fetch tooling (binary/compressed content, not the paper's text)
— only a search-snippet-level description was obtained, not a full read.
**Do not cite specific claims from it** until it's been read properly (try
an HTML mirror, e.g. ar5iv, or the published journal version if not on
arXiv in a parseable form) — it's listed here so it isn't forgotten, not
because it's been verified.

---

## Full citations

- Wang, Q. et al. (2022). *A six-point neuron-based ENO (NENO6) scheme for
  compressible fluid dynamics.* arXiv:2207.08500.
- Um, K., Brand, R., Fei, Y., Holl, P., Thuerey, N. (2020). *Solver-in-the-
  Loop: Learning from Differentiable Physics to Interact with Iterative
  PDE-Solvers.* NeurIPS 2020. arXiv:2007.00016.
- Kochkov, D., Smith, J. A., Alieva, A., Wang, Q., Brenner, M. P., Hoyer,
  S. (2021). *Machine learning–accelerated computational fluid dynamics.*
  PNAS 118(21). DOI: 10.1073/pnas.2101784118.
- Bar-Sinai, Y., Hoyer, S., Hickey, J., Brenner, M. P. (2019). *Learning
  data-driven discretizations for partial differential equations.* PNAS
  116(31), 15344-15349. DOI: 10.1073/pnas.1814058116.
- McGreivy, N., Hakim, A. (2024). *Weak baselines and reporting biases
  lead to overoptimism in machine learning for fluid-related partial
  differential equations.* Nature Machine Intelligence 6(10), 1256-1269.
  arXiv:2407.07218.
- Singha, M. (2025). *Learning Under Laws: A Constraint-Projected Neural
  PDE Solver that Eliminates Hallucinations.* arXiv:2511.03578.
- Sun, Z., Inaba, S., Xiao, F. (2016). *Boundary Variation Diminishing
  (BVD) reconstruction: A new approach to improve Godunov schemes.*
  Journal of Computational Physics 322, 309-325. arXiv:1602.00814.
- Ross, S., Gordon, G., Bagnell, D. (2011). *A Reduction of Imitation
  Learning and Structured Prediction to No-Regret Online Learning.*
  AISTATS 2011.
- Not fully read, flagged above: *Learning second-order TVD flux limiters
  using differentiable solvers.* arXiv:2503.09625.
