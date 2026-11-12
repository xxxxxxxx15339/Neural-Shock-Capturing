# Artifact audit and method selection (frozen before matched evaluation)

Repository HEAD: `ffec5617a471ef68f91d21e353936936fd068532`, existing branch
`research/accuracy-upgrade-training-2026-11-12`. Existing uncommitted accuracy-upgrade
training files are preserved. This task does not train, modify checkpoints, or push.

The complete inventory is [CHECKPOINT_INVENTORY.csv](CHECKPOINT_INVENTORY.csv).
The initial scan covers 59 HDF5/pickle artifacts (including 12 dataset copies,
not 59 trained networks), 3,261 evidence/source files and 133,871 archived scalar
measurements. Exact paths, checksums, schemas, load status and evidence locations
are recorded. [archived_files.csv](results/raw/archived_files.csv) catalogs images,
PDFs, histories, numerical references, datasets and scripts; [archived_metrics.csv](results/raw/archived_metrics.csv)
retains JSON pointers and hashes rather than merging incompatible experiments.

| Display identity | Selected artifact | Architecture | Selection and inference |
|---|---|---|---|
| TENO5 | `src/differentiable/teno5_jax.py`, checked against NumPy implementation | deterministic 5-point, 3 candidates | epsilon=1e-40, C=1, q=6, cutoff=1e-5, d=(.1,.6,.3), center fallback |
| NENO6 | `vendor/NNENO/mytrainmodel1` | 6→64→32→16→8→4, 3,228 parameters | authentic author checkpoint, max(1,maxabs) normalization, discard probability≥.5; author candidate polynomials and optimized weights |
| NN-TENO5 | `models/nn_teno5.h5` | 5→64→32→16→8→3, 3,155 parameters | original Keras classifier; min-range normalization; archived calibrated discard thresholds (.66,.50,.69); hard masks |
| NN-TENO5-V2 (Learned Mixing Head), Euler | `models/nn_teno5euler_unrolled_jax_params_curriculum_phase21_richgate.pkl` | 3,155 backbone + 8→8→1 head (81 parameters) | richest finite, trained, validation-selected historical separate head; hard neural reconstruction blended with classical flux |
| Same display identity, scalar specialization | `models/nn_teno5_unrolled_jax_params_run6_learnedlambda.pkl` | 3,155 backbone + 3→1 head (4 parameters) | saved scalar learned head; not presented as the Euler rich-head architecture |

All three local HDF5 models load with Keras `compile=False`. Original classifier
weights are exported without training and checked against Keras predictions.
`SeemsGood3.h5` is a scalar-output regression model, not NN-TENO5. The oracle HDF5
classifier is a different training experiment. The earliest Euler JAX checkpoint
`models/nn_teno5euler_unrolled_jax_params.pkl` contains nonfinite weights and is
unusable; it is retained. Training-resume dictionaries are not inference schemas.

Phase18 and phase20 Euler checkpoints both contain a 3-probability linear head;
the word `v2` in phase20's filename does not indicate the rich architecture.
Phase21 contains the genuine 8→8→1 head. Its source script
`scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX_LearnedLambda_v3.py`
jointly trains backbone and head for 200 K=15 updates and chooses the best
short-Sod validation loss. Historical training/validation uses direct conservative
state repairs; those repairs are not used in the matched evaluation. Some source
docstrings incorrectly say 18 head inputs: checkpoint shapes and execution say 8.
Scalar run6 jointly trains on advection/Burgers and selects by canonical validation.

In Euler, the same saved head is applied separately to each characteristic field
and each positive/negative split-flux stencil: six coefficients per interface.
Blending occurs in those scalar characteristic reconstructions before projection
to the shared conservative face flux. A plotted interface mean is an average of
those six values, not one saved fluid-level coefficient. This distinction matters
for interpreting the scalar blending formula and any physical guarantees.

`models/v2/euler_gated_dataset_v2_seed2026.pkl` has **no separate head**. Its
configured lambda=.25 is fixed, as are the recent accuracy-upgrade models' blends.
Neither is substituted for a learned-head result. Their archived results remain
in the evidence catalog with their actual checkpoint identities.

## NENO6 provenance

The [paper](https://arxiv.org/abs/2207.08500), Appendix D page 37, links directly to
the [author repository](https://github.com/Lilian6/NNENO). We retrieved immutable
commit `b91427693fcba1496570477e25fcc55debf8e5cb`, preserved the Apache-2.0 license,
and loaded its state dictionary with `torch.load(weights_only=True)` and strict
author model shapes. The checkpoint SHA-256 is
`1a4a121c4b3b37fcf16eb60846b657d0c48229d848ceb0ce133ab0edee827f40`.
Author classes are extracted by AST; the GUI experiment is not executed on import.
The author reconstruction and an 8-step periodic advection smoke run are reference
fixtures for the JAX port. Source identities are in `vendor/NNENO/SOURCE.json`.
No guessed coefficients or synthesized weights are used. The author repository
provides scalar advection code; Euler and Burgers runs are explicitly ports of
that reconstruction into this repository's common solver, not reproductions of
the publication's entire Euler implementation. The all-discard zero denominator
is retained and reported as failure if encountered.

The paper's section4.1 specifies a **fourth-order optimized background scheme**;
the6 in NENO6 denotes stencil width, not a sixth-order accuracy guarantee.
The full nonlinear network's convergence is measured separately.

## Proposed versus executed features

`Documentation/Papers/proj.pdf` proposes normalized wider stencils, optimal weights,
smoothness ratios, neighboring jumps, Mach and pressure-gradient features.
`src/data/feature_vector.py` implements offline extraction, including an explicitly
approximate isolated-stencil jump. Genuine neighboring jumps have separate JAX
utilities. Neither selected classifier consumes that complete representation.
Phase21 consumes **three troubled probabilities + five normalized stencil values**;
run6 consumes only the three probabilities. No Mach/pressure descriptor is passed
to these saved heads. A six-value feature-extraction utility is not NENO6.

## Numerical and evidence contracts

The main Euler comparison retains historical characteristic split-flux reconstruction
of point-sampled conservative states. It is not the experimental finite-volume
state-flux correction. Treating these point samples as exact cell averages would
introduce the documented nonlinear consistency issue. Scalar runs retain the
repository's state-reconstruction/Lax–Friedrichs flux (alpha=1). Consequently the
Burgers implementation itself can limit smooth order; polynomial order is not a
full-solver accuracy result. Methods share SSPRK3, precision of numerical states,
grids, timesteps, references and boundary contracts; original Keras classifier
arithmetic remains float32 and is checked for inference parity.

Original classifier and learned-head backbones differ. Their headline difference
is a whole-method comparison. A separate same-backbone pure/fixed/learned ablation
is required to attribute benefit to the head. Fixed lambda is declared in advance
as sigmoid(-3), the historical initialization, not fitted to the new results.

Canonical problems were inspected and used in historical validation. This package
is replication and a frozen matched evaluation, not an untouched generalization
test. No hyperparameters are changed after evaluation. Archived numbers with
different CFL, grids, references, repair policies or checkpoints are never merged
into the new matched table. Earlier archived runtime measurements are contextual,
not cross-environment speed comparisons.
