"""Diagnose NN-TENO5 spatial order and smoothness-gated corrections.

Run from any working directory:
    conda run -n teno-jax python -u \
        scripts/unrolled_physics_loss/check_NNTENO5_Order.py

Requires the existing local gated_teno5_jax.py implementation and
models/nn_teno5_unrolled_jax_params_run3_pretrained.pkl.
"""

from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import jax

# Set before importing jax.numpy or project modules that create JAX arrays.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src.analysis.convergence import (
    periodic_cell_averages,
    smooth_periodic_state,
)
from src.differentiable.gated_teno5_jax import gated_nn_teno5_reconstruct
from src.differentiable.numerics_jax import (
    lax_friedrichs_flux,
    periodic_stencils,
    rollout,
)
from src.differentiable.projection_jax import project_keep_weight
from src.differentiable.teno5_jax import (
    DEFAULT_CUTOFF,
    teno5_reconstruct_jax,
    teno5_selection_values,
    teno5_soft_reconstruct_jax,
)
from src.differentiable.teno5_jax_toy import toy_keep_weight

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "nn_teno5_unrolled_jax_params_run3_pretrained.pkl"
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained checkpoint not found: {MODEL_PATH}")

with MODEL_PATH.open("rb") as file:
    checkpoint = pickle.load(file)

params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in checkpoint]


def nn_reconstruct(stencils, mode="soft"):
    """Original neural-only reconstruction (no smooth-region gating)."""
    keep_soft = toy_keep_weight(params, stencils)
    keep = project_keep_weight(
        1.0 - keep_soft,
        mode=mode,
        threshold=0.5,
    )
    return teno5_soft_reconstruct_jax(stencils, keep)


def gated_reconstruct(stencils, mode="soft", strength=1.0):
    """Deterministic TENO5 in smooth regions, NN correction in troubled ones."""
    return gated_nn_teno5_reconstruct(
        params,
        stencils,
        mode=mode,
        strength=strength,
    )


def square_wave(x):
    z = np.mod(x, 1.0)
    return ((z >= 0.25) & (z < 0.75)).astype(float)


# ---------------------------------------------------------------------------
# 1. Semi-discrete spatial convergence (no SSPRK3 time-truncation effects)
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("SMOOTH PERIODIC ADVECTION: SEMI-DISCRETE SPATIAL CONVERGENCE")
print("=" * 78)

schemes = {
    "TENO5-JAX": teno5_reconstruct_jax,
    "NN-TENO5 hard": lambda s: nn_reconstruct(s, mode="hard"),
    "NN-TENO5 soft": lambda s: nn_reconstruct(s, mode="soft"),
    "All-keep control": lambda s: teno5_soft_reconstruct_jax(
        s,
        jnp.ones(s.shape[:-1] + (3,), dtype=s.dtype),
    ),
    "Gated NN hard": lambda s: gated_reconstruct(s, mode="hard"),
    "Gated NN soft": lambda s: gated_reconstruct(s, mode="soft"),
}

for name, reconstruct in schemes.items():
    previous_error = None
    print(f"\n{name}")
    print(f"{'N':>6} {'L1 error':>16} {'Order':>10}")

    for n in (40, 80, 160, 320, 640):
        dx = 1.0 / n
        _, cell_averages = periodic_cell_averages(n)

        flux_difference = lax_friedrichs_flux(
            jnp.asarray(cell_averages),
            lambda v: v,
            1.0,
            reconstruct,
        )
        numerical_derivative = np.asarray(flux_difference) / dx

        faces = np.arange(n + 1, dtype=float) * dx
        exact_derivative = (
            smooth_periodic_state(faces[1:])
            - smooth_periodic_state(faces[:-1])
        ) / dx

        error = np.mean(np.abs(numerical_derivative - exact_derivative))
        order = (
            float("nan")
            if previous_error is None
            else np.log2(previous_error / error)
        )
        print(f"{n:6d} {error:16.8e} {order:10.5f}")
        previous_error = error


# ---------------------------------------------------------------------------
# 2. Inspect neural candidate-selection behavior in smooth regions
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("NEURAL STENCIL SELECTION DIAGNOSTIC")
print("=" * 78)

linear_weights = np.array([0.1, 0.6, 0.3])

for n in (40, 80, 160, 320, 640):
    _, cell_averages = periodic_cell_averages(n)
    stencils = periodic_stencils(jnp.asarray(cell_averages))

    soft_keep = np.asarray(toy_keep_weight(params, stencils))
    hard_keep = soft_keep > 0.5

    chi = np.asarray(teno5_selection_values(stencils))
    teno_keep = chi >= DEFAULT_CUTOFF

    active_weights = soft_keep * linear_weights
    denominator = active_weights.sum(axis=-1, keepdims=True)
    effective_weights = np.divide(
        active_weights,
        denominator,
        out=np.zeros_like(active_weights),
        where=denominator > 0.0,
    )

    all_keep_fraction = np.mean(np.all(hard_keep, axis=-1))
    all_reject_fraction = np.mean(~np.any(hard_keep, axis=-1))
    teno_all_keep_fraction = np.mean(np.all(teno_keep, axis=-1))
    mean_weight_l1_gap = np.mean(
        np.sum(np.abs(effective_weights - linear_weights), axis=-1)
    )

    print(f"\nN = {n}")
    print(f"  TENO all-keep:       {teno_all_keep_fraction:.2%}")
    print(f"  NN hard all-keep:    {all_keep_fraction:.2%}")
    print(f"  NN hard all-reject:  {all_reject_fraction:.2%}")
    print(f"  Soft weight L1 gap:  {mean_weight_l1_gap:.6e}")
    print(f"  Mean soft keep:      {soft_keep.mean(axis=0)}")
    print(f"  Mean effective w:    {effective_weights.mean(axis=0)}")

    patterns, counts = np.unique(
        hard_keep.astype(int), axis=0, return_counts=True
    )
    print("  Hard mask frequencies:")
    for pattern, count in zip(patterns, counts):
        print(f"    {pattern.tolist()}: {count / n:.2%}")


# ---------------------------------------------------------------------------
# 3. Gate activity on the initial square-wave state
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("DISCONTINUITY GATE DIAGNOSTIC")
print("=" * 78)

for n in (80, 160, 320, 640):
    x = (np.arange(n) + 0.5) / n
    stencils = periodic_stencils(jnp.asarray(square_wave(x)))

    chi = np.asarray(teno5_selection_values(stencils))
    troubled = np.any(chi < DEFAULT_CUTOFF, axis=-1)

    baseline = np.asarray(teno5_reconstruct_jax(stencils))
    gated = np.asarray(gated_reconstruct(stencils, mode="hard", strength=1.0))
    correction = np.abs(gated - baseline)

    print(f"\nN = {n}")
    print(f"  Troubled stencils: {troubled.sum()} / {n}")
    print(f"  Gate activation:   {troubled.mean():.2%}")
    if np.any(troubled):
        print(
            "  Mean correction in troubled region: "
            f"{correction[troubled].mean():.6e}"
        )
    else:
        print("  No troubled stencils")
    print(f"  Maximum correction: {correction.max():.6e}")
    if np.any(~troubled):
        print(
            "  Correction outside gate: "
            f"{correction[~troubled].max():.6e}"
        )
    else:
        print("  Gate active everywhere")


# ---------------------------------------------------------------------------
# 4. Full discontinuous-advection rollout and correction-strength sweep
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("FULL DISCONTINUOUS ADVECTION BENCHMARK: CORRECTION-STRENGTH SWEEP")
print("=" * 78)

final_time = 0.2
cfl = 0.5

for n in (80, 160):
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    u0 = square_wave(x)
    exact = square_wave(x - final_time)

    step_count = int(np.ceil(final_time / (cfl * dx)))
    dt = final_time / step_count

    reconstructions = {
        "TENO5": teno5_reconstruct_jax,
        "Gated hard": lambda s: gated_reconstruct(
            s, mode="hard", strength=1.0
        ),
        "Gated soft": lambda s: gated_reconstruct(
            s, mode="soft", strength=1.0
        ),
    }

    # Bind each strength as a default argument to avoid late-bound closures.
    for strength in (0, 0.25, 0.75, 1.0):
        reconstructions[f"Gated soft lambda={strength}"] = (
            lambda s, lam=strength: gated_reconstruct(
                s, mode="soft", strength=lam
            )
        )

    results = {}
    errors = {}
    print(f"\nN = {n} | steps = {step_count} | dt = {dt:.6e}")

    for name, reconstruct in reconstructions.items():
        def flux_fn(u, reconstruct=reconstruct):
            return lax_friedrichs_flux(u, lambda v: v, 1.0, reconstruct)

        final = np.asarray(
            rollout(jnp.asarray(u0), flux_fn, dt, dx, step_count)
        )
        results[name] = final

        l1 = float(np.mean(np.abs(final - exact)))
        errors[name] = l1
        tv = float(np.sum(np.abs(final - np.roll(final, 1))))
        overshoot = max(0.0, float(final.max() - 1.0))
        undershoot = max(0.0, float(-final.min()))

        print(
            f"N={n:3d} | {name:23s} | "
            f"L1={l1:.7e} | TV={tv:.5f} | "
            f"over={overshoot:.3e} | under={undershoot:.3e}"
        )

    baseline = results["TENO5"]
    baseline_error = errors["TENO5"]

    print("\nChanges relative to deterministic TENO5:")
    for name in reconstructions:
        if name == "TENO5":
            continue
        max_difference = float(np.max(np.abs(results[name] - baseline)))
        improvement = 100.0 * (baseline_error - errors[name]) / baseline_error
        print(
            f"  {name:23s} | "
            f"L1 improvement={improvement:+7.2f}% | "
            f"max |u-u_TENO5|={max_difference:.7e}"
        )
    print("-" * 78)
