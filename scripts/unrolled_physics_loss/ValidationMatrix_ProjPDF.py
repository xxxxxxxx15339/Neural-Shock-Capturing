# -*- coding: utf-8 -*-
"""Import-safe historical scalar evaluator. No model loads or evaluations occur at import.

For final four-problem reproduction use Evaluate_Benchmarks.py.

Phase 7: the Unrolled Physics Loss proposal's Table 1 validation matrix, using Phase 6's trained network.

Maps the Unrolled Physics Loss proposal's benchmark table onto the JAX infrastructure built in Phases
0-6, computing both the Unrolled Physics Loss proposal's own requested metric AND this project's own
`beats_teacher` real-solution criterion (Results.md Phase 6) for every row
-- per UNROLLED_PHYSICS_LOSS_GUIDE.md Phase 7's own instruction not to drop
that check just because the Unrolled Physics Loss proposal's table doesn't ask for it.

Coverage, stated plainly:
- Advection Lineaire, Burgers 1D: use `run3_pretrained`'s trained network
  (Phase 6). Burgers is a genuine out-of-distribution test -- the network
  was trained only on linear advection dynamics, never on Burgers' 0.5*u^2
  flux.
- Shu-Osher 1D, Sod (Euler): NN-TENO5euler is NOT evaluated here.
  UNROLLED_PHYSICS_LOSS_GUIDE.md's Phase 6 status note documents why in full:
  Euler training hit a genuine, root-caused-but-unsolved numerical
  instability (Roe-eigenbasis conditioning issue), not a missing result --
  reporting a number from that NaN-corrupted checkpoint would misrepresent
  it as a real comparison. These rows report the deterministic TENO5euler
  baseline only.

Burgers' exact reference (`burgers_characteristic_reference`) is copied
from `TENO5_vs_NNTENO5_Scalar.py` rather than imported from it: that module
does `from keras.models import load_model` at the top, which would pull
Keras into this script's `teno-jax` environment -- see
`src/differentiable/__init__.py` for why that coupling is avoided
throughout this whole differentiable-JAX side of the project.

Run inside `teno-jax`:

    conda run -n teno-jax python -u scripts/unrolled_physics_loss/ValidationMatrix_ProjPDF.py
"""

import os
import sys as _sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)


import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np

from src.core.Equations import adv, invBurg
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation
from src.core.TimeSteppingMethods import SSPRK3
from src.schemes import TENO5

from src.differentiable.losses_jax import total_variation
from src.differentiable.numerics_jax import lax_friedrichs_flux, rollout
from src.differentiable.projection_jax import mixed_reconstruct, project_keep_weight
from src.differentiable.teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from src.differentiable.teno5_jax_toy import toy_keep_weight

CELLS = 64
X = np.linspace(0.0, 1.0, CELLS, endpoint=False)


def burgers_initial_profile(x):
    return 0.5 + 0.25 * np.sin(2.0 * np.pi * x)


def burgers_characteristic_reference(x, final_time):
    """Exact smooth Burgers solution before shock formation (t < 2/pi).
    Verbatim copy of TENO5_vs_NNTENO5_Scalar.py's function of the same
    name -- see module docstring for why this is copied, not imported.
    """
    shock_time = 2.0 / np.pi
    if final_time >= shock_time:
        raise ValueError('final_time must be < {:.8f}.'.format(shock_time))

    mapped_start = final_time * burgers_initial_profile(np.array([0.0]))[0]
    target = np.mod(x, 1.0)
    target = np.where(target < mapped_start, target + 1.0, target)

    lower = np.zeros_like(target)
    upper = np.ones_like(target)
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        mapped_midpoint = midpoint + final_time * burgers_initial_profile(midpoint)
        belongs_right = mapped_midpoint < target
        lower = np.where(belongs_right, midpoint, lower)
        upper = np.where(belongs_right, upper, midpoint)

    foot = 0.5 * (lower + upper)
    return burgers_initial_profile(foot)


from src.differentiable.model_io import load_checkpoint
from src.differentiable.reconstruction import make_reconstruction

_NETWORK_RECONSTRUCT = None


def _network_reconstruct(stencils):
    """Lazily load the historical run3/max-troubled blend, without side effects at import."""
    global _NETWORK_RECONSTRUCT
    if _NETWORK_RECONSTRUCT is None:
        checkpoint = load_checkpoint('models/nn_teno5_unrolled_jax_params_run3_pretrained.pkl')
        _NETWORK_RECONSTRUCT = make_reconstruction(checkpoint, 'max-blend')
    return _NETWORK_RECONSTRUCT(stencils)


def run_scalar(physical_flux, wave_speed, u0_np, reconstruct, final_time, cfl=0.5):
    dx = 1.0 / CELLS
    step_count = int(np.ceil(final_time * wave_speed / (cfl * dx)))
    dt = final_time / step_count
    flux_fn = lambda u: lax_friedrichs_flux(u, physical_flux, wave_speed, reconstruct)
    return rollout(jnp.asarray(u0_np), flux_fn, dt, dx, step_count)


def l1_l2(state, exact):
    diff = np.asarray(state) - np.asarray(exact)
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))


def step_profile(x):
    return ((np.mod(x, 1.0) >= 0.25) & (np.mod(x, 1.0) < 0.75)).astype(float)


def burgers_flux(u):
    return 0.5 * u ** 2


def main():
    results = []

    print('=' * 78)
    print('Row 1: Advection Lineaire -- L1/L2 error, TV; beats_teacher')
    print('=' * 78)




    final_time = 1.0
    u0 = step_profile(X)
    exact = step_profile(X - final_time)  # wave_speed = 1, periodic

    nn_final = np.asarray(run_scalar(lambda u: u, 1.0, u0, _network_reconstruct, final_time))
    teno_final = np.asarray(run_scalar(lambda u: u, 1.0, u0, teno5_reconstruct_jax, final_time))

    nn_l1, nn_l2 = l1_l2(nn_final, exact)
    teno_l1, teno_l2 = l1_l2(teno_final, exact)
    nn_tv = float(total_variation(jnp.asarray(nn_final)))
    teno_tv = float(total_variation(jnp.asarray(teno_final)))
    beats = nn_l2 <= teno_l2

    print('NN-TENO5(JAX): L1={:.4e} L2={:.4e} TV={:.4f}'.format(nn_l1, nn_l2, nn_tv))
    print('TENO5        : L1={:.4e} L2={:.4e} TV={:.4f}'.format(teno_l1, teno_l2, teno_tv))
    print('beats_teacher (L2):', 'WIN' if beats else 'loss')
    results.append(('Advection', 'L2 error', nn_l2, teno_l2, beats))


    print('\n' + '=' * 78)
    print('Row 2: Burgers 1D -- error vs exact (pre-shock), shock-speed proxy; beats_teacher')
    print('=' * 78)

    burgers_time = 0.5  # < shock_time = 2/pi ~ 0.6366
    u0_burgers = burgers_initial_profile(X)
    exact_burgers = burgers_characteristic_reference(X, burgers_time)




    nn_final_b = np.asarray(run_scalar(burgers_flux, 1.0, u0_burgers, _network_reconstruct, burgers_time))
    teno_final_b = np.asarray(run_scalar(burgers_flux, 1.0, u0_burgers, teno5_reconstruct_jax, burgers_time))

    nn_l1_b, nn_l2_b = l1_l2(nn_final_b, exact_burgers)
    teno_l1_b, teno_l2_b = l1_l2(teno_final_b, exact_burgers)
    beats_b = nn_l2_b <= teno_l2_b

    print('(out-of-distribution: network trained only on linear advection dynamics)')
    print('NN-TENO5(JAX): L1={:.4e} L2={:.4e}'.format(nn_l1_b, nn_l2_b))
    print('TENO5        : L1={:.4e} L2={:.4e}'.format(teno_l1_b, teno_l2_b))
    print('beats_teacher (L2):', 'WIN' if beats_b else 'loss')
    results.append(('Burgers', 'L2 error', nn_l2_b, teno_l2_b, beats_b))


    print('\n' + '=' * 78)
    print('Rows 3-4: Shu-Osher 1D, Sod (Euler) -- NOT evaluated for NN-TENO5euler')
    print('=' * 78)
    print('This historical scalar matrix does not evaluate Euler. Use Evaluate_Benchmarks.py')
    print('for the frozen Phase 18 Euler model and reference-convergence diagnostics.')
    results.append(('Shu-Osher', 'spectral peak amplitude', None, None, None))
    results.append(('Sod', 'positivity of P, rho', None, None, None))


    print('\n' + '=' * 78)
    print('Summary')
    print('=' * 78)
    for name, metric, nn_val, teno_val, beats in results:
        if beats is None:
            print('{:12s}: NOT EVALUATED for NN-TENO5 (see above)'.format(name))
        else:
            print('{:12s}: {} -- {}'.format(name, metric, 'BEATS TENO5' if beats else 'loses to TENO5'))


if __name__ == '__main__':
    main()
