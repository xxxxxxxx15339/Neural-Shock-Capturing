# -*- coding: utf-8 -*-
"""JAX-path three-way Euler comparison: TENO5euler(JAX) vs NN-TENO5euler(JAX)
vs the real solution.

The Euler counterpart of two existing scripts:
- `ValidationMatrix_ProjPDF.py` (this folder): the same three-way comparison,
  but scalar (advection/Burgers) rather than Euler.
- `TENO5_vs_NNTENO5_Euler.py` (`scripts/legacy_imitation_training/`): the same
  Euler benchmarks, but on the Keras/TF NN-TENO5euler classifier rather than
  the JAX unrolled-physics-loss network.

"Real solution", per problem:
- Sod: the exact Riemann solution (`src.analysis.euler_regression.exact_sod_solution`)
  -- the same reference `TENO5_vs_NNTENO5_Euler.py` uses.
- Shu-Osher: no closed-form exact solution exists (it's a shock/turbulence
  interaction). Per `TENO5_vs_NNTENO5_Euler.py`'s own convention, a converged
  high-resolution deterministic run stands in as the reference -- TENO5euler(JAX)
  at `--reference-cells` here, in place of that script's WENO7euler, because this
  project's JAX port only has TENO5euler (see `src/differentiable/euler_jax.py`).

NN-TENO5euler(JAX) status -- read before trusting a "NOT EVALUATED" row here as
a bug: per UNROLLED_PHYSICS_LOSS_GUIDE.md's Phase 6/7 status note, Euler
unrolled-physics-loss training hit a genuine, root-caused Roe-eigenbasis
conditioning instability that was never resolved. Both checkpoints this
script can load (`models/nn_teno5euler_unrolled_jax_params.pkl` and
`..._curriculum.pkl`) were verified (at the time this script was written) to
already contain non-finite weights -- the network itself is NaN-corrupted,
not just a rollout starting from it. This script checks that explicitly at
load time and reports NOT EVALUATED rather than passing a meaningless number
through as a real comparison (the same honesty rule `ValidationMatrix_ProjPDF.py`
follows for Euler). It still runs the TENO5euler-vs-exact/reference half of
the comparison regardless, and the loading/finiteness/positivity-guard
machinery here is written to work unmodified against a future, actually
-converged checkpoint (`--params-path`) -- this is not a one-off failure
report, it is the harness Phase 6/7's "next step" (a proper constrained
projection, see the guide) would be evaluated with.

Run inside the `teno-jax` conda environment:

    conda run -n teno-jax python -u scripts/unrolled_physics_loss/TENO5euler_vs_NNTENO5euler_JAX.py
"""

import os
import sys as _sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import argparse
import json
import time

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle

from src.analysis.euler_regression import (
    exact_sod_solution,
    interpolate_reference,
    primitive_variables,
    regression_metrics,
)
from src.core.eulerEquations import spds
from src.initial_conditions.InitialConditions import shuOsher, sod

from src.differentiable.euler_jax import euler_positivity_repair, get_euler_flux_jax
from src.differentiable.numerics_jax import rollout
from src.differentiable.projection_jax import mixed_reconstruct, project_keep_weight
from src.differentiable.teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from src.differentiable.teno5_jax_toy import toy_keep_weight, toy_lambda


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Three-way JAX-path Euler comparison: TENO5euler(JAX), '
            'NN-TENO5euler(JAX), and the real solution.'
        )
    )
    parser.add_argument(
        '--params-path',
        default=os.path.join('models', 'nn_teno5euler_unrolled_jax_params_curriculum.pkl'),
        help='Trained NN-TENO5euler(JAX) checkpoint (list of (W, b) pairs, as saved by '
             'Train_NNTENO5euler_UnrolledJAX*.py). Pass an empty string to skip the '
             'network entirely and only report TENO5euler vs. the real solution.',
    )
    parser.add_argument('--cells', type=int, default=120)
    parser.add_argument('--reference-cells', type=int, default=320)
    parser.add_argument('--cfl', type=float, default=0.4)
    parser.add_argument(
        '--wave-speed-safety',
        type=float,
        default=1.3,
        help='Safety margin on the initial-condition max characteristic speed used '
             'to pick a fixed dt (this JAX path uses a fixed dt per rollout, not the '
             'adaptive per-step CFL the NumPy eulerSimulation uses).',
    )
    parser.add_argument('--output-path', default=os.path.join('models', 'nn_teno5euler_jax_comparison.json'))
    parser.add_argument('--save-dir', default=os.path.join('models', 'nn_teno5euler_jax_comparison_plots'))
    parser.add_argument('--no-show', action='store_true')
    return parser.parse_args()


def load_network_params(params_path):
    """Loads either checkpoint format in this project: a flat `[(w, b), ...]`
    backbone-only list (every checkpoint before Phase 18), or Phase 18's
    `{'backbone': [...], 'lambda_head': (w, b)}` dict (the dedicated,
    separately-trained mixing-coefficient head -- see
    `Train_NNTENO5euler_UnrolledJAX_LearnedLambda.py`). Returns
    `(backbone_params, lambda_head_or_None, error_or_None)`.
    """
    if not params_path:
        return None, None, 'no --params-path given'
    full_path = params_path if os.path.isabs(params_path) else os.path.join(_REPO_ROOT, params_path)
    if not os.path.exists(full_path):
        return None, None, 'checkpoint not found at {}'.format(full_path)
    with open(full_path, 'rb') as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict):
        backbone = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded['backbone']]
        lambda_head = (jnp.asarray(loaded['lambda_head'][0]), jnp.asarray(loaded['lambda_head'][1]))
    else:
        backbone = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded]
        lambda_head = None

    all_finite = all(
        bool(np.isfinite(np.asarray(w)).all()) and bool(np.isfinite(np.asarray(b)).all())
        for w, b in backbone
    ) and (lambda_head is None or (
        bool(np.isfinite(np.asarray(lambda_head[0])).all()) and bool(np.isfinite(np.asarray(lambda_head[1])).all())
    ))
    if not all_finite:
        return None, None, (
            'checkpoint at {} contains non-finite weights -- see this script\'s module '
            'docstring and UNROLLED_PHYSICS_LOSS_GUIDE.md\'s Phase 6/7 status note'
        ).format(full_path)
    return backbone, lambda_head, None


def make_network_reconstruct(params, lambda_head=None):
    """Convex-mixed with TENO5's own reconstruction, not the network's raw
    candidate-combination alone. Two mixing-coefficient sources, matching
    whichever checkpoint format was loaded:
    - `lambda_head=None` (every checkpoint before Phase 18): Phase 16's
      zero-new-parameter `lambda = max_candidate(troubled_probability)`.
    - `lambda_head` given (Phase 18's dedicated head): a separately-trained
      `lambda`, decoupled from the candidate keep-weight signal -- see
      `teno5_jax_toy.toy_lambda`'s docstring for why that decoupling was
      needed (Phase 17's reuse-based training regressed three times running).
    Either way, this provably reduces to TENO5's exact reconstruction
    wherever `lambda` is 0, rather than relying on training alone to have
    learned that.
    """
    def reconstruct(stencils):
        troubled = 1.0 - toy_keep_weight(params, stencils)

        def network_reconstruct(s):
            return teno5_soft_reconstruct_jax(s, project_keep_weight(troubled, mode='hard'))

        lam = toy_lambda(lambda_head, troubled) if lambda_head is not None else None
        return mixed_reconstruct(troubled, stencils, network_reconstruct, teno5_reconstruct_jax, mode='hard', lam=lam)

    return reconstruct


def estimate_max_wave_speed(initial_state, safety):
    return float(np.max(np.abs(spds(initial_state))) * safety)


def euler_jax_rollout(reconstruct, initial_np, dt, dx, step_count, guard=None):
    flux_fn = get_euler_flux_jax(reconstruct, boundary='transmissive')
    final = rollout(jnp.asarray(initial_np), flux_fn, dt, dx, step_count, state_guard=guard)
    return np.asarray(final)


def state_checks(state):
    density, _, pressure = primitive_variables(state)
    return {
        'finite': bool(np.isfinite(state).all()),
        'minimum_density': float(np.min(density)),
        'minimum_pressure': float(np.min(pressure)),
        'positive_density': bool(np.all(density > 0.0)),
        'positive_pressure': bool(np.all(pressure > 0.0)),
    }


def run_nn_euler(reconstruct, initial_np, dt, dx, step_count):
    """Run the network-driven rollout; retry with a positivity-repair guard
    if the unguarded run goes non-finite. Reports which path was actually
    used -- see `euler_positivity_repair`'s docstring (src/differentiable/euler_jax.py)
    for why a guarded-but-finite result is still reported as such, not silently
    presented as equivalent to an unguarded run.
    """
    unguarded = euler_jax_rollout(reconstruct, initial_np, dt, dx, step_count)
    if np.isfinite(unguarded).all():
        return unguarded, {'guard_used': False, 'finite': True}

    guarded = euler_jax_rollout(
        reconstruct, initial_np, dt, dx, step_count, guard=euler_positivity_repair
    )
    finite = bool(np.isfinite(guarded).all())
    return (guarded if finite else None), {'guard_used': True, 'finite': finite}


def run_problem(
    name,
    initial_condition,
    length,
    final_time,
    cells,
    cfl,
    wave_speed_safety,
    network_params,
    network_status,
    lambda_head,
    reference_x,
    reference_state,
    reference_label,
):
    x = np.arange(cells, dtype=float) * length / cells
    initial_np = initial_condition(x)
    dx = length / cells
    max_wave_speed = estimate_max_wave_speed(initial_np, wave_speed_safety)
    step_count = max(1, int(np.ceil(final_time * max_wave_speed / (cfl * dx))))
    actual_dt = final_time / step_count

    reference = (
        reference_state if reference_x is None
        else interpolate_reference(reference_x, reference_state, x)
    )

    teno_started = time.perf_counter()
    teno_final = euler_jax_rollout(teno5_reconstruct_jax, initial_np, actual_dt, dx, step_count)
    teno_runtime = time.perf_counter() - teno_started
    teno_metrics = regression_metrics(x, teno_final, initial_np, final_time, reference)
    teno_metrics.update(state_checks(teno_final))
    teno_metrics['runtime_seconds'] = teno_runtime

    result = {
        'cells': cells,
        'final_time': final_time,
        'steps': step_count,
        'reference': reference_label,
        'teno5euler': teno_metrics,
        'nn_teno5euler': None,
        'beats_teacher': None,
    }
    plot_data = {'x': x, 'reference': reference, 'teno5euler': teno_final, 'nn_teno5euler': None}

    print('  TENO5euler(JAX)   : mean state L1 vs {} = {:.4e}'.format(
        reference_label,
        np.mean([teno_metrics['density_l1'], teno_metrics['momentum_l1'], teno_metrics['energy_l1']]),
    ))

    if network_params is None:
        result['nn_teno5euler_status'] = 'NOT EVALUATED: {}'.format(network_status)
        print('  NN-TENO5euler(JAX): NOT EVALUATED -- {}'.format(network_status))
        return result, plot_data

    reconstruct = make_network_reconstruct(network_params, lambda_head=lambda_head)
    nn_started = time.perf_counter()
    nn_final, nn_status = run_nn_euler(reconstruct, initial_np, actual_dt, dx, step_count)
    nn_runtime = time.perf_counter() - nn_started

    if nn_final is None:
        result['nn_teno5euler_status'] = (
            'NOT EVALUATED: rollout non-finite even under the positivity-repair guard'
        )
        print('  NN-TENO5euler(JAX): NOT EVALUATED -- rollout non-finite even under the guard')
        return result, plot_data

    nn_metrics = regression_metrics(x, nn_final, initial_np, final_time, reference)
    nn_metrics.update(state_checks(nn_final))
    nn_metrics['runtime_seconds'] = nn_runtime
    nn_metrics['positivity_guard_used'] = nn_status['guard_used']
    result['nn_teno5euler'] = nn_metrics
    plot_data['nn_teno5euler'] = nn_final

    teno_l1 = np.mean([teno_metrics['density_l1'], teno_metrics['momentum_l1'], teno_metrics['energy_l1']])
    nn_l1 = np.mean([nn_metrics['density_l1'], nn_metrics['momentum_l1'], nn_metrics['energy_l1']])
    beats = bool(nn_l1 <= teno_l1)
    result['beats_teacher'] = {
        'metric': 'mean(density_l1, momentum_l1, energy_l1) vs {}'.format(reference_label),
        'teno5euler': float(teno_l1),
        'nn_teno5euler': float(nn_l1),
        'nn_wins': beats,
    }
    guard_note = ' (positivity-repair guard was needed to keep this finite)' if nn_status['guard_used'] else ''
    print('  NN-TENO5euler(JAX): mean state L1 vs {} = {:.4e}{}'.format(reference_label, nn_l1, guard_note))
    print('  beats_teacher:', 'WIN' if beats else 'loss')
    return result, plot_data


def plot_problem(name, plot_data):
    x = plot_data['x']
    reference_density, reference_velocity, reference_pressure = primitive_variables(plot_data['reference'])
    teno_density, teno_velocity, teno_pressure = primitive_variables(plot_data['teno5euler'])
    nn_state = plot_data['nn_teno5euler']

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    panels = [
        (axes[0], 'Density', reference_density, teno_density),
        (axes[1], 'Velocity', reference_velocity, teno_velocity),
        (axes[2], 'Pressure', reference_pressure, teno_pressure),
    ]
    if nn_state is not None:
        nn_density, nn_velocity, nn_pressure = primitive_variables(nn_state)
        nn_values = (nn_density, nn_velocity, nn_pressure)
    else:
        nn_values = (None, None, None)

    for (axis, ylabel, reference_values, teno_values), nn_value in zip(panels, nn_values):
        axis.plot(x, reference_values, 'k:', linewidth=2, label='Reference')
        axis.plot(x, teno_values, label='TENO5euler(JAX)')
        if nn_value is not None:
            axis.plot(x, nn_value, '--', label='NN-TENO5euler(JAX)')
        axis.set_ylabel(ylabel)
    axes[0].legend()
    axes[-1].set_xlabel('x')
    figure.suptitle(name)
    return figure


def section(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def main():
    args = parse_args()

    network_params, lambda_head, network_status = load_network_params(args.params_path)
    if network_params is not None:
        print('Loaded NN-TENO5euler(JAX) checkpoint from {}'.format(args.params_path))
    else:
        print('NN-TENO5euler(JAX) will be NOT EVALUATED: {}'.format(network_status))

    results = {
        'params_path': args.params_path,
        'network_available': network_params is not None,
        'network_status': network_status,
        'cells': args.cells,
        'reference_cells': args.reference_cells,
        'cfl': args.cfl,
        'problems': {},
    }
    figures = {}

    section('Sod -- reference: exact Riemann solution')
    sod_x = np.arange(args.cells, dtype=float) / args.cells
    result, plot_data = run_problem(
        'sod', sod(), 1.0, 0.2, args.cells, args.cfl, args.wave_speed_safety,
        network_params, network_status, lambda_head,
        None, exact_sod_solution(sod_x, 0.2), 'exact Riemann solution',
    )
    results['problems']['sod'] = result
    figures['sod'] = plot_problem('Sod', plot_data)

    section('Shu-Osher -- reference: TENO5euler(JAX) at {} cells'.format(args.reference_cells))
    print('Building high-resolution reference...')
    ref_x = np.arange(args.reference_cells, dtype=float) * 10.0 / args.reference_cells
    ref_dx = 10.0 / args.reference_cells
    ref_initial = shuOsher()(ref_x)
    ref_max_wave_speed = estimate_max_wave_speed(ref_initial, args.wave_speed_safety)
    ref_step_count = max(1, int(np.ceil(1.8 * ref_max_wave_speed / (args.cfl * ref_dx))))
    ref_dt = 1.8 / ref_step_count
    ref_final = euler_jax_rollout(teno5_reconstruct_jax, ref_initial, ref_dt, ref_dx, ref_step_count)

    result, plot_data = run_problem(
        'shu_osher', shuOsher(), 10.0, 1.8, args.cells, args.cfl, args.wave_speed_safety,
        network_params, network_status, lambda_head,
        ref_x, ref_final, 'TENO5euler(JAX) at {} cells'.format(args.reference_cells),
    )
    results['problems']['shu_osher'] = result
    figures['shu_osher'] = plot_problem('Shu-Osher', plot_data)

    section('Summary')
    for name, result in results['problems'].items():
        beats = result['beats_teacher']
        if beats is None:
            print('{:12s}: NN-TENO5euler(JAX) {}'.format(name, result['nn_teno5euler_status']))
        else:
            print('{:12s}: {} vs {}'.format(
                name, 'NN-TENO5euler(JAX) BEATS TENO5euler(JAX)' if beats['nn_wins']
                else 'NN-TENO5euler(JAX) loses to TENO5euler(JAX)', result['reference'],
            ))

    if args.output_path:
        output_path = args.output_path if os.path.isabs(args.output_path) else os.path.join(_REPO_ROOT, args.output_path)
        with open(output_path, 'w', encoding='utf-8') as stream:
            json.dump(results, stream, indent=2)
        print('\nWrote metrics to {}'.format(output_path))

    if args.save_dir:
        save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(_REPO_ROOT, args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for name, figure in figures.items():
            figure_path = os.path.join(save_dir, '{}.png'.format(name))
            figure.savefig(figure_path, dpi=150, bbox_inches='tight')
            print('Saved {}'.format(figure_path))

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
