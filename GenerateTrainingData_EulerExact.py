# -*- coding: utf-8 -*-
"""Generate Euler training data with EXACT (not scheme-imitated) targets.

`GenerateTrainingData_Euler.py` trains the network to imitate WENO5-Z5's (or
WENO7's) own reconstruction from the same stencil -- a distillation
objective that provably caps the network at its teacher's quality (see
Report_Results.md, Part 4). This script instead targets the true physical
interface value from the exact 1D Euler Riemann solution
(`src.analysis.exact_riemann_solution`), for many random left/right states.
That removes the imitation ceiling at its root: the network has a genuine,
non-zero learning signal everywhere a classical scheme's own reconstruction
would differ from the true answer, not just on the <1% of stencils where
WENO5 and a marginally-better local formula happen to disagree.

Method, per random Riemann problem:
  1. Sample random physically valid left/right primitive states and a
     random advection time/window, so the shock/rarefaction/contact sits at
     a random sub-cell offset relative to the grid (varies "resolution").
  2. Build the 5-point INPUT stencil (characteristic, Lax-Friedrichs split)
     from CELL AVERAGES of the exact solution -- the same representation
     `NNEuler5`/`WENO5euler` operate on at runtime.
  3. Build the TARGET from the exact solution's point value AT the
     interface (not a cell average, not a scheme's reconstruction of it),
     passed through the identical Lax-Friedrichs split/characteristic
     projection used by the runtime scheme, so pos/neg targets are
     consistent with what `f_half_pos + f_half_neg` should reconstruct.

Output files match GenerateTrainingData_Euler.py's format so
Train_WENO5NN_Euler.py can consume either interchangeably:

    data/eulerAvgsExact.csv    shape (5, sample_count)
    data/eulerFluxExact.csv    shape (sample_count, 1)
    data/eulerGroupsExact.csv  shape (sample_count, 1)
"""

import argparse
import os

import numpy as np

from src.analysis import exact_riemann_solution
from src.core.SimulationClasses import eulerSimulation
from src.core.TimeSteppingMethods import SSPRK3
from src.core.eulerEquations import (
    GAMMA,
    flux,
    getEulerFlux,
    project_to_characteristic,
    roe_eigenbasis,
    spds,
)
from src.schemes import WENO7euler
from GenerateTrainingData_Euler import gather_stencils, make_shu_osher_initial_condition


CATEGORY_PROPORTIONS = {
    'riemann': 0.40,
    'shock_oscillation': 0.25,
    'smooth': 0.20,
    'constant': 0.05,
    'boundary': 0.10,
}


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate Euler characteristic training data with exact '
            'Riemann-solution targets instead of WENO5-Z5/WENO7 imitation.'
        )
    )
    parser.add_argument('--output-inputs', default='data/eulerAvgsExact.csv')
    parser.add_argument('--output-targets', default='data/eulerFluxExact.csv')
    parser.add_argument('--output-groups', default='data/eulerGroupsExact.csv')
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--cells', type=int, default=64,
                         help='Cells per generated Riemann/smooth problem.')
    parser.add_argument('--problems', type=int, default=1500,
                         help='Number of random Riemann problems to draw.')
    parser.add_argument('--smooth-waves', type=int, default=200)
    parser.add_argument('--constant-states', type=int, default=40)
    parser.add_argument('--quadrature-points', type=int, default=9,
                         help='Sub-samples per cell for cell-average quadrature.')
    parser.add_argument('--shu-variants', type=int, default=16,
                         help='Random Shu-Osher-like trajectories to run.')
    parser.add_argument('--shu-fine-cells', type=int, default=800,
                         help='Resolution of the fine reference trajectory.')
    parser.add_argument('--shu-snapshots', type=int, default=24,
                         help='Retained snapshots per Shu-Osher-like trajectory.')
    parser.add_argument('--radius', type=int, default=2,
                         help=(
                             'Stencil half-width. 2 gives 5-point input '
                             '(NNEuler5/WENO51stOrder); 3 gives 7-point '
                             '(NNEuler7/WENO71stOrder).'
                         ))
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def validate_args(args):
    if args.samples < 1:
        raise ValueError('--samples must be positive.')
    if args.cells < 8:
        raise ValueError('--cells must be at least 8.')
    if args.radius < 1:
        raise ValueError('--radius must be at least 1.')
    if args.problems < 1:
        raise ValueError('--problems must be positive.')
    if args.quadrature_points < 1:
        raise ValueError('--quadrature-points must be positive.')


# ---------------------------------------------------------------------------
# Exact-solution sampling
# ---------------------------------------------------------------------------

def conservative_state(density, velocity, pressure):
    energy = pressure/(GAMMA-1.0)+0.5*density*velocity**2
    return np.column_stack((density, density*velocity, energy))


def random_riemann_params(rng):
    """Sample a random, physically valid two-state Riemann problem.

    Ranges are chosen to comfortably cover the Sod/Shu-Osher test regimes
    (density ~0.1-5, pressure ~0.1-12) with headroom, while avoiding the
    rare extreme-contrast combinations (e.g. high density + high pressure
    + high velocity simultaneously) that produce enormous characteristic
    flux magnitudes and destabilize MSE training via outlier domination.
    """
    rho_l = rng.uniform(0.15, 4.0)
    rho_r = rng.uniform(0.15, 4.0)
    vel_l = rng.uniform(-1.5, 1.5)
    vel_r = rng.uniform(-1.5, 1.5)
    p_l = rng.uniform(0.1, 11.0)
    p_r = rng.uniform(0.1, 11.0)
    return rho_l, vel_l, p_l, rho_r, vel_r, p_r


def cell_average_exact(params, cell_centers, dx, time, discontinuity,
                        quadrature_points):
    """Cell-average conservative state from the exact Riemann solution."""
    offsets = np.linspace(-0.5*dx, 0.5*dx, quadrature_points, endpoint=True)
    x_sub = cell_centers[:, None]+offsets[None, :]
    flat = exact_riemann_solution(x_sub.reshape(-1), time, *params,
                                   discontinuity=discontinuity)
    return flat.reshape(cell_centers.shape[0], quadrature_points, 3).mean(axis=1)


def interface_exact(params, interface_x, time, discontinuity):
    """Point value of the exact Riemann solution AT each interface."""
    return exact_riemann_solution(interface_x, time, *params,
                                   discontinuity=discontinuity)


# ---------------------------------------------------------------------------
# Shock/entropy-wave interaction (Shu-Osher-like): fine-reference targets
#
# No closed-form exact solution exists once a shock interacts with a smooth
# entropy wave. The previous approach (GenerateTrainingData_Euler.py) used
# WENO7 on a locally wider 7-point stencil as the "higher fidelity" target
# -- but WENO5 and WENO7 are both local formulas and agree on >99% of
# stencils in this region (Report_Results.md, Part 4), so that taught the
# network almost nothing new. Here the reference instead comes from a WENO7
# trajectory run at far higher spatial resolution, interpolated in space
# and time onto the training grid -- genuinely new sub-grid information a
# local formula cannot see, not just a wider local formula at the same
# resolution.
# ---------------------------------------------------------------------------

def run_trajectory_with_times(initial_condition, cells, length, final_time, cfl):
    """Run a WENO7 Euler trajectory, keeping every accepted step and its time."""
    simulation = eulerSimulation(
        cells, 2, length, final_time, SSPRK3(), getEulerFlux(WENO7euler()),
        initial_condition, 3, max_cfl=cfl,
        wave_speed_function=lambda state: np.max(np.abs(spds(state))),
    )
    history = np.transpose(simulation.runEuler(), (1, 0, 2))  # (nt, cells, 3)
    return simulation.last_times, history


def interpolate_fine_state(fine_times, fine_history, fine_x, query_time, query_x):
    """Fine trajectory's state at (query_time, query_x) via time+space interpolation."""
    index = np.clip(np.searchsorted(fine_times, query_time), 1, len(fine_times)-1)
    t0, t1 = fine_times[index-1], fine_times[index]
    weight = 0.0 if t1 <= t0 else (query_time-t0)/(t1-t0)
    state_at_time = (1.0-weight)*fine_history[index-1]+weight*fine_history[index]
    return np.column_stack([
        np.interp(query_x, fine_x, state_at_time[:, component])
        for component in range(3)
    ])


def random_shu_osher_initial_condition(rng):
    shock_position = rng.uniform(0.6, 1.4)
    amplitude = rng.uniform(0.1, 0.3)
    wave_number = rng.uniform(3.5, 6.5)
    phase = rng.uniform(0, 2*np.pi)
    return make_shu_osher_initial_condition(
        shock_position=shock_position, amplitude=amplitude,
        wave_number=wave_number, phase=phase,
    )


def shock_oscillation_states(args, rng):
    """Yield (group_id, coarse_state, fine_interpolated_state) per snapshot."""
    length = 10.0
    final_time = 1.8
    cfl = 0.3
    coarse_x = (np.arange(args.cells, dtype=float)+0.5)*length/args.cells
    coarse_interfaces = (np.arange(1, args.cells+1, dtype=float))*length/args.cells
    fine_x = (np.arange(args.shu_fine_cells, dtype=float)+0.5)*length/args.shu_fine_cells

    for variant in range(args.shu_variants):
        initial_condition = random_shu_osher_initial_condition(rng)

        coarse_times, coarse_history = run_trajectory_with_times(
            initial_condition, args.cells, length, final_time, cfl,
        )
        fine_times, fine_history = run_trajectory_with_times(
            initial_condition, args.shu_fine_cells, length, final_time, cfl,
        )

        retained = min(args.shu_snapshots, len(coarse_times))
        snapshot_indices = np.unique(np.linspace(
            1, len(coarse_times)-1, retained,
        ).astype(int))

        for snapshot in snapshot_indices:
            coarse_state = coarse_history[snapshot]
            query_time = coarse_times[snapshot]
            fine_at_interfaces = interpolate_fine_state(
                fine_times, fine_history, fine_x, query_time, coarse_interfaces,
            )
            if not (np.isfinite(coarse_state).all()
                    and np.isfinite(fine_at_interfaces).all()):
                continue
            yield (
                'shu_{:03d}_t{:03d}'.format(variant, snapshot),
                coarse_state, fine_at_interfaces,
            )


# ---------------------------------------------------------------------------
# Characteristic split-flux construction (input: stencil; target: exact point)
# ---------------------------------------------------------------------------

def build_samples(state, exact_interface_state, cell_count, radius=2):
    """Return (inputs, targets, is_boundary) for one problem's interfaces.

    `state`: (cells, 3) cell averages, used for both the input stencils and
    the local Roe basis/wave speed (exactly what the runtime scheme sees).
    `exact_interface_state`: (cells, 3) exact point value at each cell's
    right interface, at the same instant -- used only for the target.
    `radius`: stencil half-width; radius=2 gives the original 5-point
    input, radius=3 gives 7-point (matching NNEuler7/WENO71stOrder).
    """
    physical_flux = flux(state)
    speeds = np.max(np.abs(spds(state)), axis=1)
    right_speed = np.concatenate((speeds[1:], speeds[-1:]))
    alpha = np.maximum(speeds, right_speed)

    left_basis, _ = roe_eigenbasis(state, boundary='transmissive')

    positive_state_stencil = gather_stencils(state, offset=0, radius=radius, reverse=False)
    positive_flux_stencil = gather_stencils(physical_flux, offset=0, radius=radius, reverse=False)
    negative_state_stencil = gather_stencils(state, offset=1, radius=radius, reverse=True)
    negative_flux_stencil = gather_stencils(physical_flux, offset=1, radius=radius, reverse=True)

    positive_characteristic_state = project_to_characteristic(positive_state_stencil, left_basis)
    positive_characteristic_flux = project_to_characteristic(positive_flux_stencil, left_basis)
    negative_characteristic_state = project_to_characteristic(negative_state_stencil, left_basis)
    negative_characteristic_flux = project_to_characteristic(negative_flux_stencil, left_basis)

    alpha_col = alpha[:, None, None]
    positive_split_flux = 0.5*(positive_characteristic_flux+alpha_col*positive_characteristic_state)
    negative_split_flux = 0.5*(negative_characteristic_flux-alpha_col*negative_characteristic_state)

    # Exact target: same split-flux decomposition, evaluated at the true
    # interface point instead of reconstructed from the stencil.
    exact_flux = flux(exact_interface_state)
    exact_char_state = project_to_characteristic(
        exact_interface_state[:, :, None], left_basis,
    )[:, :, 0]
    exact_char_flux = project_to_characteristic(
        exact_flux[:, :, None], left_basis,
    )[:, :, 0]
    positive_target = 0.5*(exact_char_flux+alpha[:, None]*exact_char_state)
    negative_target = 0.5*(exact_char_flux-alpha[:, None]*exact_char_state)

    interface_indices = np.arange(cell_count)
    interior = (
        (interface_indices >= radius)
        & (interface_indices <= cell_count-2-radius)
    )
    field_count = positive_split_flux.shape[1]
    is_boundary_per_branch = np.broadcast_to(
        (~interior)[:, None], (cell_count, field_count),
    ).reshape(-1)

    inputs = np.concatenate((positive_split_flux, negative_split_flux), axis=0)
    targets = np.concatenate((positive_target, negative_target), axis=0)
    is_boundary = np.concatenate((is_boundary_per_branch, is_boundary_per_branch))

    stencil_width = 2*radius+1
    inputs = inputs.reshape(-1, stencil_width)
    targets = targets.reshape(-1)

    # A few rare, extreme-contrast Riemann configurations produce
    # characteristic flux magnitudes an order of magnitude beyond the rest
    # of the dataset -- these dominate MSE training via outlier gradients
    # rather than teaching genuine shock behavior, so drop them here.
    magnitude_bound = 30.0
    valid = (
        np.isfinite(inputs).all(axis=1) & np.isfinite(targets)
        & (np.abs(inputs).max(axis=1) <= magnitude_bound)
        & (np.abs(targets) <= magnitude_bound)
    )
    return inputs[valid], targets[valid], is_boundary[valid]


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def generated_states(args, rng):
    """Yield (category, group_id, state, exact_interface_state, cell_count)."""

    length = 1.0
    x = (np.arange(args.cells, dtype=float)+0.5)*length/args.cells  # cell centers
    interfaces = np.arange(1, args.cells+1, dtype=float)*length/args.cells  # right edges
    dx = length/args.cells

    # --- random Riemann problems ---
    for index in range(args.problems):
        params = random_riemann_params(rng)
        discontinuity = rng.uniform(0.3, 0.7)
        # Random time: small enough that waves stay inside [0,1], large
        # enough that the shock/rarefaction has separated from the origin.
        time = rng.uniform(0.02, 0.15)

        state = cell_average_exact(params, x, dx, time, discontinuity,
                                    args.quadrature_points)
        exact_at_interface = interface_exact(params, interfaces, time,
                                              discontinuity)
        if not (np.isfinite(state).all() and np.isfinite(exact_at_interface).all()):
            continue
        if np.any(state[:, 0] <= 0):
            continue

        yield 'riemann', 'riemann_{:04d}'.format(index), state, exact_at_interface

    # --- smooth waves: exact solution IS the state, no shock to resolve ---
    frequencies = (0.8, 1.5, 3.0, 5.0, 8.0)
    for index in range(args.smooth_waves):
        frequency = rng.choice(frequencies)
        phase = rng.uniform(0, 2*np.pi)
        amplitude = rng.uniform(0.05, 0.3)

        argument_centers = 2*np.pi*frequency*x+phase
        density = 1.0+amplitude*np.sin(argument_centers)
        velocity = 0.15*np.cos(argument_centers)
        pressure = 1.0+0.1*amplitude*np.sin(argument_centers)
        state = conservative_state(density, velocity, pressure)

        argument_interfaces = 2*np.pi*frequency*interfaces+phase
        density_i = 1.0+amplitude*np.sin(argument_interfaces)
        velocity_i = 0.15*np.cos(argument_interfaces)
        pressure_i = 1.0+0.1*amplitude*np.sin(argument_interfaces)
        exact_at_interface = conservative_state(density_i, velocity_i, pressure_i)

        yield 'smooth', 'smooth_{:04d}'.format(index), state, exact_at_interface

    # --- constant states: trivial but necessary for the const_n fallback path ---
    for index in range(args.constant_states):
        density = rng.uniform(0.3, 4.0)
        velocity = rng.uniform(-1.5, 1.5)
        pressure = rng.uniform(0.2, 10.0)
        state = conservative_state(
            np.full(args.cells, density), np.full(args.cells, velocity),
            np.full(args.cells, pressure),
        )
        exact_at_interface = conservative_state(
            np.full(args.cells, density), np.full(args.cells, velocity),
            np.full(args.cells, pressure),
        )
        yield 'constant', 'constant_{:04d}'.format(index), state, exact_at_interface

    # --- Shu-Osher-like shock/entropy-wave interactions ---
    for group_id, coarse_state, fine_at_interface in shock_oscillation_states(args, rng):
        yield 'shock_oscillation', group_id, coarse_state, fine_at_interface


# ---------------------------------------------------------------------------
# Dataset balancing (same scheme as GenerateTrainingData_Euler.py)
# ---------------------------------------------------------------------------

def requested_category_counts(sample_count):
    categories = list(CATEGORY_PROPORTIONS)
    counts = {}
    allocated = 0
    for category in categories[:-1]:
        count = int(sample_count*CATEGORY_PROPORTIONS[category])
        counts[category] = count
        allocated += count
    counts[categories[-1]] = sample_count-allocated
    return counts


def sample_balanced_dataset(category_data, sample_count, rng):
    requested = requested_category_counts(sample_count)
    selected_inputs, selected_targets, selected_groups = [], [], []
    remaining_inputs, remaining_targets, remaining_groups = [], [], []

    for category in CATEGORY_PROPORTIONS:
        inputs, targets, groups = category_data[category]
        permutation = rng.permutation(len(inputs))
        take_count = min(requested[category], len(inputs))
        selected = permutation[:take_count]
        unused = permutation[take_count:]

        selected_inputs.append(inputs[selected])
        selected_targets.append(targets[selected])
        selected_groups.append(groups[selected])

        if len(unused):
            remaining_inputs.append(inputs[unused])
            remaining_targets.append(targets[unused])
            remaining_groups.append(groups[unused])

    current_count = sum(len(v) for v in selected_inputs)
    missing_count = sample_count-current_count
    if missing_count > 0:
        if not remaining_inputs:
            raise ValueError('No unused samples remain to satisfy the requested count.')
        extra_inputs = np.concatenate(remaining_inputs, axis=0)
        extra_targets = np.concatenate(remaining_targets, axis=0)
        extra_groups = np.concatenate(remaining_groups, axis=0)
        if len(extra_inputs) < missing_count:
            raise ValueError(
                'Requested {} samples, but only {} available. Reduce '
                '--samples or increase --problems/--smooth-waves.'.format(
                    sample_count, current_count+len(extra_inputs)))
        extra_selection = rng.choice(len(extra_inputs), size=missing_count, replace=False)
        selected_inputs.append(extra_inputs[extra_selection])
        selected_targets.append(extra_targets[extra_selection])
        selected_groups.append(extra_groups[extra_selection])

    inputs = np.concatenate(selected_inputs, axis=0)
    targets = np.concatenate(selected_targets, axis=0)
    groups = np.concatenate(selected_groups, axis=0)
    order = rng.permutation(len(inputs))
    return inputs[order], targets[order], groups[order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ensure_parent_directory(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    validate_args(args)
    rng = np.random.RandomState(args.seed)

    collected = {category: [] for category in CATEGORY_PROPORTIONS}
    problem_count = 0

    print('Generating exact-Riemann Euler training samples...', flush=True)

    for category, group_id, state, exact_at_interface in generated_states(args, rng):
        inputs, targets, is_boundary = build_samples(
            state, exact_at_interface, args.cells, radius=args.radius,
        )
        groups = np.full(len(inputs), group_id, dtype=object)

        collected[category].append((inputs[~is_boundary], targets[~is_boundary], groups[~is_boundary]))
        collected['boundary'].append((inputs[is_boundary], targets[is_boundary], groups[is_boundary]))
        problem_count += 1

    print('Available samples before balancing:', flush=True)
    category_data = {}
    for category in CATEGORY_PROPORTIONS:
        if not collected[category]:
            raise ValueError('No samples were generated for category {}.'.format(category))
        parts = collected[category]
        category_data[category] = (
            np.concatenate([p[0] for p in parts], axis=0),
            np.concatenate([p[1] for p in parts], axis=0),
            np.concatenate([p[2] for p in parts], axis=0),
        )
        print('  {:20s}: {}'.format(category, len(category_data[category][0])), flush=True)

    inputs, targets, groups = sample_balanced_dataset(category_data, args.samples, rng)

    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError('Final dataset contains NaN or infinity.')

    for path in (args.output_inputs, args.output_targets, args.output_groups):
        ensure_parent_directory(path)

    np.savetxt(args.output_inputs, inputs.T, delimiter=',')
    np.savetxt(args.output_targets, targets[:, None], delimiter=',')
    np.savetxt(args.output_groups, groups[:, None], delimiter=',', fmt='%s')

    print('\nGenerated dataset successfully.', flush=True)
    print('  Problems processed     : {}'.format(problem_count))
    print('  Unique trajectory IDs  : {}'.format(len(np.unique(groups))))
    print('  Total samples          : {}'.format(len(inputs)))
    print('  Input file             : {}'.format(args.output_inputs))
    print('  Target file            : {}'.format(args.output_targets))
    print('  Group file             : {}'.format(args.output_groups))


if __name__ == '__main__':
    main()
