# -*- coding: utf-8 -*-
"""Generate characteristic Euler training data for WENO5-NN.

The network input is a five-point Lax--Friedrichs split characteristic-flux
stencil. The target is a WENO-Z5 reconstruction calculated from exactly the
same five-point stencil.

WENO7 is used only to generate stable Euler trajectories. It is not used as
the reconstruction target.

The generated files are:

    data/eulerAvgs.csv
        Shape: (5, sample_count)

    data/eulerFlux.csv
        Shape: (sample_count, 1)

    data/eulerGroups.csv
        Shape: (sample_count, 1)

The group file contains trajectory identifiers so that training,
validation and testing can be separated by trajectory instead of by
correlated individual stencil rows.
"""

import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import os

import numpy as np

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


CATEGORY_PROPORTIONS = {
    'smooth': 0.35,
    'shock': 0.25,
    'shock_oscillation': 0.20,
    'constant': 0.10,
    'boundary': 0.10,
}


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate five-point characteristic Euler data with '
            'WENO-Z5 reconstruction targets.'
        )
    )

    parser.add_argument(
        '--output-inputs',
        default='data/eulerAvgs.csv',
    )

    parser.add_argument(
        '--output-targets',
        default='data/eulerFlux.csv',
    )

    parser.add_argument(
        '--output-groups',
        default='data/eulerGroups.csv',
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=100000,
        help='Total requested number of generated samples.',
    )

    parser.add_argument(
        '--cells',
        type=int,
        default=256,
        help='Cells used for generated Euler trajectories.',
    )

    parser.add_argument(
        '--cfl',
        type=float,
        default=0.25,
    )

    parser.add_argument(
        '--snapshots',
        type=int,
        default=24,
        help='Maximum number of snapshots retained from each trajectory.',
    )

    parser.add_argument(
        '--shock-variants',
        type=int,
        default=6,
        help='Number of Sod-like shock-tube trajectories.',
    )

    parser.add_argument(
        '--shu-variants',
        type=int,
        default=6,
        help='Number of Shu--Osher-like trajectories.',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    return parser.parse_args()


def validate_args(args):
    if args.samples < 1:
        raise ValueError('--samples must be positive.')

    if args.cells < 8:
        raise ValueError('--cells must be at least 8.')

    if args.cfl <= 0.0:
        raise ValueError('--cfl must be positive.')

    if args.snapshots < 2:
        raise ValueError('--snapshots must be at least 2.')

    if args.shock_variants < 1:
        raise ValueError('--shock-variants must be positive.')

    if args.shu_variants < 1:
        raise ValueError('--shu-variants must be positive.')


# ---------------------------------------------------------------------------
# Euler initial conditions
# ---------------------------------------------------------------------------

def conservative_state(density, velocity, pressure):
    """Convert primitive variables into Euler conservative variables."""

    density = np.asarray(density, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    pressure = np.asarray(pressure, dtype=float)

    energy = (
        pressure / (GAMMA - 1.0)
        + 0.5 * density * velocity**2
    )

    return np.column_stack((
        density,
        density * velocity,
        energy,
    ))


def smooth_wave(x, frequency, phase, amplitude):
    """Construct a smooth, physically valid Euler state."""

    argument = frequency * (x - 5.0) + phase

    density = 1.0 + amplitude * np.sin(argument)
    velocity = 0.15 * np.cos(argument)
    pressure = 1.0 + 0.10 * amplitude * np.sin(argument)

    return conservative_state(
        density,
        velocity,
        pressure,
    )


def make_riemann_initial_condition(
    discontinuity,
    left_density=1.0,
    left_velocity=0.0,
    left_pressure=1.0,
    right_density=0.125,
    right_velocity=0.0,
    right_pressure=0.1,
):
    """Create a Sod-like Riemann initial condition."""

    def initial_condition(x):
        left = x < discontinuity

        density = np.where(
            left,
            left_density,
            right_density,
        )

        velocity = np.where(
            left,
            left_velocity,
            right_velocity,
        )

        pressure = np.where(
            left,
            left_pressure,
            right_pressure,
        )

        return conservative_state(
            density,
            velocity,
            pressure,
        )

    return initial_condition


def make_shu_osher_initial_condition(
    shock_position=1.0,
    amplitude=0.2,
    wave_number=5.0,
    phase=0.0,
):
    """Create a translated Shu--Osher-like initial condition on [0, 10]."""

    def initial_condition(x):
        left = x < shock_position

        right_density = (
            1.0
            + amplitude * np.sin(
                wave_number * (x - 5.0) + phase
            )
        )

        density = np.where(
            left,
            3.857143,
            right_density,
        )

        velocity = np.where(
            left,
            2.629369,
            0.0,
        )

        pressure = np.where(
            left,
            10.3333,
            1.0,
        )

        return conservative_state(
            density,
            velocity,
            pressure,
        )

    return initial_condition


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def run_trajectory(
    initial_condition,
    cells,
    length,
    final_time,
    cfl,
    snapshot_count,
):
    """Run a WENO7 Euler trajectory and retain evenly spaced snapshots."""

    simulation = eulerSimulation(
        cells,
        2,
        length,
        final_time,
        SSPRK3(),
        getEulerFlux(WENO7euler()),
        initial_condition,
        3,
        max_cfl=cfl,
        wave_speed_function=lambda state: np.max(
            np.abs(spds(state))
        ),
    )

    # runEuler returns (cells, times, variables).
    history = np.transpose(
        simulation.runEuler(),
        (1, 0, 2),
    )

    retained_count = min(
        snapshot_count,
        len(history),
    )

    snapshot_indices = np.unique(
        np.linspace(
            0,
            len(history) - 1,
            retained_count,
        ).astype(int)
    )

    return [
        history[index]
        for index in snapshot_indices
    ]


def generated_states(args, rng):
    """Yield category, trajectory ID and Euler state."""

    # ------------------------------------------------------------------
    # Smooth states
    # ------------------------------------------------------------------

    x_smooth = (
        np.arange(args.cells, dtype=float)
        * 10.0
        / args.cells
    )

    frequencies = (
        0.8,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        5.0,
        7.0,
        9.0,
        11.0,
    )

    phases = (
        0.0,
        0.7,
        1.3,
        2.1,
    )

    smooth_index = 0

    for frequency in frequencies:
        for phase in phases:
            amplitude = rng.uniform(0.05, 0.25)

            group_id = 'smooth_{:03d}'.format(
                smooth_index
            )

            state = smooth_wave(
                x_smooth,
                frequency,
                phase,
                amplitude,
            )

            yield 'smooth', group_id, state

            smooth_index += 1

    # ------------------------------------------------------------------
    # Sod-like shock tubes
    #
    # Sod must be evolved on [0, 1] until approximately t = 0.2.
    # The previous code incorrectly evolved Sod on [0, 10] until t = 1.8.
    # ------------------------------------------------------------------

    for variant in range(args.shock_variants):
        if args.shock_variants == 1:
            fraction = 0.5
        else:
            fraction = (
                variant
                / float(args.shock_variants - 1)
            )

        discontinuity = 0.35 + 0.30 * fraction
        right_density = 0.10 + 0.10 * fraction
        right_pressure = 0.08 + 0.08 * fraction

        initial_condition = make_riemann_initial_condition(
            discontinuity=discontinuity,
            right_density=right_density,
            right_pressure=right_pressure,
        )

        group_id = 'shock_{:03d}'.format(
            variant
        )

        states = run_trajectory(
            initial_condition=initial_condition,
            cells=args.cells,
            length=1.0,
            final_time=0.2,
            cfl=args.cfl,
            snapshot_count=args.snapshots,
        )

        for state in states:
            yield 'shock', group_id, state

    # ------------------------------------------------------------------
    # Shu--Osher-like shock/oscillation interactions
    # ------------------------------------------------------------------

    for variant in range(args.shu_variants):
        if args.shu_variants == 1:
            fraction = 0.5
        else:
            fraction = (
                variant
                / float(args.shu_variants - 1)
            )

        shock_position = 0.8 + 0.4 * fraction
        amplitude = 0.15 + 0.10 * fraction
        wave_number = 4.5 + 1.0 * fraction
        phase = 0.35 * variant

        initial_condition = make_shu_osher_initial_condition(
            shock_position=shock_position,
            amplitude=amplitude,
            wave_number=wave_number,
            phase=phase,
        )

        group_id = 'shu_{:03d}'.format(
            variant
        )

        states = run_trajectory(
            initial_condition=initial_condition,
            cells=args.cells,
            length=10.0,
            final_time=1.8,
            cfl=args.cfl,
            snapshot_count=args.snapshots,
        )

        for state in states:
            yield 'shock_oscillation', group_id, state

    # ------------------------------------------------------------------
    # Constant states
    # ------------------------------------------------------------------

    for variant in range(12):
        density = rng.uniform(0.5, 3.0)
        velocity = rng.uniform(-1.0, 2.5)
        pressure = rng.uniform(0.2, 12.0)

        constant_state = conservative_state(
            np.full(args.cells, density),
            np.full(args.cells, velocity),
            np.full(args.cells, pressure),
        )

        group_id = 'constant_{:03d}'.format(
            variant
        )

        yield 'constant', group_id, constant_state


# ---------------------------------------------------------------------------
# WENO-Z5 teacher
# ---------------------------------------------------------------------------

def weno_z5_reconstruction(values):
    """Return WENO-Z5 interface values from five-point stencils.

    Parameters
    ----------
    values:
        Array whose final axis has length five.

    Notes
    -----
    The stencil is normalized using its own minimum and range before
    calculating the nonlinear weights. The reconstructed target is then
    returned to its original physical scale.

    This is consistent with the min/max preprocessing used during WENO-NN
    training and inference.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.shape[-1] != 5:
        raise ValueError(
            'WENO-Z5 requires five-point stencils.'
        )

    minimum = np.min(
        values,
        axis=-1,
        keepdims=True,
    )

    maximum = np.max(
        values,
        axis=-1,
        keepdims=True,
    )

    value_range = maximum - minimum

    nonconstant = (
        value_range[..., 0] > 1.0e-14
    )

    normalized = np.zeros_like(
        values,
        dtype=float,
    )

    np.divide(
        values - minimum,
        value_range,
        out=normalized,
        where=value_range > 1.0e-14,
    )

    value0 = normalized[..., 0]
    value1 = normalized[..., 1]
    value2 = normalized[..., 2]
    value3 = normalized[..., 3]
    value4 = normalized[..., 4]

    candidate0 = (
        (1.0 / 3.0) * value0
        - (7.0 / 6.0) * value1
        + (11.0 / 6.0) * value2
    )

    candidate1 = (
        -(1.0 / 6.0) * value1
        + (5.0 / 6.0) * value2
        + (1.0 / 3.0) * value3
    )

    candidate2 = (
        (1.0 / 3.0) * value2
        + (5.0 / 6.0) * value3
        - (1.0 / 6.0) * value4
    )

    beta0 = (
        (13.0 / 12.0)
        * (value0 - 2.0 * value1 + value2)**2
        + 0.25
        * (value0 - 4.0 * value1 + 3.0 * value2)**2
    )

    beta1 = (
        (13.0 / 12.0)
        * (value1 - 2.0 * value2 + value3)**2
        + 0.25
        * (value1 - value3)**2
    )

    beta2 = (
        (13.0 / 12.0)
        * (value2 - 2.0 * value3 + value4)**2
        + 0.25
        * (3.0 * value2 - 4.0 * value3 + value4)**2
    )

    # WENO-Z global smoothness indicator.
    tau5 = np.abs(
        beta0 - beta2
    )

    epsilon = 1.0e-6
    power = 2.0

    alpha0 = (
        0.1
        * (
            1.0
            + (
                tau5
                / (beta0 + epsilon)
            )**power
        )
    )

    alpha1 = (
        0.6
        * (
            1.0
            + (
                tau5
                / (beta1 + epsilon)
            )**power
        )
    )

    alpha2 = (
        0.3
        * (
            1.0
            + (
                tau5
                / (beta2 + epsilon)
            )**power
        )
    )

    alpha_sum = (
        alpha0
        + alpha1
        + alpha2
    )

    weight0 = alpha0 / alpha_sum
    weight1 = alpha1 / alpha_sum
    weight2 = alpha2 / alpha_sum

    normalized_target = (
        weight0 * candidate0
        + weight1 * candidate1
        + weight2 * candidate2
    )

    target = (
        normalized_target
        * value_range[..., 0]
        + minimum[..., 0]
    )

    # A constant stencil must reconstruct exactly its constant value.
    target = np.where(
        nonconstant,
        target,
        values[..., 2],
    )

    return target


def weno7_reconstruction(values):
    """Return classical WENO7 interface values from seven-point stencils.

    Used as a higher-fidelity reconstruction target for the
    ``shock_oscillation`` category: instead of imitating WENO5's own
    five-point answer, the network is trained toward what a seventh-order
    scheme -- looking at a wider local neighborhood -- would reconstruct at
    the same interface. This gives the network room to systematically
    differ from (and improve on) WENO5 in the region where it needs to,
    rather than capping its target at WENO5's own quality everywhere.

    Same min/max normalize-then-rescale convention as
    ``weno_z5_reconstruction``, so targets stay on a comparable scale.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.shape[-1] != 7:
        raise ValueError(
            'WENO7 requires seven-point stencils.'
        )

    minimum = np.min(values, axis=-1, keepdims=True)
    maximum = np.max(values, axis=-1, keepdims=True)
    value_range = maximum - minimum
    nonconstant = value_range[..., 0] > 1.0e-14

    normalized = np.zeros_like(values, dtype=float)
    np.divide(
        values - minimum,
        value_range,
        out=normalized,
        where=value_range > 1.0e-14,
    )

    v0, v1, v2, v3, v4, v5, v6 = (normalized[..., i] for i in range(7))

    f1 = -1 / 4 * v0 + 13 / 12 * v1 - 23 / 12 * v2 + 25 / 12 * v3
    f2 = 1 / 12 * v1 - 5 / 12 * v2 + 13 / 12 * v3 + 1 / 4 * v4
    f3 = -1 / 12 * v2 + 7 / 12 * v3 + 7 / 12 * v4 - 1 / 12 * v5
    f4 = 1 / 4 * v3 + 13 / 12 * v4 - 5 / 12 * v5 + 1 / 12 * v6

    B1 = ((2107 / 240) * v3 ** 2 - (1567 / 40) * v3 * v2 + (3521 / 120) * v3 * v1 - (309 / 40) * v3 * v0
          + (11003 / 240) * v2 ** 2 - (8623 / 120) * v2 * v1 + (2321 / 120) * v2 * v0
          + (7043 / 240) * v1 ** 2 - (647 / 40) * v1 * v0 + (547 / 240) * v0 ** 2)

    B2 = ((3443 / 240) * v3 ** 2 - (1261 / 120) * v3 * v4 - (2983 / 120) * v3 * v2 + (267 / 40) * v3 * v1
          + (547 / 240) * v4 ** 2 + (961 / 120) * v4 * v2 - (247 / 120) * v4 * v1
          + (2843 / 240) * v2 ** 2 - (821 / 120) * v2 * v1 + (89 / 80) * v1 ** 2)

    B3 = ((3443 / 240) * v3 ** 2 - (2983 / 120) * v3 * v4 + (267 / 40) * v3 * v5 - (1261 / 120) * v3 * v2
          + (2843 / 240) * v4 ** 2 - (821 / 120) * v4 * v5 + (961 / 120) * v4 * v2
          + (89 / 80) * v5 ** 2 - (247 / 120) * v5 * v2 + (547 / 240) * v2 ** 2)

    B4 = ((2107 / 240) * v3 ** 2 - (1567 / 40) * v3 * v4 + (3521 / 120) * v3 * v5 - (309 / 40) * v3 * v6
          + (11003 / 240) * v4 ** 2 - (8623 / 120) * v4 * v5 + (2321 / 120) * v4 * v6
          + (7043 / 240) * v5 ** 2 - (647 / 40) * v5 * v6 + (547 / 240) * v6 ** 2)

    ep = 1.0e-6
    g1, g2, g3, g4 = 1 / 35, 12 / 35, 18 / 35, 4 / 35

    wt1 = g1 / (ep + B1) ** 2
    wt2 = g2 / (ep + B2) ** 2
    wt3 = g3 / (ep + B3) ** 2
    wt4 = g4 / (ep + B4) ** 2
    wts = wt1 + wt2 + wt3 + wt4

    normalized_target = (
        f1 * wt1 / wts
        + f2 * wt2 / wts
        + f3 * wt3 / wts
        + f4 * wt4 / wts
    )

    target = normalized_target * value_range[..., 0] + minimum[..., 0]

    target = np.where(
        nonconstant,
        target,
        values[..., 3],
    )

    return target


# ---------------------------------------------------------------------------
# Characteristic split-stencil extraction
# ---------------------------------------------------------------------------

def gather_stencils(
    values,
    offset,
    radius,
    reverse=False,
):
    """Match FiniteVolumeMethodEuler.partU for an arbitrary stencil width."""

    cell_count = len(values)

    indices = np.clip(
        np.arange(cell_count)[:, None]
        + offset
        + np.arange(-radius, radius + 1)[None, :],
        0,
        cell_count - 1,
    )

    gathered = np.transpose(
        values[indices],
        (0, 2, 1),
    )

    if reverse:
        gathered = np.flip(
            gathered,
            axis=2,
        )

    return gathered


def gather_five_point_stencils(
    values,
    offset,
    reverse=False,
):
    """Match FiniteVolumeMethodEuler.partU for a five-point stencil."""

    return gather_stencils(values, offset, radius=2, reverse=reverse)


def split_characteristic_stencils(state, use_reference_target=False):
    """Return matching five-point inputs and reconstruction targets.

    Positive and negative Lax--Friedrichs split characteristic fluxes are
    treated as independent scalar samples, exactly as they are reconstructed
    by the Euler solver. The five-point input stencil is always used (it
    must match what NNEuler5 sees at inference time); by default the
    target is WENO5-Z5's own reconstruction from that same five-point
    stencil. When ``use_reference_target`` is set, the target instead comes
    from ``weno7_reconstruction`` on a wider seven-point stencil around the
    same interface -- a higher-fidelity answer the network can learn to
    approach without being capped at WENO5's own quality.
    """

    state = np.asarray(
        state,
        dtype=float,
    )

    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError(
            'Euler state must have shape (cells, 3).'
        )

    if not np.isfinite(state).all():
        raise ValueError(
            'Euler state contains NaN or infinity.'
        )

    cell_count = len(state)

    physical_flux = flux(state)

    speeds = np.max(
        np.abs(spds(state)),
        axis=1,
    )

    right_speed = np.concatenate((
        speeds[1:],
        speeds[-1:],
    ))

    alpha = np.maximum(
        speeds,
        right_speed,
    )

    left_basis, _ = roe_eigenbasis(
        state,
        boundary='transmissive',
    )

    positive_state_stencil = gather_five_point_stencils(
        state,
        offset=0,
        reverse=False,
    )

    positive_flux_stencil = gather_five_point_stencils(
        physical_flux,
        offset=0,
        reverse=False,
    )

    negative_state_stencil = gather_five_point_stencils(
        state,
        offset=1,
        reverse=True,
    )

    negative_flux_stencil = gather_five_point_stencils(
        physical_flux,
        offset=1,
        reverse=True,
    )

    positive_characteristic_state = project_to_characteristic(
        positive_state_stencil,
        left_basis,
    )

    positive_characteristic_flux = project_to_characteristic(
        positive_flux_stencil,
        left_basis,
    )

    negative_characteristic_state = project_to_characteristic(
        negative_state_stencil,
        left_basis,
    )

    negative_characteristic_flux = project_to_characteristic(
        negative_flux_stencil,
        left_basis,
    )

    alpha = alpha[:, None, None]

    positive_split_flux = 0.5 * (
        positive_characteristic_flux
        + alpha * positive_characteristic_state
    )

    negative_split_flux = 0.5 * (
        negative_characteristic_flux
        - alpha * negative_characteristic_state
    )

    # Interfaces where the five-point positive or negative stencil was
    # clipped by the transmissive boundary (repeated edge cells rather than
    # true neighbors) are tagged rather than discarded, so the resulting
    # ghost-cell-clipped stencils can be trained on explicitly instead of
    # being entirely absent from the dataset.
    #
    # Positive stencil: i-2 ... i+2
    # Negative stencil before reversal: i-1 ... i+3
    #
    # Both are unclipped for 2 <= i <= cell_count-4.
    interface_indices = np.arange(
        cell_count
    )

    interior = (
        (interface_indices >= 2)
        & (interface_indices <= cell_count - 4)
    )

    field_count = positive_split_flux.shape[1]

    is_boundary_per_branch = np.broadcast_to(
        (~interior)[:, None],
        (cell_count, field_count),
    ).reshape(-1)

    inputs = np.concatenate((
        positive_split_flux,
        negative_split_flux,
    ), axis=0)

    if use_reference_target:
        positive_state_stencil_wide = gather_stencils(
            state, offset=0, radius=3, reverse=False,
        )
        positive_flux_stencil_wide = gather_stencils(
            physical_flux, offset=0, radius=3, reverse=False,
        )
        negative_state_stencil_wide = gather_stencils(
            state, offset=1, radius=3, reverse=True,
        )
        negative_flux_stencil_wide = gather_stencils(
            physical_flux, offset=1, radius=3, reverse=True,
        )

        positive_characteristic_state_wide = project_to_characteristic(
            positive_state_stencil_wide, left_basis,
        )
        positive_characteristic_flux_wide = project_to_characteristic(
            positive_flux_stencil_wide, left_basis,
        )
        negative_characteristic_state_wide = project_to_characteristic(
            negative_state_stencil_wide, left_basis,
        )
        negative_characteristic_flux_wide = project_to_characteristic(
            negative_flux_stencil_wide, left_basis,
        )

        positive_split_flux_wide = 0.5 * (
            positive_characteristic_flux_wide
            + alpha * positive_characteristic_state_wide
        )
        negative_split_flux_wide = 0.5 * (
            negative_characteristic_flux_wide
            - alpha * negative_characteristic_state_wide
        )

        wide_inputs = np.concatenate((
            positive_split_flux_wide,
            negative_split_flux_wide,
        ), axis=0)

        targets = weno7_reconstruction(wide_inputs)
    else:
        targets = weno_z5_reconstruction(
            inputs
        )

    is_boundary = np.concatenate((
        is_boundary_per_branch,
        is_boundary_per_branch,
    ))

    # Each characteristic component becomes an independent scalar sample.
    inputs = inputs.reshape(
        -1,
        5,
    )

    targets = targets.reshape(
        -1,
    )

    finite = (
        np.isfinite(inputs).all(axis=1)
        & np.isfinite(targets)
    )

    return (
        inputs[finite],
        targets[finite],
        is_boundary[finite],
    )


# ---------------------------------------------------------------------------
# Dataset balancing and sampling
# ---------------------------------------------------------------------------

def concatenate_category(category_items):
    """Concatenate all arrays collected for one physical category."""

    inputs = np.concatenate([
        item[0]
        for item in category_items
    ], axis=0)

    targets = np.concatenate([
        item[1]
        for item in category_items
    ], axis=0)

    groups = np.concatenate([
        item[2]
        for item in category_items
    ], axis=0)

    return inputs, targets, groups


def requested_category_counts(sample_count):
    """Return integer category quotas that sum exactly to sample_count.

    All categories but the last take floor(sample_count * proportion); the
    last category absorbs the rounding remainder so the quotas sum exactly.
    """

    categories = list(CATEGORY_PROPORTIONS)
    counts = {}
    allocated = 0

    for category in categories[:-1]:
        count = int(sample_count * CATEGORY_PROPORTIONS[category])
        counts[category] = count
        allocated += count

    counts[categories[-1]] = sample_count - allocated

    return counts


def sample_balanced_dataset(
    category_data,
    sample_count,
    rng,
):
    """Sample the requested category mixture without replacement."""

    requested = requested_category_counts(
        sample_count
    )

    selected_inputs = []
    selected_targets = []
    selected_groups = []
    selected_categories = []

    remaining_inputs = []
    remaining_targets = []
    remaining_groups = []
    remaining_categories = []

    for category in CATEGORY_PROPORTIONS:
        inputs, targets, groups = category_data[
            category
        ]

        permutation = rng.permutation(
            len(inputs)
        )

        take_count = min(
            requested[category],
            len(inputs),
        )

        selected = permutation[:take_count]
        unused = permutation[take_count:]

        selected_inputs.append(
            inputs[selected]
        )

        selected_targets.append(
            targets[selected]
        )

        selected_groups.append(
            groups[selected]
        )

        selected_categories.append(
            np.full(
                take_count,
                category,
                dtype=object,
            )
        )

        if len(unused):
            remaining_inputs.append(
                inputs[unused]
            )

            remaining_targets.append(
                targets[unused]
            )

            remaining_groups.append(
                groups[unused]
            )

            remaining_categories.append(
                np.full(
                    len(unused),
                    category,
                    dtype=object,
                )
            )

    current_count = sum(
        len(values)
        for values in selected_inputs
    )

    missing_count = (
        sample_count
        - current_count
    )

    if missing_count > 0:
        if not remaining_inputs:
            raise ValueError(
                'No unused samples remain to satisfy the requested count.'
            )

        extra_inputs = np.concatenate(
            remaining_inputs,
            axis=0,
        )

        extra_targets = np.concatenate(
            remaining_targets,
            axis=0,
        )

        extra_groups = np.concatenate(
            remaining_groups,
            axis=0,
        )

        extra_categories = np.concatenate(
            remaining_categories,
            axis=0,
        )

        if len(extra_inputs) < missing_count:
            raise ValueError(
                'Requested {} samples, but only {} unique samples are '
                'available. Reduce --samples or generate more trajectories.'
                .format(
                    sample_count,
                    current_count + len(extra_inputs),
                )
            )

        extra_selection = rng.choice(
            len(extra_inputs),
            size=missing_count,
            replace=False,
        )

        selected_inputs.append(
            extra_inputs[extra_selection]
        )

        selected_targets.append(
            extra_targets[extra_selection]
        )

        selected_groups.append(
            extra_groups[extra_selection]
        )

        selected_categories.append(
            extra_categories[extra_selection]
        )

    inputs = np.concatenate(
        selected_inputs,
        axis=0,
    )

    targets = np.concatenate(
        selected_targets,
        axis=0,
    )

    groups = np.concatenate(
        selected_groups,
        axis=0,
    )

    categories = np.concatenate(
        selected_categories,
        axis=0,
    )

    return inputs, targets, groups, categories


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def ensure_parent_directory(path):
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )


def main():
    args = parse_args()
    validate_args(args)

    rng = np.random.RandomState(
        args.seed
    )

    collected = {
        category: []
        for category in CATEGORY_PROPORTIONS
    }

    print(
        'Generating Euler trajectories...',
        flush=True,
    )

    trajectory_ids = set()
    state_count = 0

    for category, group_id, state in generated_states(
        args,
        rng,
    ):
        inputs, targets, is_boundary = split_characteristic_stencils(
            state,
            use_reference_target=(category == 'shock_oscillation'),
        )

        groups = np.full(
            len(inputs),
            group_id,
            dtype=object,
        )

        collected[category].append((
            inputs[~is_boundary],
            targets[~is_boundary],
            groups[~is_boundary],
        ))

        collected['boundary'].append((
            inputs[is_boundary],
            targets[is_boundary],
            groups[is_boundary],
        ))

        trajectory_ids.add(
            group_id
        )

        state_count += 1

    category_data = {}

    print(
        'Available samples before balancing:',
        flush=True,
    )

    for category in CATEGORY_PROPORTIONS:
        if not collected[category]:
            raise ValueError(
                'No samples were generated for category {}.'
                .format(category)
            )

        category_data[category] = concatenate_category(
            collected[category]
        )

        print(
            '  {:20s}: {}'.format(
                category,
                len(category_data[category][0]),
            ),
            flush=True,
        )

    (
        inputs,
        targets,
        groups,
        categories,
    ) = sample_balanced_dataset(
        category_data,
        args.samples,
        rng,
    )

    order = rng.permutation(
        len(inputs)
    )

    inputs = inputs[order]
    targets = targets[order]
    groups = groups[order]
    categories = categories[order]

    if not np.isfinite(inputs).all():
        raise ValueError(
            'Final input dataset contains NaN or infinity.'
        )

    if not np.isfinite(targets).all():
        raise ValueError(
            'Final target dataset contains NaN or infinity.'
        )

    for path in (
        args.output_inputs,
        args.output_targets,
        args.output_groups,
    ):
        ensure_parent_directory(path)

    # Preserve compatibility with Train_WENO5NN_Euler.py:
    # five rows and one column per sample.
    np.savetxt(
        args.output_inputs,
        inputs.T,
        delimiter=',',
    )

    np.savetxt(
        args.output_targets,
        targets[:, None],
        delimiter=',',
    )

    np.savetxt(
        args.output_groups,
        groups[:, None],
        delimiter=',',
        fmt='%s',
    )

    print(
        '\nGenerated dataset successfully.',
        flush=True,
    )

    print(
        '  States processed      : {}'.format(
            state_count
        )
    )

    print(
        '  Unique trajectory IDs : {}'.format(
            len(np.unique(groups))
        )
    )

    print(
        '  Total samples         : {}'.format(
            len(inputs)
        )
    )

    print(
        '  Input shape           : {}'.format(
            inputs.T.shape
        )
    )

    print(
        '  Target shape          : {}'.format(
            targets[:, None].shape
        )
    )

    print(
        '  Input file            : {}'.format(
            args.output_inputs
        )
    )

    print(
        '  Target file           : {}'.format(
            args.output_targets
        )
    )

    print(
        '  Group file            : {}'.format(
            args.output_groups
        )
    )

    print(
        '\nFinal category counts:',
        flush=True,
    )

    for category in CATEGORY_PROPORTIONS:
        print(
            '  {:20s}: {}'.format(
                category,
                int(np.sum(categories == category)),
            )
        )


if __name__ == '__main__':
    main()
