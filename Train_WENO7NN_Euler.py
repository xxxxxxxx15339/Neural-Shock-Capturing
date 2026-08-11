# -*- coding: utf-8 -*-
"""Train the WENO5-NN reconstruction on generated Euler data.

The model is trained on characteristic Euler reconstruction samples and
selected using numerical Sod and Shu--Osher benchmarks.

Important:
    - Training uses the training subset.
    - Early stopping uses the validation subset.
    - Sod and Shu--Osher are numerical model-selection benchmarks.
    - The test subset is evaluated only after a model has passed every
      numerical acceptance condition.
"""

import argparse
import datetime
import os
import random

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping
from keras.layers import Input, Lambda
from keras.models import Model as KerasModel

from src.analysis.euler_regression import (
    exact_sod_solution,
    interpolate_reference,
    primitive_variables,
    regression_metrics,
)
from src.data.preprocessing import scale_training_data
from src.initial_conditions.InitialConditions import shuOsher, sod
from src.networks.WenoNetworks import WENO71stOrder, _weno7_coefficients
from src.schemes import NNEuler7, WENO5euler, WENO7euler
from WENO5_vs_WENO5NN_Sod import run_history
from GenerateTrainingData_Euler import weno_z5_reconstruction


# ---------------------------------------------------------------------------
# Training output
# ---------------------------------------------------------------------------

class CompactTrainingLogger(Callback):
    """Print one compact line after every training epoch."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        print(
            '  Epoch {:03d} | loss {:.6e} | val_loss {:.6e}'.format(
                epoch + 1,
                logs.get('loss', float('nan')),
                logs.get('val_loss', float('nan')),
            ),
            flush=True,
        )


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train WENO5-NN using Euler characteristic stencils.'
    )

    # Dataset files
    parser.add_argument(
        '--input-data',
        default='data/eulerAvgsExact7.csv',
    )
    parser.add_argument(
        '--target-data',
        default='data/eulerFluxExact7.csv',
    )
    parser.add_argument(
        '--group-data',
        default='data/eulerGroupsExact7.csv',
        help=(
            'Optional sample-group file. Unique trajectory IDs are used for '
            'group-level splitting when enough distinct IDs are available.'
        ),
    )

    # Saved model
    parser.add_argument(
        '--model-path',
        default='trained_euler_weno7nn.h5',
    )

    # Neural-network training
    parser.add_argument(
        '--regularization',
        type=float,
        default=0.10,
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=4,
        help=(
            'Width of the correction MLP hidden layers. The original '
            'WENO71stOrder used 4 (matching the 4 WENO7 sub-stencils), '
            'which may be too small to learn corrections from a 7-point '
            'input; try larger values (e.g. 32).'
        ),
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=20,
    )

    # Euler benchmarks
    parser.add_argument(
        '--cells',
        type=int,
        default=120,
    )
    parser.add_argument(
        '--reference-cells',
        type=int,
        default=320,
    )
    parser.add_argument(
        '--cfl',
        type=float,
        default=0.35,
    )

    # Shu--Osher oscillatory region
    parser.add_argument(
        '--shu-region-start',
        type=float,
        default=5.5,
    )
    parser.add_argument(
        '--shu-region-end',
        type=float,
        default=7.3,
    )

    # Required improvement over classical WENO5.
    # 0.02 means that regional L1 must be at least 2% lower.
    parser.add_argument(
        '--required-shu-improvement',
        type=float,
        default=0.02,
    )

    # Hard-example reweighting: training samples whose target diverges from
    # what classical WENO5 would itself reconstruct (on the same stencil) by
    # more than --divergence-threshold are upweighted by
    # --divergence-boost. This concentrates gradient on the rare stencils
    # where beating WENO5 is actually possible, instead of that signal being
    # averaged away by the overwhelming majority of samples where any
    # reasonable scheme agrees with WENO5 almost exactly.
    parser.add_argument(
        '--divergence-threshold',
        type=float,
        default=0.01,
    )
    parser.add_argument(
        '--divergence-boost',
        type=float,
        default=50.0,
    )

    return parser.parse_args()


def validate_args(args):
    """Validate user-provided training and benchmark parameters."""

    if args.epochs < 1:
        raise ValueError('--epochs must be positive.')

    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive.')

    if args.patience < 1:
        raise ValueError('--patience must be positive.')

    if args.max_attempts < 1:
        raise ValueError('--max-attempts must be positive.')

    if args.cells < 7:
        raise ValueError('--cells must be at least 7.')

    if args.reference_cells < 7:
        raise ValueError('--reference-cells must be at least 7.')

    if args.learning_rate <= 0.0:
        raise ValueError('--learning-rate must be positive.')

    if args.regularization < 0.0:
        raise ValueError('--regularization cannot be negative.')

    if args.cfl <= 0.0:
        raise ValueError('--cfl must be positive.')

    if args.shu_region_start >= args.shu_region_end:
        raise ValueError(
            '--shu-region-start must be smaller than --shu-region-end.'
        )

    if not 0.0 <= args.required_shu_improvement < 1.0:
        raise ValueError(
            '--required-shu-improvement must belong to [0, 1).'
        )

    if args.divergence_threshold < 0.0:
        raise ValueError('--divergence-threshold cannot be negative.')

    if args.divergence_boost < 1.0:
        raise ValueError('--divergence-boost must be at least 1.0.')


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def configure_tensorflow_logging():
    """Use the TensorFlow 1.x or TensorFlow 2.x logging interface."""

    if hasattr(tf, 'logging'):
        tf.logging.set_verbosity(tf.logging.ERROR)
    else:
        tf.get_logger().setLevel('ERROR')


def set_random_seeds(seed):
    """Set Python, NumPy and TensorFlow random seeds."""

    random.seed(seed)
    np.random.seed(seed)

    if hasattr(tf, 'set_random_seed'):
        tf.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Dataset loading and splitting
# ---------------------------------------------------------------------------

def load_csv_data(input_path, target_path):
    """Load seven-point input stencils and scalar flux targets."""

    inputs = np.loadtxt(input_path, delimiter=',')
    targets = np.loadtxt(target_path, delimiter=',')

    if inputs.ndim != 2:
        raise ValueError('Euler input data must be a two-dimensional matrix.')

    if inputs.shape[0] != 7:
        raise ValueError(
            'Euler input data must have shape (7, samples); received {}.'
            .format(inputs.shape)
        )

    inputs = inputs.T
    targets = np.asarray(targets).reshape(-1, 1)

    if len(inputs) != len(targets):
        raise ValueError(
            'Euler input and target sample counts differ: {} versus {}.'
            .format(len(inputs), len(targets))
        )

    if not np.isfinite(inputs).all():
        raise ValueError('Euler input data contain NaN or infinity.')

    if not np.isfinite(targets).all():
        raise ValueError('Euler target data contain NaN or infinity.')

    return (
        inputs.astype(np.float32),
        targets.astype(np.float32),
    )


def load_optional_groups(group_path, sample_count):
    """Load optional group or trajectory IDs."""

    if not group_path:
        return None

    if not os.path.isfile(group_path):
        print(
            'Group file was not found. Using shuffled row-level splitting: {}'
            .format(group_path),
            flush=True,
        )
        return None

    groups = np.loadtxt(
        group_path,
        delimiter=',',
        dtype=str,
    )

    groups = np.asarray(groups).reshape(-1)

    if len(groups) != sample_count:
        raise ValueError(
            'Group count does not match sample count: {} versus {}.'
            .format(len(groups), sample_count)
        )

    return groups


def row_level_split(inputs, targets, seed):
    """Create shuffled 70/15/15 row-level subsets."""

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(inputs))

    train_end = int(0.70 * len(indices))
    validation_end = int(0.85 * len(indices))

    if (
        train_end == 0
        or validation_end <= train_end
        or validation_end >= len(indices)
    ):
        raise ValueError(
            'Dataset is too small for a 70/15/15 split.'
        )

    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    return (
        inputs[train_indices],
        targets[train_indices],
        inputs[validation_indices],
        targets[validation_indices],
        inputs[test_indices],
        targets[test_indices],
        'shuffled row-level split',
    )


def stratified_row_split(inputs, targets, groups, seed):
    """Create a row-level split while preserving broad group proportions."""

    rng = np.random.RandomState(seed)

    train_indices = []
    validation_indices = []
    test_indices = []

    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        indices = rng.permutation(indices)

        train_end = int(0.70 * len(indices))
        validation_end = int(0.85 * len(indices))

        train_indices.extend(indices[:train_end])
        validation_indices.extend(indices[train_end:validation_end])
        test_indices.extend(indices[validation_end:])

    train_indices = rng.permutation(np.asarray(train_indices, dtype=int))
    validation_indices = rng.permutation(
        np.asarray(validation_indices, dtype=int)
    )
    test_indices = rng.permutation(np.asarray(test_indices, dtype=int))

    if (
        len(train_indices) == 0
        or len(validation_indices) == 0
        or len(test_indices) == 0
    ):
        return row_level_split(inputs, targets, seed)

    return (
        inputs[train_indices],
        targets[train_indices],
        inputs[validation_indices],
        targets[validation_indices],
        inputs[test_indices],
        targets[test_indices],
        'stratified row-level split',
    )


def trajectory_group_split(inputs, targets, groups, seed):
    """Split complete trajectory groups into train, validation and test."""

    rng = np.random.RandomState(seed)

    unique_groups = np.unique(groups)
    unique_groups = rng.permutation(unique_groups)

    train_end = int(0.70 * len(unique_groups))
    validation_end = int(0.85 * len(unique_groups))

    if (
        train_end == 0
        or validation_end <= train_end
        or validation_end >= len(unique_groups)
    ):
        return stratified_row_split(
            inputs,
            targets,
            groups,
            seed,
        )

    train_groups = unique_groups[:train_end]
    validation_groups = unique_groups[train_end:validation_end]
    test_groups = unique_groups[validation_end:]

    train_mask = np.isin(groups, train_groups)
    validation_mask = np.isin(groups, validation_groups)
    test_mask = np.isin(groups, test_groups)

    if (
        not np.any(train_mask)
        or not np.any(validation_mask)
        or not np.any(test_mask)
    ):
        return stratified_row_split(
            inputs,
            targets,
            groups,
            seed,
        )

    return (
        inputs[train_mask],
        targets[train_mask],
        inputs[validation_mask],
        targets[validation_mask],
        inputs[test_mask],
        targets[test_mask],
        'trajectory-level group split',
    )


def split_dataset(inputs, targets, groups, seed):
    """Select the strongest split supported by the current group file."""

    if groups is None:
        return row_level_split(
            inputs,
            targets,
            seed,
        )

    unique_group_count = len(np.unique(groups))

    # The current generator may contain only the four broad labels:
    # smooth, shock, shock_oscillation and constant. These are not unique
    # trajectory IDs, so complete-group splitting would place entire physics
    # categories in different subsets. Use stratification in that case.
    if unique_group_count < 10:
        print(
            'Group file contains only {} unique labels. These appear to be '
            'broad data categories rather than trajectory IDs. Using a '
            'stratified row-level split.'.format(unique_group_count),
            flush=True,
        )

        return stratified_row_split(
            inputs,
            targets,
            groups,
            seed,
        )

    return trajectory_group_split(
        inputs,
        targets,
        groups,
        seed,
    )


# ---------------------------------------------------------------------------
# Euler simulation utilities
# ---------------------------------------------------------------------------

def run_final(
    scheme_builder,
    initial_condition,
    cells,
    length,
    final_time,
    cfl,
):
    """Run an Euler solution while retaining its full time history."""

    return run_history(
        scheme_builder,
        initial_condition,
        cells,
        length,
        final_time,
        cfl,
    )


def history_is_physical(history):
    """Check all Euler states for finite positive density and pressure."""

    states = np.asarray(history).reshape(-1, 3)

    if not np.isfinite(states).all():
        return False

    density = states[:, 0]

    if np.any(density <= 0.0):
        return False

    _, _, pressure = primitive_variables(states)

    if not np.isfinite(pressure).all():
        return False

    return bool(np.all(pressure > 0.0))


def final_total_variation(history):
    """Return density total variation at the final simulation time."""

    density = history[-1, :, 0]

    return float(
        np.sum(np.abs(np.diff(density)))
    )


def density_history_metrics(history):
    """Return maximum density TV and maximum local shock width in cells."""

    density = history[:, :, 0]

    total_variation = np.sum(
        np.abs(np.diff(density, axis=1)),
        axis=1,
    )

    gradients = np.abs(
        np.diff(density, axis=1)
    )

    widths = []

    for state, gradient_row in zip(density, gradients):
        center = int(np.argmax(gradient_row))

        # Keep the measurement local so the Shu--Osher oscillatory field is
        # not interpreted as one enormous shock.
        radius = min(8, len(state) // 4)

        left_index = max(0, center - radius)
        right_index = min(len(state) - 1, center + radius)

        left_state = state[left_index]
        right_state = state[right_index]

        jump = abs(right_state - left_state)

        if jump <= 1.0e-14:
            widths.append(0)
            continue

        lower = min(left_state, right_state) + 0.10 * jump
        upper = max(left_state, right_state) - 0.10 * jump

        local_start = max(0, center - radius)
        local_end = min(
            len(state),
            center + radius + 2,
        )

        local_state = state[local_start:local_end]

        transition = (
            (local_state >= lower)
            & (local_state <= upper)
        )

        center_local = center - local_start

        if (
            center_local >= len(transition)
            or not transition[center_local]
        ):
            widths.append(1)
            continue

        left = center_local
        right = center_local

        while left > 0 and transition[left - 1]:
            left -= 1

        while (
            right + 1 < len(transition)
            and transition[right + 1]
        ):
            right += 1

        widths.append(right - left + 1)

    return (
        float(np.max(total_variation)),
        int(np.max(widths)),
    )


def maximum_conservation_error(metrics):
    """Return the maximum component-wise conservation error."""

    return float(
        np.max(
            np.abs(
                np.asarray(metrics['conservation_error'], dtype=float)
            )
        )
    )


# ---------------------------------------------------------------------------
# Fixed references and WENO5 baselines
# ---------------------------------------------------------------------------

def make_evaluation_context(args):
    """Compute references and WENO5 baselines once before training."""

    print(
        'Computing analytical Sod reference...',
        flush=True,
    )

    sod_reference_x = np.linspace(
        0.0,
        1.0,
        args.reference_cells,
        endpoint=False,
    )

    sod_reference_fine = exact_sod_solution(
        sod_reference_x,
        0.2,
    )

    print(
        'Computing refined WENO7 Shu--Osher reference...',
        flush=True,
    )

    shu_reference_x, _, shu_reference_history = run_final(
        WENO7euler,
        shuOsher(),
        args.reference_cells,
        10.0,
        1.8,
        args.cfl,
    )

    print(
        'Computing fixed WENO5 Sod and Shu--Osher baselines...',
        flush=True,
    )

    sod_x, _, sod_weno_history = run_final(
        WENO5euler,
        sod(),
        args.cells,
        1.0,
        0.2,
        args.cfl,
    )

    shu_x, _, shu_weno_history = run_final(
        WENO5euler,
        shuOsher(),
        args.cells,
        10.0,
        1.8,
        args.cfl,
    )

    sod_reference = interpolate_reference(
        sod_reference_x,
        sod_reference_fine,
        sod_x,
    )

    shu_reference = interpolate_reference(
        shu_reference_x,
        shu_reference_history[-1],
        shu_x,
    )

    sod_initial = sod()(sod_x)
    shu_initial = shuOsher()(shu_x)

    sod_weno_metrics = regression_metrics(
        sod_x,
        sod_weno_history[-1],
        sod_initial,
        0.2,
        sod_reference,
    )

    shu_weno_metrics = regression_metrics(
        shu_x,
        shu_weno_history[-1],
        shu_initial,
        1.8,
        shu_reference,
    )

    regional_mask = (
        (shu_x >= args.shu_region_start)
        & (shu_x <= args.shu_region_end)
    )

    if not np.any(regional_mask):
        raise ValueError(
            'The selected Shu--Osher region contains no coarse-grid cells.'
        )

    shu_density_weno = shu_weno_history[-1, :, 0]
    shu_density_reference = shu_reference[:, 0]

    regional_l1_weno = float(
        np.mean(
            np.abs(
                shu_density_weno[regional_mask]
                - shu_density_reference[regional_mask]
            )
        )
    )

    regional_linf_weno = float(
        np.max(
            np.abs(
                shu_density_weno[regional_mask]
                - shu_density_reference[regional_mask]
            )
        )
    )

    sod_max_tv_weno, sod_width_weno = density_history_metrics(
        sod_weno_history
    )

    shu_max_tv_weno, shu_width_weno = density_history_metrics(
        shu_weno_history
    )

    return {
        'sod_x': sod_x,
        'shu_x': shu_x,

        'sod_reference': sod_reference,
        'shu_reference': shu_reference,

        'sod_initial': sod_initial,
        'shu_initial': shu_initial,

        'sod_weno_history': sod_weno_history,
        'shu_weno_history': shu_weno_history,

        'sod_weno_metrics': sod_weno_metrics,
        'shu_weno_metrics': shu_weno_metrics,

        'regional_mask': regional_mask,
        'shu_regional_l1_weno': regional_l1_weno,
        'shu_regional_linf_weno': regional_linf_weno,

        'sod_final_tv_weno': final_total_variation(
            sod_weno_history
        ),
        'sod_final_tv_reference': float(
            np.sum(
                np.abs(
                    np.diff(sod_reference[:, 0])
                )
            )
        ),
        'sod_max_tv_weno': sod_max_tv_weno,
        'sod_width_weno': sod_width_weno,

        'shu_final_tv_weno': final_total_variation(
            shu_weno_history
        ),
        'shu_final_tv_reference': float(
            np.sum(
                np.abs(
                    np.diff(shu_reference[:, 0])
                )
            )
        ),
        'shu_max_tv_weno': shu_max_tv_weno,
        'shu_width_weno': shu_width_weno,
    }


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(model, args, context):
    """Evaluate one neural model using Sod and Shu--Osher benchmarks."""

    sod_x = context['sod_x']
    shu_x = context['shu_x']

    _, _, sod_nn_history = run_final(
        lambda: NNEuler7(model),
        sod(),
        args.cells,
        1.0,
        0.2,
        args.cfl,
    )

    _, _, shu_nn_history = run_final(
        lambda: NNEuler7(model),
        shuOsher(),
        args.cells,
        10.0,
        1.8,
        args.cfl,
    )

    physical_pass = (
        history_is_physical(sod_nn_history)
        and history_is_physical(shu_nn_history)
    )

    if not physical_pass:
        raise FloatingPointError(
            'The neural Euler solution produced a nonphysical state.'
        )

    sod_nn = regression_metrics(
        sod_x,
        sod_nn_history[-1],
        context['sod_initial'],
        0.2,
        context['sod_reference'],
    )

    shu_nn = regression_metrics(
        shu_x,
        shu_nn_history[-1],
        context['shu_initial'],
        1.8,
        context['shu_reference'],
    )

    regional_mask = context['regional_mask']

    shu_density_nn = shu_nn_history[-1, :, 0]
    shu_density_reference = context['shu_reference'][:, 0]

    shu_regional_l1_nn = float(
        np.mean(
            np.abs(
                shu_density_nn[regional_mask]
                - shu_density_reference[regional_mask]
            )
        )
    )

    shu_regional_linf_nn = float(
        np.max(
            np.abs(
                shu_density_nn[regional_mask]
                - shu_density_reference[regional_mask]
            )
        )
    )

    sod_final_tv_nn = final_total_variation(
        sod_nn_history
    )

    shu_final_tv_nn = final_total_variation(
        shu_nn_history
    )

    sod_max_tv_nn, sod_width_nn = density_history_metrics(
        sod_nn_history
    )

    shu_max_tv_nn, shu_width_nn = density_history_metrics(
        shu_nn_history
    )

    regional_l1_target = (
        1.0 - args.required_shu_improvement
    ) * context['shu_regional_l1_weno']

    sod_tv_tolerance = 1.0e-3

    sod_tv_closer_to_reference = (
        abs(
            sod_final_tv_nn
            - context['sod_final_tv_reference']
        )
        <=
        abs(
            context['sod_final_tv_weno']
            - context['sod_final_tv_reference']
        )
        + sod_tv_tolerance
    )

    shu_tv_closer_to_reference = (
        abs(
            shu_final_tv_nn
            - context['shu_final_tv_reference']
        )
        <=
        abs(
            context['shu_final_tv_weno']
            - context['shu_final_tv_reference']
        )
    )

    # This prevents a model from passing merely because an excessively large
    # TV happens to be numerically closer to the refined reference TV.
    shu_tv_not_excessive = (
        shu_final_tv_nn
        <= 1.10 * context['shu_final_tv_reference']
    )

    shu_max_tv_guard = (
        shu_max_tv_nn
        <= 1.15 * max(
            context['shu_max_tv_weno'],
            context['shu_final_tv_reference'],
        )
    )

    criteria = {
        # General physical validity
        'physical_pass': bool(physical_pass),

        # Sod global errors
        'sod_density_l1': (
            sod_nn['density_l1'] < 0.012
        ),
        'sod_momentum_l1': (
            sod_nn['momentum_l1'] < 0.012
        ),
        'sod_energy_l1': (
            sod_nn['energy_l1'] < 0.025
        ),

        # Sod physical bounds
        'sod_minimum_density': (
            sod_nn['minimum_density'] > 0.12
        ),
        'sod_minimum_pressure': (
            sod_nn['minimum_pressure'] > 0.095
        ),
        'sod_density_overshoot': (
            sod_nn['density_overshoot'] < 2.0e-4
        ),
        'sod_pressure_overshoot': (
            sod_nn['pressure_overshoot'] < 2.0e-4
        ),

        # Sod wave locations and conservation
        'sod_shock_location': (
            sod_nn['shock_location_error'] < 0.02
        ),
        'sod_contact_location': (
            sod_nn['contact_location_error'] < 0.03
        ),
        'sod_conservation': (
            maximum_conservation_error(sod_nn) < 1.0e-8
        ),

        # Sod TV and shock width
        'sod_total_variation': (
            sod_tv_closer_to_reference
        ),
        'sod_shock_width': (
            sod_width_nn
            <= context['sod_width_weno'] + 1
        ),

        # Shu--Osher global errors
        'shu_density_l1': (
            shu_nn['density_l1'] < 0.16
        ),
        'shu_momentum_l1': (
            shu_nn['momentum_l1'] < 0.25
        ),
        'shu_energy_l1': (
            shu_nn['energy_l1'] < 0.95
        ),

        # Shu--Osher physical bounds
        'shu_minimum_density': (
            shu_nn['minimum_density'] > 0.75
        ),
        'shu_minimum_pressure': (
            shu_nn['minimum_pressure'] > 0.99
        ),
        'shu_density_overshoot': (
            shu_nn['density_overshoot'] < 0.05
        ),
        'shu_pressure_overshoot': (
            shu_nn['pressure_overshoot'] < 1.0e-3
        ),

        # Shu--Osher wave locations and conservation
        'shu_shock_location': (
            shu_nn['shock_location_error'] < 0.15
        ),
        'shu_contact_location': (
            shu_nn['contact_location_error'] < 0.5
        ),
        'shu_conservation': (
            maximum_conservation_error(shu_nn) < 0.003
        ),

        # Actual improvement in the oscillatory region
        'shu_regional_l1': (
            shu_regional_l1_nn <= regional_l1_target
        ),
        'shu_regional_linf': (
            shu_regional_linf_nn
            <= context['shu_regional_linf_weno']
        ),

        # Prevent artificial oscillation amplification
        'shu_total_variation_closer': (
            shu_tv_closer_to_reference
        ),
        'shu_total_variation_bound': (
            shu_tv_not_excessive
        ),
        'shu_maximum_tv_guard': (
            shu_max_tv_guard
        ),

        # Do not significantly widen the principal shock
        'shu_shock_width': (
            shu_width_nn
            <= context['shu_width_weno'] + 1
        ),
    }

    accepted = bool(
        all(criteria.values())
    )

    return {
        'sod': sod_nn,
        'shu': shu_nn,

        'sod_weno': context['sod_weno_metrics'],
        'shu_weno': context['shu_weno_metrics'],

        'shu_regional_l1_nn': shu_regional_l1_nn,
        'shu_regional_l1_weno': context[
            'shu_regional_l1_weno'
        ],
        'shu_regional_l1_target': regional_l1_target,

        'shu_regional_linf_nn': shu_regional_linf_nn,
        'shu_regional_linf_weno': context[
            'shu_regional_linf_weno'
        ],

        'sod_final_tv_nn': sod_final_tv_nn,
        'sod_final_tv_weno': context['sod_final_tv_weno'],
        'sod_final_tv_reference': context[
            'sod_final_tv_reference'
        ],
        'sod_max_tv_nn': sod_max_tv_nn,
        'sod_max_tv_weno': context['sod_max_tv_weno'],
        'sod_width_nn': sod_width_nn,
        'sod_width_weno': context['sod_width_weno'],

        'shu_final_tv_nn': shu_final_tv_nn,
        'shu_final_tv_weno': context['shu_final_tv_weno'],
        'shu_final_tv_reference': context[
            'shu_final_tv_reference'
        ],
        'shu_max_tv_nn': shu_max_tv_nn,
        'shu_max_tv_weno': context['shu_max_tv_weno'],
        'shu_width_nn': shu_width_nn,
        'shu_width_weno': context['shu_width_weno'],

        'physical_pass': physical_pass,
        'criteria': criteria,
        'accepted': accepted,
    }


# ---------------------------------------------------------------------------
# Reporting and candidate ranking
# ---------------------------------------------------------------------------

def print_candidate_metrics(metrics, validation_loss, duration):
    """Print numerical metrics and every acceptance condition."""

    print(
        '  Sod density/momentum/energy L1: '
        '{:.6e} / {:.6e} / {:.6e}'.format(
            metrics['sod']['density_l1'],
            metrics['sod']['momentum_l1'],
            metrics['sod']['energy_l1'],
        )
    )

    print(
        '  Shu density/momentum/energy L1: '
        '{:.6e} / {:.6e} / {:.6e}'.format(
            metrics['shu']['density_l1'],
            metrics['shu']['momentum_l1'],
            metrics['shu']['energy_l1'],
        )
    )

    print(
        '  Shu regional L1 NN/WENO5/target: '
        '{:.6e} / {:.6e} / {:.6e}'.format(
            metrics['shu_regional_l1_nn'],
            metrics['shu_regional_l1_weno'],
            metrics['shu_regional_l1_target'],
        )
    )

    print(
        '  Shu regional Linf NN/WENO5: '
        '{:.6e} / {:.6e}'.format(
            metrics['shu_regional_linf_nn'],
            metrics['shu_regional_linf_weno'],
        )
    )

    print(
        '  Sod final TV NN/WENO5/reference: '
        '{:.6e} / {:.6e} / {:.6e}'.format(
            metrics['sod_final_tv_nn'],
            metrics['sod_final_tv_weno'],
            metrics['sod_final_tv_reference'],
        )
    )

    print(
        '  Shu final TV NN/WENO5/reference: '
        '{:.6e} / {:.6e} / {:.6e}'.format(
            metrics['shu_final_tv_nn'],
            metrics['shu_final_tv_weno'],
            metrics['shu_final_tv_reference'],
        )
    )

    print(
        '  Shu maximum TV NN/WENO5: '
        '{:.6e} / {:.6e}'.format(
            metrics['shu_max_tv_nn'],
            metrics['shu_max_tv_weno'],
        )
    )

    print(
        '  Shock width Sod NN/WENO5: {} / {}'.format(
            metrics['sod_width_nn'],
            metrics['sod_width_weno'],
        )
    )

    print(
        '  Shock width Shu NN/WENO5: {} / {}'.format(
            metrics['shu_width_nn'],
            metrics['shu_width_weno'],
        )
    )

    print(
        '  Validation loss | {:.8e}'.format(
            validation_loss
        ),
        flush=True,
    )

    print(
        '  Duration        | {}'.format(duration),
        flush=True,
    )

    print('  Acceptance conditions:')

    for name, passed in metrics['criteria'].items():
        print(
            '    {:32s}: {}'.format(
                name,
                'PASS' if passed else 'FAIL',
            )
        )

    failed = [
        name
        for name, passed in metrics['criteria'].items()
        if not passed
    ]

    print(
        '  Accepted: {}'.format(
            metrics['accepted']
        )
    )

    if failed:
        print(
            '  Failed conditions: {}'.format(
                ', '.join(failed)
            )
        )


def candidate_score(metrics, validation_loss):
    """Rank rejected candidates without consulting the test set."""

    failed_count = sum(
        not passed
        for passed in metrics['criteria'].values()
    )

    regional_deficit = max(
        0.0,
        metrics['shu_regional_l1_nn']
        - metrics['shu_regional_l1_target'],
    )

    regional_linf_deficit = max(
        0.0,
        metrics['shu_regional_linf_nn']
        - metrics['shu_regional_linf_weno'],
    )

    tv_distance = abs(
        metrics['shu_final_tv_nn']
        - metrics['shu_final_tv_reference']
    )

    return (
        failed_count,
        regional_deficit,
        regional_linf_deficit,
        metrics['shu_regional_l1_nn'],
        tv_distance,
        validation_loss,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_args(args)

    configure_tensorflow_logging()
    set_random_seeds(args.seed)

    inputs, targets = load_csv_data(
        args.input_data,
        args.target_data,
    )

    groups = load_optional_groups(
        args.group_data,
        len(inputs),
    )

    # This normalization is consistent with the current NNEuler7 wrapper:
    # each input stencil and target are scaled with the stencil minimum/range.
    inputs, targets = scale_training_data(
        inputs,
        targets,
    )

    # A tiny fraction of scaled stencils (locally-constant sub-windows
    # combined with a sharp jump elsewhere) hit a genuine numerical
    # singularity in WENO7's smoothness-indicator formula (not a
    # gradient-explosion issue -- confirmed by direct forward-pass
    # inspection), producing NaN from the first training step onward.
    # Drop them before training; they are too rare (~0.01%) to matter.
    _singularity_input = Input(shape=(inputs.shape[1],))
    _singularity_probe = KerasModel(
        inputs=_singularity_input,
        outputs=Lambda(_weno7_coefficients)(_singularity_input),
    )
    _coefficients = _singularity_probe.predict(inputs, verbose=0, batch_size=8192)
    finite_stencil = np.isfinite(_coefficients).all(axis=1)
    if not finite_stencil.all():
        print(
            'Dropping {} / {} stencils that hit a WENO7 smoothness-'
            'indicator singularity after scaling.'.format(
                (~finite_stencil).sum(), len(finite_stencil),
            ),
            flush=True,
        )
        inputs = inputs[finite_stencil]
        targets = targets[finite_stencil]
        groups = groups[finite_stencil] if groups is not None else None

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        split_description,
    ) = split_dataset(
        inputs,
        targets,
        groups,
        args.seed,
    )

    print(
        'Dataset split: {}'.format(split_description),
        flush=True,
    )

    print(
        'Training Euler WENO5-NN with '
        'train/validation/test sizes {}/{}/{}.'.format(
            len(x_train),
            len(x_validation),
            len(x_test),
        ),
        flush=True,
    )

    print(
        'Batch size: {} | learning rate: {} | regularization: {}'
        .format(
            args.batch_size,
            args.learning_rate,
            args.regularization,
        ),
        flush=True,
    )

    # Reward beating WENO5 on the rare stencils where a better answer
    # actually exists: weight each training sample by how far its own
    # target diverges from what classical WENO5 would reconstruct from the
    # identical stencil. Samples whose target equals WENO5's own answer
    # (the overwhelming majority) get weight 1; samples that were
    # deliberately retargeted away from WENO5 (e.g. via
    # --use-reference-target in the data generator) and diverge by more
    # than --divergence-threshold get amplified by --divergence-boost, so
    # they aren't averaged away by the rest of the dataset.
    # weno_z5_reconstruction expects a 5-point stencil; use the 7-point
    # window's center 5 columns (indices 1..5), the same relative window
    # WENO5-Z5 itself would see.
    train_weno5_reference = weno_z5_reconstruction(x_train[:, 1:6])
    train_divergence = np.abs(y_train[:, 0] - train_weno5_reference)
    train_sample_weight = np.where(
        train_divergence > args.divergence_threshold,
        args.divergence_boost,
        1.0,
    )
    boosted_count = int(np.sum(train_sample_weight > 1.0))

    print(
        'Divergence reweighting: {} / {} training samples ({:.3f}%) '
        'diverge from WENO5 by more than {} and are upweighted x{}.'.format(
            boosted_count,
            len(train_sample_weight),
            100.0 * boosted_count / len(train_sample_weight),
            args.divergence_threshold,
            args.divergence_boost,
        ),
        flush=True,
    )

    print(
        'Shu--Osher evaluation region: [{}, {}]'.format(
            args.shu_region_start,
            args.shu_region_end,
        ),
        flush=True,
    )

    print(
        'Required Shu regional improvement: {:.2f}%'.format(
            100.0 * args.required_shu_improvement
        ),
        flush=True,
    )

    evaluation_context = make_evaluation_context(args)

    print(
        'WENO5 Shu regional L1 baseline: {:.8e}'.format(
            evaluation_context['shu_regional_l1_weno']
        ),
        flush=True,
    )

    best = None

    for attempt in range(1, args.max_attempts + 1):
        started = datetime.datetime.now()
        attempt_seed = args.seed + attempt - 1

        set_random_seeds(attempt_seed)

        print(
            '\n=== Euler attempt {} / {} | seed {} ==='.format(
                attempt,
                args.max_attempts,
                attempt_seed,
            ),
            flush=True,
        )

        model = WENO71stOrder(
            args.regularization,
            hidden_size=args.hidden_size,
        )

        model.compile(
            # clipnorm guards against NaN: WENO71stOrder's smoothness-
            # indicator formula has much larger constant coefficients than
            # WENO5's, so a single extreme-magnitude batch can produce a
            # gradient large enough to blow up weights even at a reduced
            # learning rate. Bounding the gradient norm per step is the
            # targeted fix, rather than shrinking the learning rate
            # globally (which doesn't protect against one bad batch).
            optimizer=optimizers.adam(
                lr=args.learning_rate,
                clipnorm=1.0,
            ),
            loss='mean_squared_error',
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            min_delta=1.0e-5,
            restore_best_weights=True,
        )

        logger = CompactTrainingLogger()

        training_history = model.fit(
            x_train,
            y_train,
            sample_weight=train_sample_weight,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(
                x_validation,
                y_validation,
            ),
            callbacks=[
                early_stopping,
                logger,
            ],
            verbose=0,
            shuffle=True,
        )

        validation_losses = training_history.history.get(
            'val_loss',
            [],
        )

        if not validation_losses:
            print(
                'Candidate rejected: no validation loss was recorded.',
                flush=True,
            )
            K.clear_session()
            continue

        best_validation_loss = float(
            np.min(validation_losses)
        )

        try:
            metrics = evaluate_candidate(
                model,
                args,
                evaluation_context,
            )
        except (
            FloatingPointError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            print(
                'Candidate rejected during Euler evaluation: {}'.format(
                    error
                ),
                flush=True,
            )

            K.clear_session()
            continue

        duration = datetime.datetime.now() - started

        print_candidate_metrics(
            metrics,
            best_validation_loss,
            duration,
        )

        score = candidate_score(
            metrics,
            best_validation_loss,
        )

        if best is None or score < best['score']:
            best = {
                'attempt': attempt,
                'seed': attempt_seed,
                'score': score,
                'metrics': metrics,
                'validation_loss': best_validation_loss,
            }

            # Always checkpoint the best-so-far candidate, even if it does
            # not clear every acceptance condition, so it can still be
            # inspected/run against the comparison scripts.
            best_checkpoint_path = args.model_path + '.best.h5'
            model.save(best_checkpoint_path)
            print(
                '  (checkpointed best-so-far candidate to {})'.format(
                    os.path.abspath(best_checkpoint_path)
                ),
                flush=True,
            )

        if metrics['accepted']:
            # The test set is consulted only once, after the candidate has
            # passed validation and every PDE acceptance condition.
            test_loss = float(
                model.evaluate(
                    x_test,
                    y_test,
                    verbose=0,
                )
            )

            model.save(
                args.model_path
            )

            print(
                '\nAccepted model saved successfully.',
                flush=True,
            )

            print(
                '  Model path | {}'.format(
                    os.path.abspath(args.model_path)
                ),
                flush=True,
            )

            print(
                '  Attempt    | {}'.format(attempt),
                flush=True,
            )

            print(
                '  Seed       | {}'.format(attempt_seed),
                flush=True,
            )

            print(
                '  Test loss  | {:.8e}'.format(test_loss),
                flush=True,
            )

            return

        K.clear_session()

    print(
        '\nNo Euler candidate met all acceptance conditions.',
        flush=True,
    )

    if best is not None:
        print(
            'Best rejected candidate:',
            flush=True,
        )

        print(
            '  Attempt         | {}'.format(
                best['attempt']
            )
        )

        print(
            '  Seed            | {}'.format(
                best['seed']
            )
        )

        print(
            '  Score           | {}'.format(
                best['score']
            )
        )

        print(
            '  Validation loss | {:.8e}'.format(
                best['validation_loss']
            )
        )

        failed = [
            name
            for name, passed
            in best['metrics']['criteria'].items()
            if not passed
        ]

        print(
            '  Failed criteria | {}'.format(
                ', '.join(failed)
            )
        )

        print(
            '  Shu regional NN/WENO5/target: '
            '{:.8e} / {:.8e} / {:.8e}'.format(
                best['metrics']['shu_regional_l1_nn'],
                best['metrics']['shu_regional_l1_weno'],
                best['metrics']['shu_regional_l1_target'],
            )
        )

    print(
        'No model was saved.',
        flush=True,
    )

    raise SystemExit(1)


if __name__ == '__main__':
    main()
