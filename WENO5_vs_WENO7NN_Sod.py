# -*- coding: utf-8 -*-
"""Compare classical WENO5 Euler with its neural-network reconstruction."""

import argparse
import os

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.euler_regression import exact_sod_solution, primitive_variables
from src.config import MODEL_PATH
from src.core.SimulationClasses import eulerSimulation
from src.core.TimeSteppingMethods import SSPRK3
from src.core.eulerEquations import getEulerFlux, spds
from src.initial_conditions.InitialConditions import sod
from src.schemes import NNEuler7, WENO5euler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the WENO5-Euler/WENO5-Euler-NN comparison.'
    )
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--cells', type=int, default=120)
    parser.add_argument('--final-time', type=float, default=0.2)
    parser.add_argument('--cfl', type=float, default=0.35)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument(
        '--save-dir', default='plots/sod_weno7nn',
        help=(
            'Directory to save each figure as a PNG. Pass an empty string '
            "('') to disable saving."
        ),
    )
    return parser.parse_args()


def run_history(scheme_builder, initial_condition, cells, length, final_time,
                cfl):
    """Run an adaptive Euler simulation and return x, t, and state history."""
    x = np.arange(cells, dtype=float) * length / cells
    simulation = eulerSimulation(
        cells, 2, length, final_time, SSPRK3(),
        getEulerFlux(scheme_builder()), initial_condition, 3,
        max_cfl=cfl,
        wave_speed_function=lambda state: np.max(np.abs(spds(state))),
    )
    # Euler's adaptive solver returns (cells, time, components).
    history = np.transpose(simulation.runEuler(), (1, 0, 2))
    return x, simulation.last_times, history


def discontinuity_width(density, dx):
    """Estimate the width of the steepest density transition in cells."""
    gradient = np.abs(np.diff(density, axis=1))
    threshold = 0.01 * np.max(gradient, axis=1, keepdims=True)
    return dx * np.sum(gradient > threshold, axis=1)


def total_variation_history(density):
    return np.sum(np.abs(np.diff(density, axis=1)), axis=1)


def main():
    args = parse_args()
    if args.cells < 7:
        raise ValueError('--cells must be at least 7 for a WENO5 stencil.')
    if args.final_time <= 0:
        raise ValueError('--final-time must be positive.')

    model = load_model(args.model_path)
    initial_condition = sod()
    x, times_weno, history_weno = run_history(
        WENO5euler, initial_condition, args.cells, 1.0,
        args.final_time, args.cfl,
    )
    _, times_nn, history_nn = run_history(
        lambda: NNEuler7(model), initial_condition, args.cells, 1.0,
        args.final_time, args.cfl,
    )

    solution_weno = history_weno[-1]
    solution_nn = history_nn[-1]
    solution_exact = exact_sod_solution(x, args.final_time)
    density_history_weno = history_weno[:, :, 0]
    density_history_nn = history_nn[:, :, 0]

    density_weno, velocity_weno, pressure_weno = primitive_variables(
        solution_weno
    )
    density_nn, velocity_nn, pressure_nn = primitive_variables(solution_nn)
    density_exact, velocity_exact, pressure_exact = primitive_variables(
        solution_exact
    )

    variables = (
        ('Density', density_weno, density_nn, density_exact),
        ('Velocity', velocity_weno, velocity_nn, velocity_exact),
        ('Pressure', pressure_weno, pressure_nn, pressure_exact),
    )
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    for axis, (name, classical, neural, exact) in zip(axes, variables):
        axis.plot(x, classical, label='WENO5 Euler')
        axis.plot(x, neural, '--', label='WENO7-NN')
        axis.plot(x, exact, 'k:', linewidth=2, label='Sod exact')
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel('x')
    figure.suptitle('Sod problem at t={:.4g}'.format(args.final_time))
    figure.tight_layout()

    # Equivalent diagnostics to WENO5_vs_WENO5NN_StepAdvection.py, applied to Euler density.
    dx = x[1] - x[0]
    plt.figure()
    plt.plot(times_weno, discontinuity_width(density_history_weno, dx),
             label='WENO5 Euler')
    plt.plot(times_nn, discontinuity_width(density_history_nn, dx),
             label='WENO7-NN')
    plt.xlabel('t')
    plt.ylabel('Discontinuity Width')
    plt.legend()

    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(11, 4))
    for axis, times, density, title in (
        (axes[0], times_weno, density_history_weno, 'WENO5 Euler'),
        (axes[1], times_nn, density_history_nn, 'WENO7-NN'),
    ):
        image = axis.contourf(x, times, density, levels=40)
        axis.set_xlabel('x')
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    axes[0].set_ylabel('t')
    figure.tight_layout()

    # Interpolate NN density onto the classical method's adaptive time grid.
    density_nn_common = np.column_stack([
        np.interp(times_weno, times_nn, density_history_nn[:, i])
        for i in range(args.cells)
    ])
    plt.figure()
    error = np.sqrt(np.mean((density_history_weno-density_nn_common)**2,
                            axis=1))
    plt.plot(times_weno, error, '.', label='WENO7-NN vs WENO5')
    plt.xlabel('Time')
    plt.ylabel('L2 Error')
    plt.legend()

    plt.figure()
    plt.plot(times_weno, total_variation_history(density_history_weno),
             '.', label='WENO5 Euler')
    plt.plot(times_nn, total_variation_history(density_history_nn),
             '.', label='WENO7-NN')
    plt.xlabel('Time')
    plt.ylabel('Total Variation')
    plt.legend()

    plt.figure()
    plt.plot(x, density_weno, label='WENO5 Euler')
    plt.plot(x, density_nn, '--', label='WENO7-NN')
    plt.plot(x, density_exact, 'k:', linewidth=2, label='Sod exact')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()

    print('model_path = {}'.format(args.model_path))
    print('max_abs_difference = {:.6e}'.format(
        np.max(np.abs(solution_weno - solution_nn))
    ))
    print('max_abs_error_weno_vs_exact = {:.6e}'.format(
        np.max(np.abs(solution_weno - solution_exact))
    ))
    print('max_abs_error_nn_vs_exact = {:.6e}'.format(
        np.max(np.abs(solution_nn - solution_exact))
    ))
    print('WENO5 Euler final state shape = {}'.format(solution_weno.shape))
    print('WENO7-NN final state shape = {}'.format(solution_nn.shape))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        figure_names = (
            'density_velocity_pressure',
            'discontinuity_width',
            'density_contour',
            'l2_error',
            'total_variation',
            'density_final',
        )
        for num, name in zip(plt.get_fignums(), figure_names):
            path = os.path.join(args.save_dir, name + '.png')
            plt.figure(num).savefig(path, dpi=130)
            print('saved {}'.format(path))

    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
