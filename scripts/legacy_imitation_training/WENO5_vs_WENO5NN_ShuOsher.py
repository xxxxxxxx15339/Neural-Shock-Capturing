# -*- coding: utf-8 -*-
"""Compare WENO5 Euler and WENO5 Euler-NN on the Shu--Osher problem."""

import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import os

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODEL_PATH
from src.initial_conditions.InitialConditions import shuOsher
from src.schemes import NNEuler5, WENO5euler
from WENO5_vs_WENO5NN_Sod import (
    discontinuity_width,
    primitive_variables,
    run_history,
    total_variation_history,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Run the WENO5-Euler/WENO5-Euler-NN Shu--Osher comparison, '
            'matching Stevens & Colonius Sec. 4.3: 300 cells, T=2, and a '
            'fine WENO5-JS reference (they use 12,800 cells; the default '
            'here is 3200 for tractable runtime -- pass a larger '
            '--reference-cells for a closer match to the paper).'
        )
    )
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--cells', type=int, default=300)
    parser.add_argument('--reference-cells', type=int, default=3200)
    parser.add_argument('--final-time', type=float, default=2.0)
    parser.add_argument('--cfl', type=float, default=0.4)
    parser.add_argument(
        '--reference-csv',
        help='CSV with columns x,density,momentum,energy. '
             'Defaults to a high-resolution WENO5-JS reference, matching '
             'the reference scheme used by the paper.',
    )
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument(
        '--save-dir', default='plots/shu_osher',
        help=(
            'Directory to save each figure as a PNG, so results are still '
            'visible if no GUI display is reachable (e.g. over SSH without '
            'X forwarding, where plt.show() silently does nothing). Pass '
            "an empty string ('') to disable saving."
        ),
    )
    return parser.parse_args()


def load_reference(args, initial_condition):
    if args.reference_csv:
        data = np.loadtxt(args.reference_csv, delimiter=',')
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(
                '--reference-csv must contain columns '
                'x,density,momentum,energy.'
            )
        return data[:, 0], data[:, 1:]

    x, _, history = run_history(
        WENO5euler, initial_condition, args.reference_cells, 10.0,
        args.final_time, args.cfl,
    )
    return x, history[-1]


def main():
    args = parse_args()
    if args.cells < 7 or args.reference_cells < 7:
        raise ValueError('Both cell counts must be at least 7.')
    if args.final_time <= 0:
        raise ValueError('--final-time must be positive.')

    model = load_model(args.model_path)
    initial_condition = shuOsher()
    x, times_weno, history_weno = run_history(
        WENO5euler, initial_condition, args.cells, 10.0,
        args.final_time, args.cfl,
    )
    _, times_nn, history_nn = run_history(
        lambda: NNEuler5(model), initial_condition, args.cells, 10.0,
        args.final_time, args.cfl,
    )
    reference_x, reference = load_reference(args, initial_condition)

    # Translate standard [-5, 5] reference coordinates if necessary.
    if reference_x[0] < 0 and x[0] >= 0:
        reference_x = reference_x - reference_x[0]
    solution_weno = history_weno[-1]
    solution_nn = history_nn[-1]
    solution_reference = np.column_stack([
        np.interp(x, reference_x, reference[:, component])
        for component in range(3)
    ])

    density_weno, velocity_weno, pressure_weno = primitive_variables(
        solution_weno
    )
    density_nn, velocity_nn, pressure_nn = primitive_variables(solution_nn)
    density_ref, velocity_ref, pressure_ref = primitive_variables(
        solution_reference
    )
    density_history_weno = history_weno[:, :, 0]
    density_history_nn = history_nn[:, :, 0]

    variables = (
        ('Density', density_weno, density_nn, density_ref),
        ('Velocity', velocity_weno, velocity_nn, velocity_ref),
        ('Pressure', pressure_weno, pressure_nn, pressure_ref),
    )
    saved_figures = []

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    for axis, (name, classical, neural, reference_state) in zip(
            axes, variables):
        axis.plot(x, classical, label='WENO5 Euler')
        axis.plot(x, neural, '--', label='WENO5 Euler-NN')
        axis.plot(x, reference_state, 'k:', linewidth=2,
                  label='Shu-Osher reference')
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel('x')
    figure.suptitle('Shu-Osher problem at t={:.4g}'.format(args.final_time))
    figure.tight_layout()
    saved_figures.append((figure, 'density_velocity_pressure'))

    dx = x[1] - x[0]
    figure = plt.figure()
    plt.plot(times_weno, discontinuity_width(density_history_weno, dx),
             label='WENO5 Euler')
    plt.plot(times_nn, discontinuity_width(density_history_nn, dx),
             label='WENO5 Euler-NN')
    plt.xlabel('t')
    plt.ylabel('Discontinuity Width')
    plt.legend()
    saved_figures.append((figure, 'discontinuity_width'))

    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(11, 4))
    for axis, times, density, title in (
        (axes[0], times_weno, density_history_weno, 'WENO5 Euler'),
        (axes[1], times_nn, density_history_nn, 'WENO5 Euler-NN'),
    ):
        image = axis.contourf(x, times, density, levels=40)
        axis.set_xlabel('x')
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    axes[0].set_ylabel('t')
    figure.tight_layout()
    saved_figures.append((figure, 'density_contour'))

    figure = plt.figure()
    plt.plot(x, density_weno, label='WENO5 Euler')
    plt.plot(x, density_nn, '--', label='WENO5 Euler-NN')
    plt.plot(x, density_ref, 'k:', linewidth=2,
             label='Shu-Osher reference')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    saved_figures.append((figure, 'density_final'))

    figure = plt.figure()
    plt.plot(times_weno, total_variation_history(density_history_weno),
             '.', label='WENO5 Euler')
    plt.plot(times_nn, total_variation_history(density_history_nn),
             '.', label='WENO5 Euler-NN')
    plt.xlabel('Time')
    plt.ylabel('Total Variation')
    plt.legend()
    saved_figures.append((figure, 'total_variation'))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for figure, name in saved_figures:
            path = os.path.join(args.save_dir, name + '.png')
            figure.savefig(path, dpi=130)
            print('saved {}'.format(path))

    print('model_path = {}'.format(args.model_path))
    print('reference = {}'.format(
        args.reference_csv or 'WENO5-JS, {} cells'.format(args.reference_cells)
    ))
    print('max_abs_difference = {:.6e}'.format(
        np.max(np.abs(solution_weno - solution_nn))
    ))
    print('max_abs_error_weno_vs_reference = {:.6e}'.format(
        np.max(np.abs(solution_weno - solution_reference))
    ))
    print('max_abs_error_nn_vs_reference = {:.6e}'.format(
        np.max(np.abs(solution_nn - solution_reference))
    ))

    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
