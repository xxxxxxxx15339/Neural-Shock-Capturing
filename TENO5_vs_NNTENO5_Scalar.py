# -*- coding: utf-8 -*-
"""Compare deterministic TENO5 and NN-TENO5 on scalar conservation laws.

The script writes numerical metrics to JSON and displays interactive
solution/error plots for each benchmark plus one combined overview figure.
"""

import argparse
import json
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from src.core.Equations import adv, invBurg
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation
from src.core.TimeSteppingMethods import SSPRK3
from src.schemes import NN_TENO5, TENO5


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare TENO5 and NN-TENO5 on scalar PDE benchmarks.'
    )
    parser.add_argument(
        '--model-path',
        default='models/nn_teno5.h5',
    )
    parser.add_argument(
        '--metadata-path',
        default='models/nn_teno5_metadata.json',
    )
    parser.add_argument(
        '--output-path',
        default='models/nn_teno5_scalar_comparison.json',
    )
    parser.add_argument(
        '--cells',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--cfl',
        type=float,
        default=0.4,
    )
    parser.add_argument(
        '--fidelity-factor',
        type=float,
        default=1.02,
    )
    parser.add_argument(
        '--save-dir',
        default='models/nn_teno5_scalar_comparison_plots',
        help=(
            'Directory to save each figure as a PNG. Pass an empty string '
            "('') to disable saving."
        ),
    )
    parser.add_argument('--no-show', action='store_true')
    return parser.parse_args()


def validate_args(args):
    if args.cells < 20:
        raise ValueError('--cells must be at least 20.')
    if not 0.0 < args.cfl <= 1.0:
        raise ValueError('--cfl must belong to (0, 1].')
    if args.fidelity_factor < 1.0:
        raise ValueError('--fidelity-factor must be at least one.')


def sine_profile(x):
    return np.sin(2.0 * np.pi * x)


def step_profile(x):
    periodic_x = np.mod(x, 1.0)
    return (
        (periodic_x >= 0.25)
        & (periodic_x < 0.75)
    ).astype(float)


def multifrequency_profile(x):
    return (
        np.sin(2.0 * np.pi * x)
        + 0.25 * np.sin(6.0 * np.pi * x)
        + 0.10 * np.cos(10.0 * np.pi * x)
    )


def burgers_initial_profile(x):
    return 0.5 + 0.25 * np.sin(2.0 * np.pi * x)


def burgers_characteristic_reference(x, final_time):
    """Evaluate the exact smooth Burgers solution before shock formation."""

    shock_time = 2.0 / np.pi

    if final_time >= shock_time:
        raise ValueError(
            'The characteristic reference requires time < {:.8f}.'
            .format(shock_time)
        )

    # Before shock formation, xi -> xi + t*u0(xi) is strictly increasing.
    # Select the periodic image of x in the image of xi in [0, 1], then use
    # bisection. This avoids Newton iterations jumping to the wrong periodic
    # branch near the steep front.
    mapped_start = (
        final_time
        * burgers_initial_profile(
            np.array([0.0])
        )[0]
    )
    target = np.mod(
        x,
        1.0,
    )
    target = np.where(
        target < mapped_start,
        target + 1.0,
        target,
    )

    lower = np.zeros_like(target)
    upper = np.ones_like(target)

    for _ in range(60):
        midpoint = 0.5 * (
            lower + upper
        )
        mapped_midpoint = (
            midpoint
            + final_time
            * burgers_initial_profile(midpoint)
        )
        belongs_right = (
            mapped_midpoint < target
        )
        lower = np.where(
            belongs_right,
            midpoint,
            lower,
        )
        upper = np.where(
            belongs_right,
            upper,
            midpoint,
        )

    foot = 0.5 * (
        lower + upper
    )
    return burgers_initial_profile(foot)


def build_problems():
    return (
        {
            'name': 'smooth_sine_advection',
            'equation': adv(),
            'wave_speed': 1.0,
            'final_time': 1.0,
            'initial': sine_profile,
            'reference': lambda x, final_time: sine_profile(
                x - final_time
            ),
        },
        {
            'name': 'step_advection',
            'equation': adv(),
            'wave_speed': 1.0,
            'final_time': 1.0,
            'initial': step_profile,
            'reference': lambda x, final_time: step_profile(
                x - final_time
            ),
        },
        {
            'name': 'multifrequency_advection',
            'equation': adv(),
            'wave_speed': 1.0,
            'final_time': 1.0,
            'initial': multifrequency_profile,
            'reference': lambda x, final_time: multifrequency_profile(
                x - final_time
            ),
        },
        {
            'name': 'burgers_smooth_to_shock',
            'equation': invBurg(),
            'wave_speed': 1.0,
            'final_time': 0.60,
            'initial': burgers_initial_profile,
            'reference': burgers_characteristic_reference,
        },
    )


def run_solution(problem, scheme, cells, cfl):
    dx = 1.0 / cells
    step_count = int(np.ceil(
        problem['final_time']
        * problem['wave_speed']
        / (cfl * dx)
    ))
    time_count = step_count + 1
    flux_splitting = LaxFriedrichs(
        problem['equation'],
        problem['wave_speed'],
    )
    simulation = Simulation(
        cells,
        time_count,
        1.0,
        problem['final_time'],
        SSPRK3(),
        flux_splitting,
        scheme,
        problem['initial'],
        max_cfl=cfl,
    )

    started = time.perf_counter()
    history = simulation.run()
    runtime = time.perf_counter() - started
    solution = history[:, -1]

    if not np.isfinite(solution).all():
        raise FloatingPointError(
            '{} produced non-finite values.'
            .format(problem['name'])
        )

    return solution, runtime, step_count


def calculate_metrics(solution, reference, initial, dx, runtime):
    error = solution - reference
    return {
        'l1_error': float(np.mean(np.abs(error))),
        'l2_error': float(np.sqrt(np.mean(error ** 2))),
        'linf_error': float(np.max(np.abs(error))),
        'total_variation': float(np.sum(
            np.abs(solution - np.roll(solution, 1))
        )),
        'overshoot': float(max(
            0.0,
            np.max(solution) - np.max(reference),
        )),
        'undershoot': float(max(
            0.0,
            np.min(reference) - np.min(solution),
        )),
        'conservation_error': float(abs(
            dx * np.sum(solution)
            - dx * np.sum(initial)
        )),
        'runtime_seconds': float(runtime),
    }


def array_difference(left, right):
    difference = np.asarray(left) - np.asarray(right)
    return {
        'l1_difference': float(
            np.mean(np.abs(difference))
        ),
        'l2_difference': float(
            np.sqrt(np.mean(difference ** 2))
        ),
        'linf_difference': float(
            np.max(np.abs(difference))
        ),
        'exactly_equal': bool(
            np.array_equal(left, right)
        ),
        'differing_cells': int(
            np.count_nonzero(difference)
        ),
    }


def compare_problem(
    problem,
    deterministic_scheme,
    neural_scheme,
    all_discard_scheme,
    all_keep_scheme,
    cells,
    cfl,
    fidelity_factor,
):
    x = np.linspace(
        0.0,
        1.0,
        cells,
        endpoint=False,
    )
    dx = 1.0 / cells
    initial = problem['initial'](x)
    reference = problem['reference'](
        x,
        problem['final_time'],
    )

    deterministic, deterministic_runtime, step_count = run_solution(
        problem,
        deterministic_scheme,
        cells,
        cfl,
    )
    neural, neural_runtime, neural_step_count = run_solution(
        problem,
        neural_scheme,
        cells,
        cfl,
    )
    all_discard, all_discard_runtime, discard_step_count = run_solution(
        problem,
        all_discard_scheme,
        cells,
        cfl,
    )
    all_keep, all_keep_runtime, keep_step_count = run_solution(
        problem,
        all_keep_scheme,
        cells,
        cfl,
    )

    if (
        neural_step_count != step_count
        or discard_step_count != step_count
        or keep_step_count != step_count
    ):
        raise AssertionError('Comparison runs used different time grids.')

    deterministic_metrics = calculate_metrics(
        deterministic,
        reference,
        initial,
        dx,
        deterministic_runtime,
    )
    neural_metrics = calculate_metrics(
        neural,
        reference,
        initial,
        dx,
        neural_runtime,
    )
    all_discard_metrics = calculate_metrics(
        all_discard,
        reference,
        initial,
        dx,
        all_discard_runtime,
    )
    all_keep_metrics = calculate_metrics(
        all_keep,
        reference,
        initial,
        dx,
        all_keep_runtime,
    )

    fidelity_limit = (
        fidelity_factor
        * deterministic_metrics['l1_error']
    )

    result = {
        'cells': cells,
        'time_steps': step_count,
        'final_time': problem['final_time'],
        'teno5': deterministic_metrics,
        'nn_teno5': neural_metrics,
        'teno5_vs_nn_teno5': array_difference(
            deterministic,
            neural,
        ),
        'threshold_wiring_test': {
            'all_discard': {
                'thresholds': [0.0, 0.0, 0.0],
                'expected_behavior': 'first_order_fallback',
                'metrics': all_discard_metrics,
            },
            'all_keep': {
                'thresholds': [1.0, 1.0, 1.0],
                'expected_behavior': 'linear_fifth_order',
                'metrics': all_keep_metrics,
            },
            'all_discard_vs_all_keep': array_difference(
                all_discard,
                all_keep,
            ),
            'calibrated_vs_all_discard': array_difference(
                neural,
                all_discard,
            ),
            'calibrated_vs_all_keep': array_difference(
                neural,
                all_keep,
            ),
            'passed': bool(
                not np.array_equal(
                    all_discard,
                    all_keep,
                )
            ),
        },
        'teacher_fidelity': {
            'metric': 'l1_error',
            'factor': fidelity_factor,
            'limit': float(fidelity_limit),
            'passed': bool(
                neural_metrics['l1_error']
                <= fidelity_limit
            ),
        },
        'beats_teacher': {
            'metric': 'l1_error',
            'teno5_error': deterministic_metrics['l1_error'],
            'nn_teno5_error': neural_metrics['l1_error'],
            'passed': bool(
                neural_metrics['l1_error']
                <= deterministic_metrics['l1_error']
            ),
        },
    }

    plot_data = {
        'x': x,
        'reference': reference,
        'teno5': deterministic,
        'nn_teno5': neural,
        'all_discard': all_discard,
        'all_keep': all_keep,
    }

    return result, plot_data


def problem_title(problem):
    return '{} at t={:g}'.format(
        problem['name'].replace('_', ' ').title(),
        problem['final_time'],
    )


def draw_solution(axis, problem, plot_data):
    axis.plot(
        plot_data['x'],
        plot_data['reference'],
        color='black',
        linewidth=2.0,
        label='Exact/reference',
    )
    axis.plot(
        plot_data['x'],
        plot_data['teno5'],
        color='#1f77b4',
        linestyle='--',
        linewidth=1.6,
        label='TENO5',
    )
    axis.plot(
        plot_data['x'],
        plot_data['nn_teno5'],
        color='#d62728',
        linestyle=':',
        linewidth=1.8,
        label='NN-TENO5',
    )
    axis.set_title(problem_title(problem))
    axis.set_xlabel('x')
    axis.set_ylabel('u(x, t)')
    axis.grid(True, alpha=0.25)


def plot_problem(problem, plot_data):
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [2.0, 1.0, 1.0]},
    )

    draw_solution(
        axes[0],
        problem,
        plot_data,
    )
    axes[0].legend(loc='best')

    axes[1].plot(
        plot_data['x'],
        plot_data['teno5'] - plot_data['reference'],
        color='#1f77b4',
        linestyle='--',
        linewidth=1.4,
        label='TENO5 error',
    )
    axes[1].plot(
        plot_data['x'],
        plot_data['nn_teno5'] - plot_data['reference'],
        color='#d62728',
        linestyle=':',
        linewidth=1.6,
        label='NN-TENO5 error',
    )
    axes[1].axhline(
        0.0,
        color='black',
        linewidth=0.8,
        alpha=0.7,
    )
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Pointwise error')
    axes[1].ticklabel_format(
        axis='y',
        style='sci',
        scilimits=(-3, 3),
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc='best')

    axes[2].plot(
        plot_data['x'],
        plot_data['nn_teno5'],
        color='#d62728',
        linewidth=1.6,
        label='Calibrated NN-TENO5',
    )
    axes[2].plot(
        plot_data['x'],
        plot_data['all_discard'],
        color='#9467bd',
        linestyle='--',
        linewidth=1.4,
        label='Threshold 0: discard all',
    )
    axes[2].plot(
        plot_data['x'],
        plot_data['all_keep'],
        color='#2ca02c',
        linestyle=':',
        linewidth=1.6,
        label='Threshold 1: keep all',
    )
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('Wiring test')
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc='best')

    figure.tight_layout()
    return figure


def plot_overview(problem_plot_data):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
    )

    for axis, (problem, plot_data) in zip(
        axes.flat,
        problem_plot_data,
    ):
        draw_solution(
            axis,
            problem,
            plot_data,
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc='lower center',
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return figure


def main():
    args = parse_args()
    validate_args(args)

    with open(
        args.metadata_path,
        'r',
        encoding='utf-8',
    ) as stream:
        metadata = json.load(stream)

    thresholds = metadata['calibrated_thresholds']
    model = load_model(args.model_path)

    deterministic_scheme = TENO5()
    neural_scheme = NN_TENO5(
        model,
        threshold=thresholds,
    )
    all_discard_scheme = NN_TENO5(
        model,
        threshold=[0.0, 0.0, 0.0],
    )
    all_keep_scheme = NN_TENO5(
        model,
        threshold=[1.0, 1.0, 1.0],
    )

    results = {
        'model_path': os.path.abspath(args.model_path),
        'metadata_path': os.path.abspath(args.metadata_path),
        'calibrated_thresholds': thresholds,
        'cells': args.cells,
        'cfl': args.cfl,
        'fidelity_factor': args.fidelity_factor,
        'problems': {},
    }

    problem_plot_data = []
    figures = []

    for problem in build_problems():
        print(
            'Running {}...'.format(problem['name']),
            flush=True,
        )
        result, plot_data = compare_problem(
            problem,
            deterministic_scheme,
            neural_scheme,
            all_discard_scheme,
            all_keep_scheme,
            args.cells,
            args.cfl,
            args.fidelity_factor,
        )
        figures.append(plot_problem(
            problem,
            plot_data,
        ))
        results['problems'][problem['name']] = result
        problem_plot_data.append(
            (
                problem,
                plot_data,
            )
        )

        print(
            '  TENO5 L1={:.8e}, NN-TENO5 L1={:.8e}, fidelity={}'
            .format(
                result['teno5']['l1_error'],
                result['nn_teno5']['l1_error'],
                result['teacher_fidelity']['passed'],
            )
        )
        direct_difference = result['teno5_vs_nn_teno5']
        wiring = result['threshold_wiring_test']
        extreme_difference = wiring['all_discard_vs_all_keep']

        print(
            '  Direct difference: L1={:.8e}, Linf={:.8e}, '
            'exactly_equal={}'.format(
                direct_difference['l1_difference'],
                direct_difference['linf_difference'],
                direct_difference['exactly_equal'],
            )
        )
        print(
            '  Threshold wiring: discard-vs-keep Linf={:.8e}, '
            'exactly_equal={}, passed={}'.format(
                extreme_difference['linf_difference'],
                extreme_difference['exactly_equal'],
                wiring['passed'],
            )
        )
        beats = result['beats_teacher']
        print(
            '  vs. real solution: TENO5={:.8e}, NN-TENO5={:.8e}, '
            'NN beats TENO5={}'.format(
                beats['teno5_error'],
                beats['nn_teno5_error'],
                beats['passed'],
            )
        )

    beats_results = [
        result['beats_teacher']['passed']
        for result in results['problems'].values()
    ]
    print(
        '\nBeats-real-solution tally: {}/{} benchmarks'.format(
            sum(beats_results),
            len(beats_results),
        )
    )

    figures.append(plot_overview(
        problem_plot_data,
    ))

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(
        args.output_path,
        'w',
        encoding='utf-8',
    ) as stream:
        json.dump(
            results,
            stream,
            indent=2,
            sort_keys=True,
        )

    print(
        '\nComparison written to {}'.format(
            os.path.abspath(args.output_path)
        )
    )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        figure_names = [
            problem['name']
            for problem, _ in problem_plot_data
        ] + ['teno5_nn_teno5_overview']
        for name, figure in zip(figure_names, figures):
            path = os.path.join(args.save_dir, name + '.png')
            figure.savefig(path, dpi=130)
            print('saved {}'.format(path))

    if not args.no_show:
        print(
            'Displaying {} interactive plot windows. '
            'Close the windows to exit.'.format(len(figures)),
            flush=True,
        )
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
