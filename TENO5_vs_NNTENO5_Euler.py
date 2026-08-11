# -*- coding: utf-8 -*-
"""Compare deterministic TENO5 and NN-TENO5 on Euler benchmarks."""

import argparse
import json
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
from keras.models import load_model

from src.analysis.euler_regression import (
    exact_sod_solution,
    interpolate_reference,
    primitive_variables,
    regression_metrics,
    run_euler_benchmark,
)
from src.core.eulerEquations import GAMMA
from src.initial_conditions.InitialConditions import shuOsher, sod
from src.schemes import (
    NN_TENO5_Euler,
    TENO5euler,
    WENO7euler,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare TENO5 and NN-TENO5 on Euler benchmarks.'
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
        default='models/nn_teno5_euler_comparison.json',
    )
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
    parser.add_argument(
        '--fidelity-factor',
        type=float,
        default=1.02,
    )
    return parser.parse_args()


def validate_args(args):
    if args.cells < 7 or args.reference_cells < 7:
        raise ValueError('Euler cell counts must be at least seven.')
    if not 0.0 < args.cfl <= 1.0:
        raise ValueError('--cfl must belong to (0, 1].')
    if args.fidelity_factor < 1.0:
        raise ValueError('--fidelity-factor must be at least one.')


def uniform_state(density, velocity, pressure):
    energy = (
        pressure / (GAMMA - 1.0)
        + 0.5 * density * velocity ** 2
    )

    def initial_condition(x):
        state = np.empty(
            (len(x), 3),
            dtype=float,
        )
        state[:, 0] = density
        state[:, 1] = density * velocity
        state[:, 2] = energy
        return state

    return initial_condition


def timed_euler_run(
    scheme_builder,
    initial_condition,
    cells,
    length,
    final_time,
    cfl,
):
    started = time.perf_counter()
    x, solution, initial = run_euler_benchmark(
        scheme_builder,
        initial_condition,
        cells,
        length,
        final_time,
        cfl=cfl,
    )
    runtime = time.perf_counter() - started
    return x, solution, initial, runtime


def state_checks(solution):
    density, _, pressure = primitive_variables(solution)
    return {
        'finite': bool(np.isfinite(solution).all()),
        'minimum_density': float(np.min(density)),
        'minimum_pressure': float(np.min(pressure)),
        'positive_density': bool(np.all(density > 0.0)),
        'positive_pressure': bool(np.all(pressure > 0.0)),
    }


def uniform_metrics(solution, initial, length, runtime):
    dx = length / len(solution)
    difference = solution - initial
    result = state_checks(solution)
    result.update({
        'l1_error': np.mean(
            np.abs(difference),
            axis=0,
        ).tolist(),
        'linf_error': np.max(
            np.abs(difference),
            axis=0,
        ).tolist(),
        'conservation_error': np.abs(
            dx * np.sum(solution - initial, axis=0)
        ).tolist(),
        'runtime_seconds': float(runtime),
    })
    return result


def reference_metrics(
    x,
    solution,
    initial,
    final_time,
    reference,
    runtime,
):
    result = regression_metrics(
        x,
        solution,
        initial,
        final_time,
        reference,
    )
    result.update(state_checks(solution))
    result['runtime_seconds'] = float(runtime)
    return result


def run_uniform_case(
    name,
    initial_condition,
    deterministic_builder,
    neural_builder,
    cells,
    cfl,
):
    length = 1.0
    final_time = 0.2
    _, deterministic, initial, deterministic_runtime = timed_euler_run(
        deterministic_builder,
        initial_condition,
        cells,
        length,
        final_time,
        cfl,
    )
    _, neural, _, neural_runtime = timed_euler_run(
        neural_builder,
        initial_condition,
        cells,
        length,
        final_time,
        cfl,
    )
    return name, {
        'cells': cells,
        'final_time': final_time,
        'teno5': uniform_metrics(
            deterministic,
            initial,
            length,
            deterministic_runtime,
        ),
        'nn_teno5': uniform_metrics(
            neural,
            initial,
            length,
            neural_runtime,
        ),
        'max_teacher_difference': float(np.max(
            np.abs(neural - deterministic)
        )),
    }


def run_reference_case(
    name,
    initial_condition,
    reference_x,
    reference,
    length,
    final_time,
    deterministic_builder,
    neural_builder,
    cells,
    cfl,
    fidelity_factor,
):
    x, deterministic, initial, deterministic_runtime = timed_euler_run(
        deterministic_builder,
        initial_condition,
        cells,
        length,
        final_time,
        cfl,
    )
    _, neural, _, neural_runtime = timed_euler_run(
        neural_builder,
        initial_condition,
        cells,
        length,
        final_time,
        cfl,
    )
    target = (
        reference
        if reference_x is None
        else interpolate_reference(
            reference_x,
            reference,
            x,
        )
    )
    deterministic_metrics = reference_metrics(
        x,
        deterministic,
        initial,
        final_time,
        target,
        deterministic_runtime,
    )
    neural_metrics = reference_metrics(
        x,
        neural,
        initial,
        final_time,
        target,
        neural_runtime,
    )
    fidelity_limit = (
        fidelity_factor
        * deterministic_metrics['density_l1']
    )
    return name, {
        'cells': cells,
        'final_time': final_time,
        'teno5': deterministic_metrics,
        'nn_teno5': neural_metrics,
        'max_teacher_difference': float(np.max(
            np.abs(neural - deterministic)
        )),
        'teacher_fidelity': {
            'metric': 'density_l1',
            'factor': fidelity_factor,
            'limit': float(fidelity_limit),
            'passed': bool(
                neural_metrics['density_l1']
                <= fidelity_limit
            ),
        },
    }


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
    deterministic_builder = TENO5euler

    def neural_builder():
        return NN_TENO5_Euler(
            model,
            threshold=thresholds,
        )

    results = {
        'model_path': os.path.abspath(args.model_path),
        'metadata_path': os.path.abspath(args.metadata_path),
        'calibrated_thresholds': thresholds,
        'cells': args.cells,
        'reference_cells': args.reference_cells,
        'cfl': args.cfl,
        'fidelity_factor': args.fidelity_factor,
        'reference': {
            'sod': 'exact Riemann solution',
            'shu_osher': 'WENO7 at {} cells'.format(
                args.reference_cells
            ),
        },
        'problems': {},
    }

    uniform_cases = (
        (
            'constant_stationary',
            uniform_state(1.0, 0.0, 1.0),
        ),
        (
            'uniform_moving',
            uniform_state(1.0, 0.5, 1.0),
        ),
    )

    for name, initial_condition in uniform_cases:
        print('Running {}...'.format(name), flush=True)
        case_name, result = run_uniform_case(
            name,
            initial_condition,
            deterministic_builder,
            neural_builder,
            args.cells,
            args.cfl,
        )
        results['problems'][case_name] = result

    print('Running Sod...', flush=True)
    sod_x = (
        np.arange(args.cells, dtype=float)
        / args.cells
    )
    case_name, result = run_reference_case(
        'sod',
        sod(),
        None,
        exact_sod_solution(sod_x, 0.2),
        1.0,
        0.2,
        deterministic_builder,
        neural_builder,
        args.cells,
        args.cfl,
        args.fidelity_factor,
    )
    results['problems'][case_name] = result

    print('Building Shu-Osher reference...', flush=True)
    shu_reference_x, shu_reference, _ = run_euler_benchmark(
        WENO7euler,
        shuOsher(),
        args.reference_cells,
        10.0,
        1.8,
        cfl=args.cfl,
    )
    print('Running Shu-Osher...', flush=True)
    case_name, result = run_reference_case(
        'shu_osher',
        shuOsher(),
        shu_reference_x,
        shu_reference,
        10.0,
        1.8,
        deterministic_builder,
        neural_builder,
        args.cells,
        args.cfl,
        args.fidelity_factor,
    )
    results['problems'][case_name] = result

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

    for name, result in results['problems'].items():
        checks = result['nn_teno5']
        print(
            '{}: finite={}, rho>0={}, p>0={}, max teacher diff={:.6e}'
            .format(
                name,
                checks['finite'],
                checks['positive_density'],
                checks['positive_pressure'],
                result['max_teacher_difference'],
            )
        )
        if 'teacher_fidelity' in result:
            print(
                '  density fidelity={}'.format(
                    result['teacher_fidelity']['passed']
                )
            )

    print(
        '\nComparison written to {}'.format(
            os.path.abspath(args.output_path)
        )
    )


if __name__ == '__main__':
    main()
