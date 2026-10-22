"""Versioned problem definitions and analytic references, without training imports."""
import copy
import json
import math
from pathlib import Path

import numpy as np

from src.analysis.euler_regression import exact_sod_solution
from src.core.eulerEquations import spds
from src.initial_conditions.InitialConditions import shuOsher, sod

MANIFEST_PATH = Path(__file__).with_name('benchmarks.json')


def read_manifest(path=MANIFEST_PATH):
    def reject_constant(value):
        raise ValueError(f'Non-finite JSON constant {value}')
    manifest = json.loads(Path(path).read_text(), parse_constant=reject_constant)
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported benchmark manifest schema')
    identifiers = []
    for suite, cases in manifest['suites'].items():
        for case in cases:
            validate_case(case)
            identifiers.append(case['id'])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Case identifiers must be unique across suites')
    if len(manifest['methods']) != 4 or set(manifest['methods']) != {'teno5', 'neural-only', 'fixed-blend', 'learned-blend'}:
        raise ValueError('Declare each of the four matched ablation methods exactly once')
    if not math.isfinite(manifest['fixed_lambda']) or not 0 <= manifest['fixed_lambda'] <= 1:
        raise ValueError('fixed_lambda must be in [0,1]')
    study = manifest['reference_study']
    if len(study['resolutions']) < 3 or study['resolutions'] != sorted(set(study['resolutions'])):
        raise ValueError('At least three strictly increasing reference resolutions are required')
    if any(not isinstance(n, int) or n < 8 for n in study['resolutions']):
        raise ValueError('Reference resolutions must be positive integers >= 8')
    if study['temporal_refinements'] != [1, 2, 4]:
        raise ValueError('Temporal convergence uses the predeclared dt, dt/2, dt/4 sequence')
    if not 0 < study['reference_uncertainty_fraction'] < 1:
        raise ValueError('Reference uncertainty fraction must be between 0 and 1')
    return manifest


def validate_case(case):
    allowed = {'id', 'problem', 'cells', 'length', 'final_time', 'cfl', 'boundary',
               'sampling', 'initial', 'wave_speed', 'alpha', 'wave_speed_safety',
               'reference_cells'}
    unknown = set(case) - allowed
    if unknown:
        raise ValueError(f'Unknown case fields: {sorted(unknown)}')
    if case['problem'] not in ('advection', 'burgers', 'sod', 'shu_osher'):
        raise ValueError('Unsupported problem')
    if not isinstance(case['cells'], int) or case['cells'] < 8:
        raise ValueError('cells must be an integer >= 8')
    for name in ('length', 'final_time', 'cfl'):
        if not math.isfinite(case[name]) or case[name] <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if case['sampling'] != 'point-i-dx':
        raise ValueError('Historical cases use point-i-dx sampling; use a separate convergence study for cell averages')
    scalar = case['problem'] in ('advection', 'burgers')
    if case['boundary'] != ('periodic' if scalar else 'transmissive'):
        raise ValueError('Unsupported boundary for this problem')
    if scalar:
        if case['alpha'] <= 0 or not math.isfinite(case['alpha']):
            raise ValueError('alpha must be positive and finite')
        profile = case['initial']
        if profile['type'] not in ('square', 'fourier'):
            raise ValueError('Unsupported scalar initial profile')
        expected = {'type', 'left', 'right', 'amplitude', 'offset'} if profile['type'] == 'square' else {'type', 'offset', 'modes'}
        if set(profile) != expected:
            raise ValueError('Unexpected scalar profile fields')
        if profile['type'] == 'square':
            if not all(math.isfinite(profile[k]) for k in ('left', 'right', 'amplitude', 'offset')) or not 0 <= profile['left'] < profile['right'] <= case['length']:
                raise ValueError('Square interval and amplitudes must be finite and inside the periodic domain')
        if profile['type'] == 'fourier':
            if not math.isfinite(profile['offset']) or not profile['modes']:
                raise ValueError('Fourier profile requires a finite offset and at least one mode')
            for amplitude, frequency, phase in profile['modes']:
                if not all(math.isfinite(v) for v in (amplitude, frequency, phase)) or frequency < 1 or frequency != int(frequency):
                    raise ValueError('Fourier modes must be finite and have positive integer periodic frequencies')
        if case['problem'] == 'burgers':
            if profile['type'] != 'fourier':
                raise ValueError('Analytic Burgers reference requires a smooth periodic Fourier profile')
            bound = sum(abs(a * 2 * np.pi * k / case['length']) for a, k, _ in profile['modes'])
            if case['final_time'] * bound >= 1:
                raise ValueError('Burgers reference is not certified pre-shock by its derivative bound')
            if abs(profile['offset']) + sum(abs(a) for a, _, _ in profile['modes']) > case['alpha']:
                raise ValueError('LF alpha is below the initial Burgers speed bound')
        elif not math.isfinite(case['wave_speed']) or case['wave_speed'] <= 0 or case['wave_speed'] > case['alpha']:
            raise ValueError('Positive advection speed must not exceed LF alpha')
    else:
        if not math.isfinite(case['wave_speed_safety']) or case['wave_speed_safety'] <= 0:
            raise ValueError('wave_speed_safety must be positive and finite')
        expected = {'discontinuity'} if case['problem'] == 'sod' else {'amplitude', 'frequency', 'phase', 'discontinuity'}
        if set(case['initial']) != expected or not all(math.isfinite(v) for v in case['initial'].values()):
            raise ValueError('Unexpected Euler initial-condition fields')
        if not 0 < case['initial']['discontinuity'] < case['length']:
            raise ValueError('Discontinuity must be inside the domain')
        if case['problem'] == 'shu_osher':
            if not 0 <= case['initial']['amplitude'] < 1 or case['initial']['frequency'] <= 0:
                raise ValueError('Shu–Osher requires positive density and frequency')
            if not isinstance(case['reference_cells'], int) or case['reference_cells'] <= case['cells']:
                raise ValueError('Numerical reference must be finer than the evaluation grid')


def grid(case):
    return np.arange(case['cells'], dtype=float) * case['length'] / case['cells']


def scalar_profile(case, x):
    profile = case['initial']
    if profile['type'] == 'square':
        periodic = np.mod(x, case['length'])
        return profile['offset'] + profile['amplitude'] * ((periodic >= profile['left']) & (periodic < profile['right']))
    result = np.full_like(x, profile['offset'], dtype=float)
    for amplitude, frequency, phase in profile['modes']:
        result += amplitude * np.sin(2.0 * np.pi * frequency * x / case['length'] + phase)
    return result


def initial_state(case, x=None):
    x = grid(case) if x is None else np.asarray(x)
    if case['problem'] in ('advection', 'burgers'):
        return scalar_profile(case, x)
    settings = case['initial']
    if case['problem'] == 'sod':
        if settings['discontinuity'] == 0.5:
            return sod()(x)
        left = x < settings['discontinuity']
        return np.column_stack((np.where(left, 1.0, 0.125), np.zeros_like(x),
                                np.where(left, 1.0 / (1.4 - 1), 0.1 / (1.4 - 1))))
    if settings == dict(amplitude=0.2, frequency=5.0, phase=0.0, discontinuity=1.0):
        return shuOsher()(x)
    left = x < settings['discontinuity']
    rho = np.where(left, 3.857143, 1 + settings['amplitude'] * np.sin(settings['frequency'] * x + settings['phase']))
    velocity = np.where(left, 2.629369, 0.0)
    pressure = np.where(left, 10.3333, 1.0)
    return np.column_stack((rho, rho * velocity, pressure / (1.4 - 1) + 0.5 * rho * velocity ** 2))


def analytic_reference(case, x=None):
    validate_case(case)
    x = grid(case) if x is None else np.asarray(x)
    t = case['final_time']
    if case['problem'] == 'advection':
        return scalar_profile(case, x - case['wave_speed'] * t)
    if case['problem'] == 'sod':
        return exact_sod_solution(x, t, discontinuity=case['initial']['discontinuity'])
    if case['problem'] != 'burgers':
        raise ValueError('Shu–Osher has no analytic reference')
    length = case['length']
    mapped_start = t * scalar_profile(case, np.array([0.0]))[0]
    target = np.mod(x, length)
    # This reduces exactly to the historical lift on the canonical case.
    target = target + length * np.ceil((mapped_start - target) / length)
    lower, upper = np.zeros_like(target), np.full_like(target, length)
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        mapped = midpoint + t * scalar_profile(case, midpoint)
        right = mapped < target
        lower = np.where(right, midpoint, lower)
        upper = np.where(right, upper, midpoint)
    foot = 0.5 * (lower + upper)
    if np.max(np.abs(foot + t * scalar_profile(case, foot) - target)) > 1e-12:
        raise ValueError('Characteristic inversion did not converge')
    return scalar_profile(case, foot)


def time_grid(case, initial=None, refinement=1):
    initial = initial_state(case) if initial is None else initial
    dx = case['length'] / case['cells']
    if case['problem'] in ('sod', 'shu_osher'):
        speed = float(np.max(np.abs(spds(initial))) * case['wave_speed_safety'])
    else:
        speed = case['alpha']
    steps = max(1, int(np.ceil(case['final_time'] * speed / (case['cfl'] * dx)))) * refinement
    return dict(dx=dx, dt=case['final_time'] / steps, steps=steps,
                initial_speed_bound=speed, temporal_refinement=refinement)


def reference_case(case, cells):
    result = copy.deepcopy(case)
    result['cells'] = cells
    # reference_cells describes the evaluation case's reference only, not
    # another nested reference for this standalone deterministic rollout.
    result['reference_cells'] = 2 * cells
    return result
