"""Cell-average spatial and time-controlled smooth convergence studies.

These are separate scientific studies, not replacements for the historical
point-sampled canonical benchmarks. No fifth-order guarantee is assumed for
the neural candidate masks or their blends.
"""
import math

import jax.numpy as jnp
import numpy as np

from .artifacts import write_json
from .diagnostics import diagnostic_rollout
from .euler_jax import get_euler_flux_jax
from .evaluation import errors
from .numerics_jax import lax_friedrichs_flux
from .reconstruction import make_reconstruction

SETTINGS = dict(spatial_resolutions=[32, 64, 128, 256],
                rollout_resolutions=[32, 64, 128], final_time=0.1,
                dt_coefficient=0.2, dt_power=5/3, quadrature_order=12,
                scalar_speed=1.0, entropy_velocity=0.7, entropy_pressure=1.0,
                profile='exp(sin(2*pi*x)); Euler rho=0.5+0.5*profile',
                sampling='finite-volume cell averages on centers (i+0.5)*dx')


def profile(x):
    return np.exp(np.sin(2 * np.pi * x))


def averaged_state(n, system, time=0.0):
    nodes, weights = np.polynomial.legendre.leggauss(SETTINGS['quadrature_order'])
    dx = 1/n
    centers = (np.arange(n) + .5) * dx
    speed = SETTINGS['entropy_velocity'] if system == 'euler' else SETTINGS['scalar_speed']
    points = centers[:, None] + .5*dx*nodes[None, :] - speed*time
    average = .5*np.sum(weights[None, :]*profile(points), axis=1)
    if system != 'euler':
        return average
    rho = .5 + .5*average
    return np.column_stack((rho, speed*rho, SETTINGS['entropy_pressure']/(1.4-1) + .5*speed**2*rho))


def spatial_error(n, system, reconstruct):
    initial = averaged_state(n, system)
    dx = 1/n
    faces = np.arange(n+1)*dx
    derivative = (profile(faces[1:])-profile(faces[:-1]))/dx
    if system == 'euler':
        speed = SETTINGS['entropy_velocity']
        exact = np.column_stack((.5*speed*derivative, .5*speed**2*derivative, .25*speed**3*derivative))
        numerical = np.asarray(get_euler_flux_jax(reconstruct, 'periodic')(jnp.asarray(initial)))/dx
    else:
        speed = SETTINGS['scalar_speed']
        exact = speed*derivative
        numerical = np.asarray(lax_friedrichs_flux(jnp.asarray(initial), lambda u: speed*u, speed, reconstruct))/dx
    return errors(numerical, exact)


def _orders(rows):
    for previous, row in zip(rows, rows[1:]):
        row['observed_l1_order'] = math.log(previous['l1']/row['l1'])/math.log(row['cells']/previous['cells']) if previous['l1'] > 0 and row['l1'] > 0 else None


def run_smooth_study(models, methods, fixed_lambda, output):
    result = dict(settings=SETTINGS, spatial={}, rollout={})
    for system in ('advection', 'euler'):
        result['spatial'][system], result['rollout'][system] = {}, {}
        model = models['euler' if system == 'euler' else 'scalar']
        for method in methods:
            reconstruct = make_reconstruction(model, method, fixed_lambda)
            rows = [dict(cells=n, **spatial_error(n, system, reconstruct)) for n in SETTINGS['spatial_resolutions']]
            _orders(rows)
            result['spatial'][system][method] = rows
            rollout_rows = []
            for n in SETTINGS['rollout_resolutions']:
                dx = 1/n
                speed_bound = 2.5 if system == 'euler' else SETTINGS['scalar_speed']
                step_count = math.ceil(SETTINGS['final_time']*speed_bound/(SETTINGS['dt_coefficient']*dx**SETTINGS['dt_power']))
                dt = SETTINGS['final_time']/step_count
                final, diagnostics, history = diagnostic_rollout(
                    averaged_state(n, system), reconstruct, dt, dx, step_count,
                    system, 'periodic', wave_speed=SETTINGS['scalar_speed'], alpha=SETTINGS['scalar_speed'])
                row = dict(cells=n, dt=dt, steps=step_count, **errors(final, averaged_state(n, system, SETTINGS['final_time'])), diagnostics=diagnostics)
                rollout_rows.append(row)
                np.savez_compressed(output/f'smooth-{system}-{method}-{n}.npz', initial=averaged_state(n,system), state=final,
                                    reference=averaged_state(n,system,SETTINGS['final_time']),
                                    **{'history_'+k:v for k,v in history.items()})
            _orders(rollout_rows)
            result['rollout'][system][method] = rollout_rows
            print(f"Smooth {system}/{method}: spatial order={rows[-1]['observed_l1_order']:.4f}; rollout order={rollout_rows[-1]['observed_l1_order']:.4f}", flush=True)
    _plot(result, output)
    return result


def _plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2,2, figsize=(11,8))
    for column, study in enumerate(('spatial','rollout')):
        for i, system in enumerate(('advection','euler')):
            for method, rows in result[study][system].items():
                axes[i,column].loglog([r['cells'] for r in rows], [r['l1'] for r in rows], 'o-', label=method)
            axes[i,column].set(xlabel='cells', ylabel='mean state L1 error', title=f'{system}: {study}')
            axes[i,column].legend(fontsize=8)
            axes[i,column].grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output/'smooth_convergence.png',dpi=160)
    figure.savefig(output/'smooth_convergence.pdf')
    plt.close(figure)
