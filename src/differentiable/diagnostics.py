"""SSPRK3 observations without changing its numerical operations.

Boundary exchange uses the actual numerical flux at each RK stage. Repair
increments carry the same RK weights as the states they modify. Raw-stage
admissibility is recorded even when an explicitly requested repair succeeds.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .euler_jax import GAMMA, euler_positivity_repair, get_euler_flux_jax, spds_jax
from .numerics_jax import lax_friedrichs_flux


def _observe(state, system, dt_dx, wave_speed):
    if system == 'euler':
        rho = state[:, 0]
        denom = jnp.where(rho != 0, rho, 1.0)
        pressure = (GAMMA - 1) * (state[:, 2] - 0.5 * state[:, 1] ** 2 / denom)
        finite = jnp.all(jnp.isfinite(state), axis=-1)
        invalid = (~finite) | (rho <= 0) | (pressure <= 0) | (~jnp.isfinite(pressure))
        max_speed = jnp.max(jnp.abs(spds_jax(state)))
    else:
        rho = state
        pressure = jnp.ones_like(state)  # omitted from scalar exports
        finite = jnp.isfinite(state)
        invalid = ~finite
        max_speed = jnp.max(jnp.abs(state)) if system == 'burgers' else wave_speed
    return dict(minimum_density=jnp.min(rho), maximum_state=jnp.max(jnp.abs(state)), minimum_pressure=jnp.min(pressure),
                invalid_rows=jnp.sum(invalid), nonfinite_rows=jnp.sum(~finite),
                max_courant=dt_dx * max_speed)


def diagnostic_rollout(initial, reconstruct, dt, dx, steps, system,
                       boundary='periodic', wave_speed=1.0, alpha=1.0,
                       repair_policy='none', repair_function=None):
    """Return final state and per-step raw-stage/repair/balance observations.

    There is no automatic guarded retry. historical-repair is an explicitly
    labeled diagnostic path, retaining the historical periodic neighbor rule.
    """
    if repair_policy not in ('none', 'historical-repair'):
        raise ValueError('Unknown repair policy')
    if repair_policy != 'none' and system != 'euler':
        raise ValueError('Euler repairs cannot be applied to scalar states')
    if repair_function is not None and repair_policy != 'historical-repair':
        raise ValueError('A supplied diagnostic repair requires an explicit repair policy')
    if steps < 1 or dt <= 0 or dx <= 0:
        raise ValueError('Positive dt, dx and step count are required')
    initial = jnp.asarray(initial, dtype=jnp.float64)
    components = 3 if system == 'euler' else 1
    if system == 'euler':
        flux = get_euler_flux_jax(reconstruct, boundary, return_boundary_flux=True)
    else:
        if boundary != 'periodic':
            raise ValueError('Scalar solver supports periodic boundaries only')
        physical = (lambda u: 0.5 * u ** 2) if system == 'burgers' else (lambda u: wave_speed * u)

        def flux(u):
            net = lax_friedrichs_flux(u, physical, alpha, reconstruct)
            zero = jnp.zeros((1,), dtype=u.dtype)
            return net, (zero, zero)

    def stage(v):
        observed = _observe(v, system, dt / dx, wave_speed)
        if repair_policy == 'historical-repair':
            after, info = (euler_positivity_repair(v, return_diagnostics=True)
                           if repair_function is None else repair_function(v))
            observed.update(
                repaired_rows=jnp.sum(info['changed_mask']),
                isolated_repairs=jnp.sum(info['isolated_mask']),
                fallback_repairs=jnp.sum(info['fallback_mask']))
            delta = jnp.sum(after - v, axis=0)
        else:
            after = v
            observed.update(repaired_rows=jnp.array(0), isolated_repairs=jnp.array(0),
                            fallback_repairs=jnp.array(0))
            delta = jnp.zeros((components,), dtype=v.dtype)
        return after, observed, delta

    def body(u, _):
        f0, (left0, right0) = flux(u)
        u1, d1, r1 = stage(u - f0 * dt / dx)
        f1, (left1, right1) = flux(u1)
        u2, d2, r2 = stage(0.75 * u + 0.25 * u1 - 0.25 * f1 * dt / dx)
        f2, (left2, right2) = flux(u2)
        u3, d3, r3 = stage((1.0 / 3.0) * u + (2.0 / 3.0) * u2 - (2.0 / 3.0) * f2 * dt / dx)
        row = {}
        for key in d1:
            if key.startswith('minimum'):
                row[key] = jnp.minimum(jnp.minimum(d1[key], d2[key]), d3[key])
            elif key in ('max_courant', 'maximum_state'):
                row[key] = jnp.maximum(jnp.maximum(d1[key], d2[key]), d3[key])
            else:
                row[key] = d1[key] + d2[key] + d3[key]
        row['boundary_exchange'] = dt * (
            (left0 - right0) / 6.0 + (left1 - right1) / 6.0
            + (2.0 / 3.0) * (left2 - right2))
        row['repair_exchange'] = dx * (r1 / 6.0 + (2.0 / 3.0) * r2 + r3)
        row['integral'] = dx * jnp.atleast_1d(jnp.sum(u3, axis=0))
        return u3, row

    final, history = jax.lax.scan(body, initial, xs=None, length=steps)
    final, history = np.asarray(final), jax.tree.map(np.asarray, history)
    first = {k: float(v) for k, v in _observe(initial, system, dt / dx, wave_speed).items()}
    initial_integral = dx * np.atleast_1d(np.sum(np.asarray(initial), axis=0))
    boundary_exchange = np.sum(history['boundary_exchange'], axis=0)
    repair_exchange = np.sum(history['repair_exchange'], axis=0)
    final_integral = dx * np.atleast_1d(np.sum(final, axis=0))
    finite = bool(np.isfinite(final).all())
    invalid_rows = int(np.sum(history['invalid_rows'])) + int(first['invalid_rows'])
    summary = dict(
        finite=finite, raw_stages_admissible=invalid_rows == 0,
        invalid_stage_rows=invalid_rows,
        nonfinite_stage_rows=int(np.sum(history['nonfinite_rows'])) + int(first['nonfinite_rows']),
        repaired_stage_rows=int(np.sum(history['repaired_rows'])),
        isolated_repair_events=int(np.sum(history['isolated_repairs'])),
        fallback_repair_events=int(np.sum(history['fallback_repairs'])),
        repair_policy=repair_policy,
        minimum_state=float(np.minimum(first['minimum_density'], np.min(history['minimum_density']))),
        maximum_absolute_stage_state=float(np.maximum(first['maximum_state'], np.max(history['maximum_state']))),
        max_courant=float(np.maximum(first['max_courant'], np.max(history['max_courant']))),
        initial_integral=initial_integral.tolist(), final_integral=final_integral.tolist(),
        integrated_boundary_exchange=boundary_exchange.tolist(),
        integrated_repair_exchange=repair_exchange.tolist(),
        conservation_residual=(final_integral - initial_integral - boundary_exchange - repair_exchange).tolist(),
        physical_balance_error=(final_integral - initial_integral - boundary_exchange).tolist(),
    )
    if system == 'euler':
        summary['minimum_density'] = summary.pop('minimum_state')
        summary['minimum_pressure'] = float(np.minimum(first['minimum_pressure'], np.min(history['minimum_pressure'])))
    summary['valid_for_accuracy_claim'] = finite and invalid_rows == 0 and summary['repaired_stage_rows'] == 0
    return final, summary, history
