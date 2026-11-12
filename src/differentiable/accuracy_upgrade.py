"""Opt-in finite-volume Euler experiment. Historical arithmetic is untouched.

Cell averages -> one Roe basis per face -> characteristic state reconstruction
-> Rusanov flux. Only admissible cell averages may enter the reconstruction.
The optional safeguard scales face states (never averages) and blends shared
fluxes with global LF. Decisions/coefficients are stop-gradient; smooth inactive
regions retain ordinary autodiff. No entropy, TVD or general accuracy guarantee.
"""
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from .euler_jax import (GAMMA, flux_jax, spds_jax, roe_eigenbasis_jax,
                        project_to_characteristic_jax,
                        project_to_conservative_jax, get_euler_flux_jax)


@dataclass(frozen=True)
class Admissibility:
    density_floor: float = 1e-12
    pressure_floor: float = 1e-12
    bisections: int = 40
    max_retries: int = 8
    low_speed_margin: float = 1.05
    max_low_cfl: float = 0.9

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.density_floor, self.pressure_floor)):
            raise ValueError('Positive finite physical floors required')
        if self.bisections < 1 or self.max_retries < 0:
            raise ValueError('Invalid bounded iteration count')
        if not math.isfinite(self.low_speed_margin) or self.low_speed_margin <= 1:
            raise ValueError('Global LF speed margin must exceed one')
        if not math.isfinite(self.max_low_cfl) or not 0 < self.max_low_cfl < 1:
            raise ValueError('Global LF CFL must be strictly below one')


def pressure(u):
    # A guarded denominator is used for diagnostics on rejected states only.
    # Valid states use the exact physical pressure, without abs/floors on U.
    rho = jnp.where(u[..., 0] > 0, u[..., 0], 1.)
    return (GAMMA-1.)*(u[..., 2]-.5*u[..., 1]**2/rho)


def admissible(u, config=Admissibility()):
    return (jnp.all(jnp.isfinite(u), axis=-1) &
            (u[..., 0] >= config.density_floor) &
            (pressure(u) >= config.pressure_floor))


def padded_states(u, boundary):
    if u.ndim != 2 or u.shape[1] != 3 or len(u) < 5:
        raise ValueError('Expected at least five Euler cell averages')
    if boundary == 'periodic':
        return jnp.concatenate((u[-3:], u, u[:3]))
    if boundary == 'transmissive':
        return jnp.concatenate((jnp.repeat(u[:1], 3, axis=0), u,
                                jnp.repeat(u[-1:], 3, axis=0)))
    if boundary == 'reflective-wall':
        sign = jnp.array([1., -1., 1.], dtype=u.dtype)
        return jnp.concatenate((u[:3][::-1]*sign, u, u[-3:][::-1]*sign))
    raise ValueError('Unsupported experimental Euler boundary: '+boundary)


def state_stencils(u, boundary='transmissive', return_context=False):
    """Canonical inference AND warm-up inputs; (n+1,3,5), both directions.

    Face f separates cells f-1/f. The right stencil is offset by one and
    mirrored; both use the SAME face-local Roe basis. Boundary faces included.
    """
    padded = padded_states(u, boundary)
    n = len(u)
    # Reuse established Roe implementation, once per call. Six harmless ghost
    # bases are extra work; no second decomposition for the mirrored direction.
    left, right = roe_eigenbasis_jax(padded, 'transmissive')
    left, right = left[2:n+3], right[2:n+3]
    centers = jnp.arange(n+1)[:, None]+2
    offsets = jnp.arange(-2, 3)[None, :]
    plus = jnp.transpose(padded[centers+offsets], (0, 2, 1))
    minus = jnp.transpose(padded[centers+1-offsets], (0, 2, 1))
    plus = project_to_characteristic_jax(plus, left)
    minus = project_to_characteristic_jax(minus, left)
    if return_context:
        return plus, minus, right, padded[2:n+3], padded[3:n+4]
    return plus, minus


def _boundary_faces(faces, boundary):
    if boundary == 'periodic':
        # Physically identical face: enforce exact shared floating-point value.
        return faces.at[-1].set(faces[0])
    if boundary == 'reflective-wall':
        # Impermeable adiabatic wall: momentum flux is wall pressure.
        faces = faces.at[0, 0].set(0.).at[0, 2].set(0.)
        return faces.at[-1, 0].set(0.).at[-1, 2].set(0.)
    return faces


def _segment_parameter(base, target, valid, iterations):
    """Largest admissible common segment coefficient, bracketed from zero.

    Convexity of {rho>=eps,p>=eps} makes its intersection with this segment
    an interval. Base must be admissible. Reject nonfinite targets upstream.
    """
    def bisect(_):
        def body(i, bracket):
            lo, hi = bracket
            mid = .5*(lo+hi)
            ok = jnp.all(valid(base+mid*(target-base)))
            return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)
        lo, _ = jax.lax.fori_loop(0, iterations, body, (jnp.array(0.), jnp.array(1.)))
        return lo*(1.-1e-10)
    return jax.lax.stop_gradient(jax.lax.cond(jnp.all(valid(target)),
                                             lambda _: jnp.array(1.), bisect, None))


def state_faces(u, reconstruct, boundary='transmissive', face_guard=False,
                config=Admissibility()):
    """High-order shared fluxes and face diagnostics, on admissible averages.

    If raw reconstruction is nonfinite, guarded mode uses cell faces and
    explicitly flags the event. Finite nonphysical reconstructions are scaled
    towards their own cell average before physical flux/sound-speed evaluation.
    This changes fluxes only, and is separately counted from update limiting.
    """
    plus, minus, basis, cell_l, cell_r = state_stencils(u, boundary, True)
    ul = project_to_conservative_jax(reconstruct(plus), basis)
    ur = project_to_conservative_jax(reconstruct(minus), basis)
    raw = jnp.stack((ul, ur))
    raw_valid = jnp.all(admissible(raw, config))
    finite = jnp.all(jnp.isfinite(raw))
    coefficient = jnp.array(1.)
    if face_guard:
        base = jnp.stack((cell_l, cell_r))
        # Sanitize before any nonlinear arithmetic, with explicit diagnostics.
        target = jax.lax.cond(finite, lambda _: raw, lambda _: base, None)
        coefficient = _segment_parameter(base, target,
                         lambda v: admissible(v, config), config.bisections)
        coefficient = jnp.where(finite, coefficient, 0.)
        ul, ur = base+coefficient*(target-base)
    speed = jnp.maximum(jnp.max(jnp.abs(spds_jax(ul)), axis=-1),
                        jnp.max(jnp.abs(spds_jax(ur)), axis=-1))
    faces = .5*(flux_jax(ul)+flux_jax(ur)-speed[:, None]*(ur-ul))
    return _boundary_faces(faces, boundary), dict(face_theta=coefficient,
                 raw_faces_valid=raw_valid, raw_faces_finite=finite)


def low_faces(u, boundary='transmissive', config=Admissibility()):
    """Global LF flux. alpha > max_i(|v_i|+c_i); dt*alpha/dx < 1.

    The FE update is a convex combination of U_i and neighbor LF-split
    states U +/- F(U)/alpha. Ideal-gas positive states remain admissible in
    exact arithmetic under this CFL (floors additionally checked in code).
    Ghost states obey the same bound. Wall reflection makes zero mass/energy
    flux automatically; the explicit wall closure leaves it unchanged.
    """
    pad = padded_states(u, boundary)
    ul, ur = pad[2:len(u)+3], pad[3:len(u)+4]
    alpha = config.low_speed_margin*jnp.max(jnp.abs(spds_jax(u)))
    faces = .5*(flux_jax(ul)+flux_jax(ur)-alpha*(ur-ul))
    return _boundary_faces(faces, boundary), alpha


def flux_variant(reconstruct, boundary='transmissive', variant='historical',
                 return_boundary_flux=False, face_guard=False, config=Admissibility()):
    if variant == 'historical':
        return get_euler_flux_jax(reconstruct, boundary, return_boundary_flux)
    if variant != 'fv_state_flux':
        raise ValueError('Unknown Euler flux variant: '+variant)
    def flux(u):
        faces, _ = state_faces(u, reconstruct, boundary, face_guard, config)
        net = jnp.diff(faces, axis=0)
        return (net, (faces[0], faces[-1])) if return_boundary_flux else net
    return flux


def safeguarded_step(u, dt, dx, reconstruct, boundary='transmissive',
                     config=Admissibility()):
    """One SSPRK3 attempt; failure returns the unmodified input, success=False.

    Every FE-like update checks low-order admissibility and CFL, then chooses
    ONE global shared-flux correction coefficient. No average is clipped.
    Boundary exchange uses SSPRK weights (1/6,1/6,2/3). Retry wrapper below
    retries the ENTIRE step. Coefficients/decisions have frozen gradients.
    """
    def fe(v):
        lo, alpha = low_faces(v, boundary, config)
        hi, face_diag = state_faces(v, reconstruct, boundary, True, config)
        low = v-dt/dx*jnp.diff(lo, axis=0)
        high = v-dt/dx*jnp.diff(hi, axis=0)
        low_ok = jnp.all(admissible(low, config)) & (dt*alpha/dx <= config.max_low_cfl)
        theta = jax.lax.cond(low_ok, lambda _: _segment_parameter(low, high,
                    lambda z: admissible(z, config), config.bisections),
                    lambda _: jnp.array(0.), None)
        faces = lo+theta*(hi-lo)
        result = v-dt/dx*jnp.diff(faces, axis=0)
        ok = low_ok & jnp.all(admissible(result, config))
        # A failed FE-like stage remains visibly failed; return v solely to
        # avoid evaluating eigenvectors on an inadmissible state downstream.
        result = jax.lax.cond(ok, lambda _: result, lambda _: v, None)
        return result, dict(ok=ok, low_ok=low_ok, theta=theta,
                    raw_invalid=jnp.sum(~admissible(high, config)),
                    min_density=jnp.min(result[:, 0]), min_pressure=jnp.min(pressure(result)),
                    exchange=dt*(faces[-1]-faces[0]),
                    cfl=dt*alpha/dx, **face_diag)

    def attempt(_):
        a, da = fe(u)
        b_fe, db = fe(a)
        b = .75*u+.25*b_fe
        c_fe, dc = fe(b)
        result = u/3.+(2./3.)*c_fe
        diagnostics = jax.tree.map(lambda *x: jnp.stack(x), da, db, dc)
        success = jnp.all(diagnostics['ok']) & jnp.all(admissible(result, config))
        exchange = (da['exchange']+db['exchange'])/6.+2.*dc['exchange']/3.
        return jax.lax.cond(success, lambda _: result, lambda _: u, None), dict(
            success=success, exchange=exchange, stages=diagnostics)

    def invalid(_):
        # No Roe/flux evaluation on invalid initial states, including under JIT.
        z = jnp.zeros(3)
        stage = dict(ok=jnp.zeros(3, dtype=bool), low_ok=jnp.zeros(3, dtype=bool),
                     theta=z, raw_invalid=jnp.ones(3, dtype=int), min_density=z,
                     min_pressure=z, exchange=jnp.zeros((3, 3)), cfl=z, face_theta=z,
                     raw_faces_valid=jnp.zeros(3, dtype=bool), raw_faces_finite=jnp.zeros(3, dtype=bool))
        return u, dict(success=jnp.array(False), exchange=z, stages=stage)
    return jax.lax.cond(jnp.all(admissible(u, config)) & (dt > 0) & (dx > 0), attempt, invalid, None)


def retry_step(u, dt, dx, reconstruct, boundary='transmissive', config=Admissibility()):
    """Bounded halving; returns accepted dt, retry count and explicit status.

    On exhausted retries, result is original u, accepted_dt=0, success=False.
    Callers MUST check success before advancing physical time.
    """
    result, diag = safeguarded_step(u, dt, dx, reconstruct, boundary, config)
    invalid_rows = jnp.sum(diag['stages']['raw_invalid'])
    low_rejections = jnp.sum(~diag['stages']['low_ok'])
    def body(i, carry):
        value, info, used, retries, invalid, rejected_low = carry
        def retry(_):
            next_dt = used*.5
            value, info = safeguarded_step(u, next_dt, dx, reconstruct, boundary, config)
            return (value, info, next_dt, retries+1,
                    invalid+jnp.sum(info['stages']['raw_invalid']),
                    rejected_low+jnp.sum(~info['stages']['low_ok']))
        return jax.lax.cond(info['success'], lambda _: carry, retry, None)
    value, diag, used, retries, invalid, rejected_low = jax.lax.fori_loop(
        0, config.max_retries, body,
        (result, diag, jnp.asarray(dt), jnp.array(0), invalid_rows, low_rejections))
    return value, dict(**diag, accepted_dt=jnp.where(diag['success'], used, 0.),
                      retries=retries, attempt_raw_invalid_rows=invalid,
                      low_order_rejected_stages=rejected_low)
