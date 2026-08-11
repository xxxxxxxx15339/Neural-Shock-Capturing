import numpy as np

from ..core.eulerEquations import GAMMA, flux, getEulerFlux, spds
from ..core.SimulationClasses import eulerSimulation
from ..core.TimeSteppingMethods import SSPRK3


def primitive_variables(state):
    density = state[:,0]
    velocity = state[:,1]/density
    pressure = (GAMMA-1)*(state[:,2]-0.5*density*velocity**2)
    return density, velocity, pressure


def run_euler_benchmark(scheme_builder, initial_condition, cells, length,
                        final_time, cfl=0.35):
    x = np.arange(cells, dtype=float)*length/cells
    initial = initial_condition(x)
    simulation = eulerSimulation(
        cells, 2, length, final_time, SSPRK3(),
        getEulerFlux(scheme_builder()), initial_condition, 3,
        max_cfl=cfl,
        wave_speed_function=lambda state: np.max(np.abs(spds(state))),
    )
    solution = simulation.runEuler()[:,-1,:]
    if simulation.max_observed_cfl > cfl+1e-12:
        raise AssertionError('Adaptive Euler CFL exceeded its requested limit.')
    return x, solution, initial


def _pressure_function(pressure, density, initial_pressure):
    sound_speed = np.sqrt(GAMMA*initial_pressure/density)
    if pressure > initial_pressure:
        a = 2/((GAMMA+1)*density)
        b = (GAMMA-1)/(GAMMA+1)*initial_pressure
        root = np.sqrt(a/(pressure+b))
        value = (pressure-initial_pressure)*root
        derivative = root*(1-0.5*(pressure-initial_pressure)/(pressure+b))
    else:
        exponent = (GAMMA-1)/(2*GAMMA)
        ratio = pressure/initial_pressure
        value = 2*sound_speed/(GAMMA-1)*(ratio**exponent-1)
        derivative = ratio**(-(GAMMA+1)/(2*GAMMA))/(density*sound_speed)
    return value, derivative


def exact_riemann_solution(x, time, rho_l, vel_l, p_l, rho_r, vel_r, p_r,
                            discontinuity=0.5):
    """General exact 1D Euler Riemann solver (Toro, Ch. 4).

    Unlike a Sod-specific solver, this handles all four wave-pattern
    combinations (shock or rarefaction independently on each side), so it
    is valid for arbitrary left/right states -- used to generate exact
    (not scheme-imitated) training targets for a wide variety of
    shock/rarefaction/contact configurations.
    """
    sound_l = np.sqrt(GAMMA*p_l/rho_l)
    sound_r = np.sqrt(GAMMA*p_r/rho_r)

    # Newton iteration for star-region pressure; _pressure_function already
    # branches shock-vs-rarefaction per side based on p_star vs. that
    # side's own initial pressure, so this handles all 4 patterns.
    p_star = max(1e-6, 0.5*(p_l+p_r))
    for _ in range(50):
        f_l, df_l = _pressure_function(p_star, rho_l, p_l)
        f_r, df_r = _pressure_function(p_star, rho_r, p_r)
        update = (f_l+f_r+vel_r-vel_l)/(df_l+df_r)
        new_p = p_star-update
        if new_p <= 0:
            new_p = 0.5*p_star
        if abs(new_p-p_star) < 1e-13*max(1.0, p_star):
            p_star = new_p
            break
        p_star = new_p
    f_l, _ = _pressure_function(p_star, rho_l, p_l)
    f_r, _ = _pressure_function(p_star, rho_r, p_r)
    vel_star = 0.5*(vel_l+vel_r+f_r-f_l)

    similarity = (x-discontinuity)/time
    density = np.empty_like(x)
    velocity = np.empty_like(x)
    pressure = np.empty_like(x)

    left_side = similarity <= vel_star

    if p_star > p_l:
        # left shock
        shock_speed = vel_l-sound_l*np.sqrt(
            (GAMMA+1)/(2*GAMMA)*p_star/p_l+(GAMMA-1)/(2*GAMMA)
        )
        density_star_l = rho_l*(
            (p_star/p_l+(GAMMA-1)/(GAMMA+1))
            / ((GAMMA-1)/(GAMMA+1)*p_star/p_l+1)
        )
        mask = left_side & (similarity < shock_speed)
        density[mask], velocity[mask], pressure[mask] = rho_l, vel_l, p_l
        mask = left_side & (similarity >= shock_speed)
        density[mask], velocity[mask], pressure[mask] = (
            density_star_l, vel_star, p_star
        )
    else:
        # left rarefaction
        sound_star_l = sound_l*(p_star/p_l)**((GAMMA-1)/(2*GAMMA))
        head = vel_l-sound_l
        tail = vel_star-sound_star_l
        density_star_l = rho_l*(p_star/p_l)**(1/GAMMA)
        mask = left_side & (similarity < head)
        density[mask], velocity[mask], pressure[mask] = rho_l, vel_l, p_l
        mask = left_side & (similarity >= head) & (similarity < tail)
        xi = similarity[mask]
        local_sound = 2/(GAMMA+1)*(sound_l+(GAMMA-1)/2*(vel_l-xi))
        velocity[mask] = 2/(GAMMA+1)*(sound_l+(GAMMA-1)/2*vel_l+xi)
        density[mask] = rho_l*(local_sound/sound_l)**(2/(GAMMA-1))
        pressure[mask] = p_l*(local_sound/sound_l)**(2*GAMMA/(GAMMA-1))
        mask = left_side & (similarity >= tail)
        density[mask], velocity[mask], pressure[mask] = (
            density_star_l, vel_star, p_star
        )

    right_side = ~left_side

    if p_star > p_r:
        # right shock
        shock_speed = vel_r+sound_r*np.sqrt(
            (GAMMA+1)/(2*GAMMA)*p_star/p_r+(GAMMA-1)/(2*GAMMA)
        )
        density_star_r = rho_r*(
            (p_star/p_r+(GAMMA-1)/(GAMMA+1))
            / ((GAMMA-1)/(GAMMA+1)*p_star/p_r+1)
        )
        mask = right_side & (similarity <= shock_speed)
        density[mask], velocity[mask], pressure[mask] = (
            density_star_r, vel_star, p_star
        )
        mask = right_side & (similarity > shock_speed)
        density[mask], velocity[mask], pressure[mask] = rho_r, vel_r, p_r
    else:
        # right rarefaction
        sound_star_r = sound_r*(p_star/p_r)**((GAMMA-1)/(2*GAMMA))
        head = vel_r+sound_r
        tail = vel_star+sound_star_r
        density_star_r = rho_r*(p_star/p_r)**(1/GAMMA)
        mask = right_side & (similarity <= tail)
        density[mask], velocity[mask], pressure[mask] = (
            density_star_r, vel_star, p_star
        )
        mask = right_side & (similarity > tail) & (similarity <= head)
        xi = similarity[mask]
        local_sound = 2/(GAMMA+1)*(sound_r-(GAMMA-1)/2*(vel_r-xi))
        velocity[mask] = 2/(GAMMA+1)*(-sound_r+(GAMMA-1)/2*vel_r+xi)
        density[mask] = rho_r*(local_sound/sound_r)**(2/(GAMMA-1))
        pressure[mask] = p_r*(local_sound/sound_r)**(2*GAMMA/(GAMMA-1))
        mask = right_side & (similarity > head)
        density[mask], velocity[mask], pressure[mask] = rho_r, vel_r, p_r

    energy = pressure/(GAMMA-1)+0.5*density*velocity**2
    return np.column_stack((density, density*velocity, energy))


def exact_sod_solution(x, time, discontinuity=0.5):
    """Exact Riemann solution for the canonical Sod initial condition."""
    return exact_riemann_solution(
        x, time, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, discontinuity=discontinuity,
    )


def _feature_locations(x, density):
    gradient = np.abs(np.gradient(density, x))
    exclusion = max(3, len(x)//40)
    strong = np.flatnonzero(gradient >= 0.5*np.max(gradient))
    shock_index = int(strong[-1])
    if shock_index <= exclusion:
        contact_index = shock_index
    else:
        window_cells = max(exclusion+1, int(np.ceil(1.5/(x[1]-x[0]))))
        window_start = max(0, shock_index-window_cells)
        contact_index = window_start+int(np.argmax(
            gradient[window_start:shock_index-exclusion]
        ))
    return float(x[shock_index]), float(x[contact_index])


def regression_metrics(x, solution, initial, final_time, reference):
    density, _, pressure = primitive_variables(solution)
    reference_density, _, reference_pressure = primitive_variables(reference)
    dx = x[1]-x[0]
    errors = np.mean(np.abs(solution-reference), axis=0)
    initial_integral = dx*np.sum(initial, axis=0)
    final_integral = dx*np.sum(solution, axis=0)
    boundary_change = final_time*(flux(initial[0:1])[0]-flux(initial[-1:])[0])
    shock, contact = _feature_locations(x, density)
    reference_shock, reference_contact = _feature_locations(x, reference_density)
    return {
        'density_l1': float(errors[0]),
        'momentum_l1': float(errors[1]),
        'energy_l1': float(errors[2]),
        'minimum_density': float(np.min(density)),
        'minimum_pressure': float(np.min(pressure)),
        'density_overshoot': float(max(
            0, np.max(density)-np.max(reference_density),
            np.min(reference_density)-np.min(density),
        )),
        'pressure_overshoot': float(max(
            0, np.max(pressure)-np.max(reference_pressure),
            np.min(reference_pressure)-np.min(pressure),
        )),
        'shock_location': shock,
        'contact_location': contact,
        'shock_location_error': abs(shock-reference_shock),
        'contact_location_error': abs(contact-reference_contact),
        'conservation_error': np.abs(
            final_integral-initial_integral-boundary_change
        ).tolist(),
    }


def interpolate_reference(reference_x, reference, target_x):
    return np.column_stack([
        np.interp(target_x, reference_x, reference[:,component])
        for component in range(3)
    ])
