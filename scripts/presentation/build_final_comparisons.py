"""Regenerate presentation comparisons from the final saved checkpoints.

Evaluate the established canonical configurations without retraining or changing
any model. Save final profiles, checked errors and readable reference overlays.
Run using the teno-jax environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.differentiable.numerics_jax import lax_friedrichs_flux, rollout
from src.differentiable.euler_jax import get_euler_flux_jax
from src.differentiable.projection_jax import project_keep_weight
from src.differentiable.teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from src.differentiable.teno5_jax_toy import toy_keep_weight, toy_lambda
from src.analysis.euler_regression import exact_sod_solution, primitive_variables
from src.initial_conditions.InitialConditions import sod, shuOsher
from src.core.eulerEquations import spds

INK, GRAY, GOOD, BAD = "#242424", "#929292", "#15803D", "#B91C1C"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#BEBEBE", "axes.labelcolor": "#595959",
                     "xtick.color": "#595959", "ytick.color": "#595959",
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def learned_reconstruction(checkpoint):
    from src.differentiable.model_io import load_checkpoint
    from src.differentiable.reconstruction import make_reconstruction
    return make_reconstruction(load_checkpoint(checkpoint), 'learned-blend')


def initial_burgers(x):
    return 0.5 + 0.25 * np.sin(2.0 * np.pi * x)


def exact_burgers(x, final_time):
    if final_time >= 2.0 / np.pi:
        raise ValueError("The analytic reference is valid only before shock formation.")
    first = final_time * initial_burgers(np.array([0.0]))[0]
    target = np.mod(x, 1.0)
    target = np.where(target < first, target + 1.0, target)
    lower, upper = np.zeros_like(target), np.ones_like(target)
    for _ in range(60):
        midpoint = .5 * (lower + upper)
        mapped = midpoint + final_time * initial_burgers(midpoint)
        right = mapped < target
        lower = np.where(right, midpoint, lower)
        upper = np.where(right, upper, midpoint)
    return initial_burgers(.5 * (lower + upper))


def scalar_run(initial, physical_flux, reconstruction, final_time):
    cells, cfl, wave_speed = len(initial), .5, 1.0
    dx = 1.0 / cells
    steps = int(np.ceil(final_time * wave_speed / (cfl * dx)))
    flux = lambda state: lax_friedrichs_flux(state, physical_flux, wave_speed, reconstruction)
    return np.asarray(rollout(jnp.asarray(initial), flux, final_time / steps, dx, steps)), steps


def euler_run(initial, length, final_time, reconstruction):
    cells, cfl = len(initial), .4
    dx = length / cells
    wave_speed = float(np.max(np.abs(spds(initial))) * 1.3)
    steps = max(1, int(np.ceil(final_time * wave_speed / (cfl * dx))))
    flux = get_euler_flux_jax(reconstruction, boundary="transmissive")
    final = np.asarray(rollout(jnp.asarray(initial), flux, final_time / steps, dx, steps))
    rho, _, pressure = primitive_variables(final)
    if not np.isfinite(final).all() or np.min(rho) <= 0 or np.min(pressure) <= 0:
        raise ValueError("The saved final model did not produce an admissible unguarded rollout.")
    return final, steps


def metric(reference, solution, system):
    difference = solution - reference
    if system == "scalar":
        return float(np.sqrt(np.mean(difference ** 2)))
    return float(np.mean(np.mean(np.abs(difference), axis=0)))


def axes_style(ax):
    ax.grid(axis="y", color="#E7E7E7", linewidth=.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=5, labelsize=12)


def overlay(name, data, output, system):
    x, reference = data["x"], data["reference"]
    teacher, neural = data["teno"], data["nn"]
    if system == "euler":
        teacher_error = np.mean(np.abs(teacher-reference), axis=1)
        neural_error = np.mean(np.abs(neural-reference), axis=1)
        reference, teacher, neural = reference[:, 0], teacher[:, 0], neural[:, 0]
        fine_x = data["fine_x"]
        fine_reference = data["fine_reference"][:, 0]
    else:
        teacher_error, neural_error = teacher-reference, neural-reference
        fine_x, fine_reference = data["fine_x"], data["fine_reference"]
    color = BAD if name == "sod" else GOOD
    fig = plt.figure(figsize=(5.95, 3.80))
    ax = fig.add_axes([.13, .49, .84, .37])
    ax.plot(fine_x, fine_reference, color=INK, linestyle=":", linewidth=1.65, label="Reference", zorder=1)
    ax.plot(x, teacher, color=GRAY, linestyle="--", linewidth=1.9, label="TENO5", zorder=2)
    ax.plot(x, neural, color=color, linewidth=1.6, label="NN-TENO5", zorder=3)
    axes_style(ax)
    ax.set_ylabel("Density" if system == "euler" else "u", fontsize=13)
    ax.legend(frameon=False, fontsize=11.5, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.25),
              handlelength=2.15, columnspacing=1.1, handletextpad=.5)
    ax.set_xlim(0, 10 if name == "shu_osher" else 1)
    low, high = float(np.min(fine_reference)), float(np.max(fine_reference))
    ax.set_ylim(low - .07 * max(high-low, .1), high + .18 * max(high-low, .1))

    # A separate error strip never obscures the full-domain solution curves.
    if name == "advection":
        lo, hi = .17, .35
    elif name == "burgers":
        center = float(x[np.argmax(np.abs(teacher-reference))])
        lo, hi = max(0, center-.13), min(1, center+.13)
    elif name == "sod":
        lo, hi = .78, .92
    else:
        lo, hi = 5.8, 7.5
    inset = fig.add_axes([.13, .13, .84, .20])
    inset.axhline(0, color=INK, linestyle=":", linewidth=.8)
    inset.plot(x, teacher_error, color=GRAY, linestyle="--", linewidth=1.35,
               marker="o", markersize=2.3, markerfacecolor="white", markeredgewidth=.65)
    inset.plot(x, neural_error, color=color, linewidth=1.25)
    inset.set_xlim(lo, hi)
    selected = (x >= lo) & (x <= hi)
    errors = np.concatenate([teacher_error[selected], neural_error[selected]])
    peak = max(float(np.max(np.abs(errors))), 1e-10)
    inset.set_ylim((-.06 if system == "euler" else -1.15)*peak, 1.15*peak)
    inset.set_title("Local mean-error zoom" if system == "euler" else "Local error zoom",
                    loc="left", fontsize=11, color="#595959", pad=6, y=1.02)
    inset.set_xlabel("x (local region)", fontsize=11, labelpad=3)
    inset.set_ylabel("Mean L₁" if system == "euler" else "Error", fontsize=11)
    inset.tick_params(length=0, labelsize=10, pad=3)
    if peak < .01:
        from matplotlib.ticker import FuncFormatter
        inset.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1000:g}"))
        inset.set_ylabel(r"Error ($\times10^{-3}$)", fontsize=10.5)
    inset.grid(axis="y", color="#ECECEC", linewidth=.5)
    fig.savefig(output, dpi=260, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Documentation/presentation_professional/figures")
    parser.add_argument("--plots-only", action="store_true", help="Redraw the saved profiles without evaluating the solvers again.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        with np.load(args.output_dir / "final_profiles.npz") as saved_profiles:
            for name in ("advection", "burgers", "sod", "shu_osher"):
                data = {key: saved_profiles[f"{name}_{key}"] for key in
                        ("x", "reference", "teno", "nn", "fine_x", "fine_reference")}
                overlay(name, data, args.output_dir / f"{name}_final.png",
                        "scalar" if name in ("advection", "burgers") else "euler")
        print("Redrew the four saved final comparisons.", flush=True)
        return
    scalar_path = ROOT / "models/nn_teno5_unrolled_jax_params_run6_learnedlambda.pkl"
    euler_path = ROOT / "models/nn_teno5euler_unrolled_jax_params_curriculum_phase18_learnedlambda.pkl"
    scalar_nn, euler_nn = learned_reconstruction(scalar_path), learned_reconstruction(euler_path)
    records, profiles, packed = {}, {}, {}
    x = np.arange(64, dtype=float)/64
    fine_x = np.linspace(0, 1, 1200)
    step = lambda points: ((np.mod(points, 1) >= .25) & (np.mod(points, 1) < .75)).astype(float)
    for name, initial, final_time, physical_flux, reference in [
        ("advection", step(x), 1., lambda u: u, step(x-1)),
        ("burgers", initial_burgers(x), .5, lambda u: .5*u**2, exact_burgers(x, .5)),
    ]:
        started = time.perf_counter()
        print("Evaluating final", name, flush=True)
        classical, steps = scalar_run(initial, physical_flux, teno5_reconstruct_jax, final_time)
        neural, _ = scalar_run(initial, physical_flux, scalar_nn, final_time)
        profiles[name] = dict(x=x, reference=reference, teno=classical, nn=neural, fine_x=fine_x,
                              fine_reference=step(fine_x-1) if name == "advection" else exact_burgers(fine_x,.5))
        records[name] = dict(metric="Scalar L2", cells=64, cfl=.5, final_time=final_time, steps=steps,
                             reference="exact solution", checkpoint=str(scalar_path.relative_to(ROOT)),
                             teno=metric(reference,classical,"scalar"), nn=metric(reference,neural,"scalar"))
        print(name, records[name], "elapsed", round(time.perf_counter()-started,2), flush=True)

    for name, length, final_time, initial_function in [("sod",1.,.2,sod()), ("shu_osher",10.,1.8,shuOsher())]:
        print("Evaluating final", name, flush=True)
        x_e = np.arange(120, dtype=float)*length/120
        initial = initial_function(x_e)
        classical, steps = euler_run(initial,length,final_time,teno5_reconstruct_jax)
        neural, _ = euler_run(initial,length,final_time,euler_nn)
        if name == "sod":
            reference = exact_sod_solution(x_e,final_time)
            fine_x_e = np.linspace(0,1,1200)
            fine_ref = exact_sod_solution(fine_x_e,final_time)
            label = "exact Riemann solution"
        else:
            print("Building the 320-cell Shu–Osher reference",flush=True)
            fine_x_e = np.arange(320,dtype=float)*length/320
            fine_ref, _ = euler_run(initial_function(fine_x_e),length,final_time,teno5_reconstruct_jax)
            reference = np.column_stack([np.interp(x_e,fine_x_e,fine_ref[:,column]) for column in range(3)])
            label = "TENO5 at 320 cells (higher-resolution numerical reference)"
        profiles[name] = dict(x=x_e, reference=reference, teno=classical, nn=neural,
                              fine_x=fine_x_e, fine_reference=fine_ref)
        records[name] = dict(metric="Mean Euler L1", cells=120,cfl=.4,final_time=final_time,steps=steps,
                             reference=label, checkpoint=str(euler_path.relative_to(ROOT)), positivity_guard_used=False,
                             teno=metric(reference,classical,"euler"),nn=metric(reference,neural,"euler"))
        print(name, records[name], flush=True)

    saved = json.loads((ROOT / "models/nn_teno5euler_jax_comparison_phase18.json").read_text())
    expected_scalar = {"advection":(.098960,.095521),"burgers":(.00081670,.00066493)}
    for name, record in records.items():
        record["ratio"] = record["nn"]/record["teno"]
        record["change_percent"] = 100*(record["ratio"]-1)
        if name in expected_scalar:
            expected = expected_scalar[name]
            tolerance = 2e-5
        else:
            check = saved["problems"][name]["beats_teacher"]
            expected = (check["teno5euler"],check["nn_teno5euler"])
            tolerance = 1e-8
        if not np.allclose([record["teno"],record["nn"]],expected,rtol=tolerance,atol=1e-10):
            raise AssertionError((name,record,expected))
        overlay(name,profiles[name],args.output_dir/f"{name}_final.png","scalar" if name in expected_scalar else "euler")
        for key, value in profiles[name].items():
            packed[f"{name}_{key}"] = value
    np.savez_compressed(args.output_dir/"final_profiles.npz",**packed)
    (args.output_dir/"final_comparison_metrics.json").write_text(json.dumps(records,indent=2)+"\n")
    print("All four final comparisons match the documented checkpoints. Saved",args.output_dir,flush=True)


if __name__ == "__main__":
    main()
