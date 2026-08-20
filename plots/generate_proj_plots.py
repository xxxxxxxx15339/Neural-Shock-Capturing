# -*- coding: utf-8 -*-
"""Generates the proj.pdf-implementation results plots (for showing alongside
the annotated PDF sent to the professor). Reads real data written by earlier
scripts (training histories, the Euler instability trace) plus a few numbers
transcribed exactly from Results.md Phase 10 / the ValidationMatrix_ProjPDF.py
and PDEDataset_NonIdentifiability_Audit.py console output.

Palette: dataviz skill's validated default (references/palette.md), light
surface, categorical slots 1-3 (blue/orange/aqua) and the status pair
(good/critical) used for win/loss framing.

Run (pure matplotlib/numpy/csv, no jax/keras needed):

    conda run -n base python plots/generate_proj_plots.py
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- palette (references/palette.md, light mode) ---------------------------
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'
GOOD = '#0ca30c'
CRITICAL = '#d03b3b'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Segoe UI', 'Arial'],
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK_SECONDARY,
    'text.color': INK_PRIMARY,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'grid.color': GRIDLINE,
    'axes.grid': True,
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,
    'axes.linewidth': 0.8,
    'font.size': 11,
})


def style_axes(ax, ylog=False):
    ax.grid(True, axis='y', linewidth=0.8, color=GRIDLINE)
    ax.grid(False, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(length=0)
    if ylog:
        ax.set_yscale('log')


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', path)
    plt.close(fig)


def load_history(run_tag):
    path = os.path.join(ROOT, 'models', 'nn_teno5_unrolled_jax_history_{}.csv'.format(run_tag))
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [{k: float(v) for k, v in row.items()} for row in reader]
    return rows


# ============================================================================
# Plot 1: the diagnosis -- L_MSE vs. lambda_tv*L_TV, unnormalized (Run 1)
# vs. normalized (Run 2). Small multiples, shared log y-axis, so the
# multi-order-of-magnitude imbalance in Run 1 is directly visible.
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
LAMBDA_TV = 0.1

panels = [
    ('run1_unnormalized', 'Run 1: L_TV as a raw sum (Eq. 7 literal)'),
    ('run2_normalized', 'Run 2: L_TV normalized (mean, not sum)'),
]

def rolling(values, window=15):
    return np.convolve(values, np.ones(window) / window, mode='valid')

for ax, (run_tag, title) in zip(axes, panels):
    rows = load_history(run_tag)
    l_mse = np.array([r['l_mse'] for r in rows])
    weighted_tv = LAMBDA_TV * np.array([r['l_tv'] for r in rows])
    steps = np.arange(len(l_mse))

    ax.plot(steps, l_mse, color=BLUE, linewidth=0.5, alpha=0.2)
    ax.plot(steps, weighted_tv, color=ORANGE, linewidth=0.5, alpha=0.2)
    smoothed_mse = rolling(l_mse)
    smoothed_tv = rolling(weighted_tv)
    ax.plot(steps[14:], smoothed_mse, color=BLUE, linewidth=2.2, label='lambda_mse x L_MSE')
    ax.plot(steps[14:], smoothed_tv, color=ORANGE, linewidth=2.2, label='lambda_tv x L_TV')

    style_axes(ax, ylog=True)
    ax.set_xlabel('training step')
    ax.set_title(title, loc='left', fontsize=11.5, color=INK_PRIMARY, pad=10)

axes[0].set_ylabel('weighted loss-term contribution (log scale)')
axes[0].legend(frameon=False, loc='lower left', fontsize=9.5)
fig.suptitle('The bug: an unnormalized L_TV dominates L_MSE by ~4x, even after its 0.1 weight',
             x=0.02, ha='left', fontsize=13, color=INK_PRIMARY, y=1.04)
fig.tight_layout()
savefig(fig, '01_scalar_training_curves.png')


# ============================================================================
# Plot 2: held-out beats_teacher results, run 3 vs deterministic TENO5
# ============================================================================

cases = ['held-out jump\nK steps', 'held-out jump\n3x K steps', 'held-out\nsmooth sine']
nn_vals = [0.1147849, 0.1162609, 0.004939791]
teno_vals = [0.1178897, 0.1253698, 0.004771968]
wins = [True, True, False]  # third is a near-tie loss

fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(cases))
width = 0.32

bars_nn = ax.bar(x - width / 2, nn_vals, width, color=BLUE, label='NN-TENO5 (JAX, unrolled-loss-trained)')
bars_teno = ax.bar(x + width / 2, teno_vals, width, color=ORANGE, label='Deterministic TENO5')

for i, (bar_nn, win) in enumerate(zip(bars_nn, wins)):
    marker = '✓ beats TENO5' if win else '≈ near-tie'
    color = GOOD if win else INK_MUTED
    height = max(nn_vals[i], teno_vals[i])
    ax.text(i, height * 1.06, marker, ha='center', fontsize=9.5, color=color, fontweight='bold')

style_axes(ax)
ax.set_xticks(x)
ax.set_xticklabels(cases, fontsize=10)
ax.set_ylabel('L2 error vs. exact solution')
ax.set_title('Scalar advection: 2 of 3 held-out cases beat TENO5', loc='left', fontsize=13, color=INK_PRIMARY, pad=14)
ax.legend(frameon=False, loc='upper left', fontsize=9.5)
ax.set_ylim(top=max(nn_vals + teno_vals) * 1.25)
savefig(fig, '02_scalar_beats_teacher.png')


# ============================================================================
# Plot 3: validation matrix -- canonical proj.pdf problems, same network
# ============================================================================

cases_v = ['Advection\n(step, final_time=1.0)', 'Burgers 1D\n(out-of-distribution)']
nn_v = [1.4018e-01, 2.9531e-03]
teno_v = [9.8960e-02, 8.1670e-04]

fig, ax = plt.subplots(figsize=(6.5, 4.8))
x = np.arange(len(cases_v))
width = 0.32
ax.bar(x - width / 2, nn_v, width, color=BLUE, label='NN-TENO5 (JAX)')
ax.bar(x + width / 2, teno_v, width, color=ORANGE, label='Deterministic TENO5')

for i in range(len(cases_v)):
    height = max(nn_v[i], teno_v[i])
    ax.text(i, height * 1.35, 'loses to TENO5', ha='center', fontsize=9.5, color=CRITICAL, fontweight='bold')

style_axes(ax, ylog=True)
ax.set_xticks(x)
ax.set_xticklabels(cases_v, fontsize=10)
ax.set_ylabel('L2 error vs. exact solution (log scale)')
ax.set_ylim(top=1.0)
ax.set_title('Canonical benchmark problems: the result is narrow, not general', loc='left', fontsize=12.5, color=INK_PRIMARY, pad=14)
ax.legend(frameon=False, loc='lower left', fontsize=9.5)
savefig(fig, '03_validation_matrix.png')


# ============================================================================
# Plot 4: Euler instability trace -- max density per step, several alpha margins
# ============================================================================

trace_path = os.path.join(ROOT, 'models', 'euler_instability_trace.csv')
with open(trace_path) as f:
    reader = csv.DictReader(f)
    trace_rows = [
        {'alpha_margin': float(r['alpha_margin']), 'step': int(r['step']),
         'max_density': float(r['max_density']), 'finite': r['finite'] == 'True'}
        for r in reader
    ]

margins = sorted(set(r['alpha_margin'] for r in trace_rows))
margin_colors = {1.0: BLUE, 1.5: AQUA, 2.0: ORANGE, 3.0: '#eda100', 5.0: CRITICAL}

fig, ax = plt.subplots(figsize=(9.5, 4.8))
for margin in margins:
    rows_m = [r for r in trace_rows if r['alpha_margin'] == margin]
    steps_m = [r['step'] for r in rows_m if r['finite']]
    density_m = [r['max_density'] for r in rows_m if r['finite']]
    color = margin_colors.get(margin, INK_MUTED)
    ax.plot(steps_m, density_m, color=color, linewidth=2.0, marker='o', markersize=4,
            label='alpha margin = {:.1f}x{}'.format(margin, ' (baseline)' if margin == 1.0 else ''))
    last_finite_step = max((r['step'] for r in rows_m if r['finite']), default=None)
    went_nan = any(not r['finite'] for r in rows_m)
    if went_nan and last_finite_step is not None:
        ax.scatter([last_finite_step + 1], [density_m[-1] * 3], marker='x', s=70, color=color, zorder=5)

style_axes(ax, ylog=True)
ax.set_xlabel('SSPRK3 step')
ax.set_ylabel('max density in state (log scale)')
ax.set_title('Euler training instability: more Lax-Friedrichs dissipation makes it WORSE', loc='left', fontsize=12.5, color=INK_PRIMARY, pad=14)
ax.legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1.0))
fig.text(0.02, -0.02, '× = state went non-finite (NaN) on the next step', fontsize=8.5, color=INK_MUTED)
fig.tight_layout()
savefig(fig, '04_euler_instability.png')


# ============================================================================
# Plot 5: non-identifiability audit -- conflict rate, value vs. label, 2 precisions
# ============================================================================

precisions = ['coarse\n(1 decimal)', 'fine\n(2 decimals)']
value_rates = [79.87, 51.86]
label_rates = [78.07, 37.08]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
x = np.arange(len(precisions))
width = 0.32
ax.bar(x - width / 2, value_rates, width, color=BLUE, label='value-regression target')
ax.bar(x + width / 2, label_rates, width, color=AQUA, label='oracle keep/discard label (mitigation)')

for i in range(len(precisions)):
    improvement = value_rates[i] - label_rates[i]
    ax.text(i, max(value_rates[i], label_rates[i]) + 3,
            '-{:.1f} pts'.format(improvement), ha='center', fontsize=9.5, color=INK_SECONDARY)

style_axes(ax)
ax.set_xticks(x)
ax.set_xticklabels(precisions, fontsize=10)
ax.set_ylabel('conflict rate among multi-trajectory bins (%)')
ax.set_ylim(0, 100)
ax.set_title('60% real-PDE data bucket: same-shape stencils, different correct answers', loc='left', fontsize=12, color=INK_PRIMARY, pad=14)
ax.legend(frameon=False, loc='upper right', fontsize=9.5)
savefig(fig, '05_non_identifiability_audit.png')

print('\nAll 5 plots written to', OUT_DIR)
