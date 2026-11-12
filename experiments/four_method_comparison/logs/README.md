# Execution log interpretation

Authoritative test logs: `cpu_regression.log` and `legacy_smoke.log`.
The full CPU run passes148 tests and37 subtests; the legacy run passes2 tests.
`adapter_tests.log` records the focused parity checks, including author left/right
stencil orientation. The full suite and `actual_input_parity.log` provide
additional numerical regression and real-input parity checks.

`cpu_regression_initial.log` preserves a failure in a new reference test: it
incorrectly required arbitrary point samples of a discontinuous entropy solution
to integrate its mass exactly. The test now uses symmetric moving-frame samples
for that quadrature invariant, retaining13 decimal places; the characteristic
residual, shock entropy condition and Rankine–Hugoniot checks are unchanged.
No numerical reference implementation or benchmark trajectory was modified.

`wall_geometry_control_initial.log` records a serialization error in the extra
control driver. `wall_geometry_filename_preflight.log` is **invalid comparison
output**, caused by Path.with_suffix truncating decimal-containing filename
stems. It repeatedly read one classical smoke cache. Those lines are not results
and are excluded from all scientific tables. The single `raw/geometry_blast_0.*`
cache is preserved as preflight evidence only. The corrected driver uses tokens
`0p001`/`0p038` and checks the cached method, time and case before accepting it.
The authoritative eight controls are `results/wall_geometry_control.csv` and
`raw/geometry_blast_0p001_*` / `raw/geometry_blast_0p038_*`.

`matched_sod.log` and `matched_all.log` retain first-pass computations. Exact
float64 metadata threshold comparisons were then audited in `finalize_runs.log`:
all27 original-network final states and success statuses were identical. Initial
Sod timings were repeated after archival numerical jobs, with identical final
states. First-pass data remain in preflight subdirectories and are not pooled
into the final summaries. Final timing arrays are in each authoritative raw JSON.

`reproduce_original_scalar.log`, `reproduce_phase18.log` and the reproduction
CSV retain both matching archived values and the unresolved neural square-wave
discrepancy. No archive was overwritten or relabeled to conceal that discrepancy.

The final Riemann wave diagnostic caps search windows using exact neighboring
wave separations. The earlier broad contact window selected Lax's stronger
nearby shock. Previous derived wave metrics are preserved in
`raw/preflight_wave_window_diagnostics/` as superseded diagnostics. Primary
solution L1 errors and all numerical trajectories are unchanged. A dedicated
exact-Lax regression test prevents this wave-identification error from recurring.
