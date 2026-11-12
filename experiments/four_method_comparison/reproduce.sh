#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
TENO_PY="${TENO_PY:-/home/bazouil/miniconda3/envs/teno-jax/bin/python}"
LEGACY_PY="${LEGACY_PY:-/home/bazouil/miniconda3/envs/weno-nn/bin/python}"
TORCH_PY="${TORCH_PY:-/home/bazouil/miniconda3/bin/python}"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg MPLCONFIGDIR=/tmp/four-method-mpl
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
P=experiments/four_method_comparison
if [[ "${REEXPORT:-0}" == 1 || ! -f "$P/results/raw/original_keras_export.npz" ]]; then
  "$LEGACY_PY" "$P/scripts/export_legacy.py" > "$P/logs/keras_load.log" 2>&1
fi
if [[ "${REEXPORT:-0}" == 1 || ! -f "$P/results/raw/neno6_author_export.npz" ]]; then
  "$TORCH_PY" "$P/scripts/export_neno6.py" > "$P/logs/neno6_load.log" 2>&1
fi
if [[ ! -f "$P/CHECKPOINT_INVENTORY.csv" ]]; then
  "$TENO_PY" "$P/scripts/collect_existing_results.py" > "$P/logs/inventory.log" 2>&1
  "$TENO_PY" "$P/scripts/enrich_inventory.py"
fi
if [[ ! -f "$P/configs/benchmark_manifest.json" ]]; then
  "$TENO_PY" "$P/scripts/freeze_protocol.py"
fi
"$TENO_PY" -m pytest tests/test_four_method_comparison.py -q > "$P/logs/adapter_tests.log" 2>&1
# Existing raw runs are identity-checked caches, not overwritten. Remove nothing.
"$TENO_PY" "$P/scripts/run_matched_benchmarks.py" all > "$P/logs/matched_reproduction.log" 2>&1
if [[ ! -f "$P/results/raw/reproduced_original_scalar.json" ]]; then
  "$LEGACY_PY" scripts/legacy_imitation_training/TENO5_vs_NNTENO5_Scalar.py \
    --output-path "$P/results/raw/reproduced_original_scalar.json" --save-dir '' --no-show \
    > "$P/logs/reproduce_original_scalar.log" 2>&1
fi
if [[ ! -f "$P/results/raw/reproduced_phase18_euler.json" ]]; then
  "$TENO_PY" scripts/unrolled_physics_loss/TENO5euler_vs_NNTENO5euler_JAX.py \
    --params-path models/nn_teno5euler_unrolled_jax_params_curriculum_phase18_learnedlambda.pkl \
    --output-path "$P/results/raw/reproduced_phase18_euler.json" --save-dir '' --no-show \
    > "$P/logs/reproduce_phase18.log" 2>&1
fi
"$TENO_PY" "$P/scripts/verify_archives.py"
"$TENO_PY" "$P/scripts/wall_geometry_control.py" > "$P/logs/wall_geometry_control.log" 2>&1
"$TENO_PY" "$P/scripts/smooth_input_diagnostics.py" > "$P/logs/smooth_diagnostics.log" 2>&1
"$TENO_PY" "$P/scripts/smooth_time_control.py" > "$P/logs/smooth_time_control.log" 2>&1
"$TENO_PY" "$P/scripts/actual_input_parity.py" extract
if [[ "${REEXPORT:-0}" == 1 || ! -f "$P/results/raw/actual_keras_predictions.npz" ]]; then
  "$LEGACY_PY" "$P/scripts/actual_input_parity.py" keras > "$P/logs/actual_keras_parity.log" 2>&1
fi
"$TENO_PY" "$P/scripts/actual_input_parity.py" verify
"$TENO_PY" "$P/scripts/calculate_metrics.py"
"$TENO_PY" "$P/scripts/generate_figures.py"
"$TENO_PY" "$P/scripts/build_report.py"
"$TENO_PY" "$P/scripts/verify_package.py"
