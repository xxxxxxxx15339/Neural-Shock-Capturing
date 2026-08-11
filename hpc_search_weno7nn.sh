#!/usr/bin/env bash
# Parallel seed/hyperparameter search for the WENO7-structured NN
# (Train_WENO7NN_Euler.py), meant for a many-core machine.
#
# Background: Train_WENO7NN_Euler.py's built-in --max-attempts runs
# attempts SEQUENTIALLY, one seed after another. Manual search on a
# single core found that regularization=0.10, hidden-size=32,
# learning-rate=0.001, patience=15 is the most promising region (it has
# already produced one candidate that beat WENO5's Shu-Osher regional L1
# by the required 2%), but no single seed has yet passed every
# acceptance criterion simultaneously (shu_regional_l1, shu_regional_linf,
# and sod_total_variation trade off against each other). This script
# fans that same search out across many CPU cores at once instead of
# waiting on one seed at a time.
#
# Usage:
#   ./hpc_search_weno7nn.sh [WORKERS] [ATTEMPTS_PER_WORKER] [BASE_SEED]
#
# Examples:
#   ./hpc_search_weno7nn.sh                # 100 workers, 1 attempt each
#   ./hpc_search_weno7nn.sh 100 3 1000      # 100 workers, 3 attempts each, seeds from 1000
#
# Each worker runs one Train_WENO7NN_Euler.py process pinned to a
# distinct seed range and a distinct regularization value (cycling
# through a small sweep around the known-good 0.10), so the run is both
# a seed search AND a light regularization sweep. Requires GNU xargs
# with -P (standard on Linux).
#
# Prerequisites (see README.md "Environment"):
#   conda create -n weno-nn python=3.7 && conda activate weno-nn
#   pip install -r requirements.txt
#
# Output:
#   hpc_search_results/worker_<i>.log     -- full training log per worker
#   hpc_search_results/worker_<i>.h5      -- ONLY exists if that worker's
#                                             candidate was accepted
#   hpc_search_results/summary.txt        -- pass/fail one-liner per worker

set -u

WORKERS="${1:-100}"
ATTEMPTS_PER_WORKER="${2:-1}"
BASE_SEED="${3:-1000}"

OUT_DIR="hpc_search_results"
mkdir -p "$OUT_DIR"

# Small regularization sweep around the best manually-found value (0.10).
# Workers cycle through these so the 100 cores cover more than just seeds.
# Encoded as a single string (not a bash array) so it survives export to
# the subshells xargs spawns -- bash cannot export arrays via environment.
REG_VALUES_STR="0.08 0.10 0.10 0.10 0.12 0.13"

echo "Launching $WORKERS workers, $ATTEMPTS_PER_WORKER attempt(s) each, base seed $BASE_SEED"
echo "Regularization sweep: $REG_VALUES_STR"
echo "Output directory: $OUT_DIR"
echo

run_worker() {
    local worker_index="$1"
    local seed=$(( BASE_SEED + worker_index * ATTEMPTS_PER_WORKER ))
    local reg_values=($REG_VALUES_STR)
    local reg_count=${#reg_values[@]}
    local reg="${reg_values[$(( worker_index % reg_count ))]}"
    local model_path="$OUT_DIR/worker_${worker_index}.h5"
    local log_path="$OUT_DIR/worker_${worker_index}.log"

    CUDA_VISIBLE_DEVICES=-1 MPLBACKEND=Agg python3 Train_WENO7NN_Euler.py \
        --max-attempts "$ATTEMPTS_PER_WORKER" \
        --seed "$seed" \
        --regularization "$reg" \
        --hidden-size 32 \
        --learning-rate 0.001 \
        --patience 15 \
        --divergence-threshold 0.01 \
        --divergence-boost 50.0 \
        --model-path "$model_path" \
        > "$log_path" 2>&1

    if [ -f "$model_path" ]; then
        echo "worker_${worker_index} seed=${seed} reg=${reg}: ACCEPTED -> ${model_path}"
    else
        echo "worker_${worker_index} seed=${seed} reg=${reg}: rejected"
    fi
}
export -f run_worker
export OUT_DIR ATTEMPTS_PER_WORKER BASE_SEED REG_VALUES_STR

seq 0 $((WORKERS - 1)) | xargs -I{} -P "$WORKERS" bash -c 'run_worker "$@"' _ {}

echo
echo "=== Summary ==="
{
    for i in $(seq 0 $((WORKERS - 1))); do
        model_path="$OUT_DIR/worker_${i}.h5"
        log_path="$OUT_DIR/worker_${i}.log"
        if [ -f "$model_path" ]; then
            echo "worker_${i}: ACCEPTED -> ${model_path}"
        elif [ -f "$log_path" ]; then
            failed=$(grep "Failed criteria" "$log_path" | tail -1)
            echo "worker_${i}: rejected (${failed:-see log})"
        else
            echo "worker_${i}: no log found"
        fi
    done
} | tee "$OUT_DIR/summary.txt"

accepted_count=$(grep -c "ACCEPTED" "$OUT_DIR/summary.txt" || true)
echo
echo "Accepted candidates: $accepted_count / $WORKERS"
if [ "$accepted_count" -gt 0 ]; then
    echo "First accepted model(s):"
    grep "ACCEPTED" "$OUT_DIR/summary.txt"
fi
