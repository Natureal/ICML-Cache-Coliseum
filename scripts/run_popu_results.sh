#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_root_dir="${OUTPUT_ROOT_DIR:-results/popu}"
boost_preds_dir="${BOOST_PREDS_DIR:-boost_traces}"
log_dir="${LOG_DIR:-logs/benchmark/popu_results}"

datasets=(
    astar
    bwaves
    cactusadm
    gems
    leslie3d
    mcf
    omnetpp
    brightkite
    bzip
    citi
    lbm
    libq
    milc
    sphinx3
    xalanc
)

mkdir -p "$output_root_dir" "$boost_preds_dir" "$log_dir"

python - <<'PY'
import sys
import numpy
import pathos
import prettytable
import tqdm

print(f"Using Python: {sys.executable}")
print(f"NumPy: {numpy.__version__}; PrettyTable: {prettytable.__version__}")
PY

for dataset in "${datasets[@]}"; do
    trace_file="traces/${dataset}/${dataset}_test.csv"
    if [[ ! -f "$trace_file" ]]; then
        echo "Missing trace file: $trace_file" >&2
        exit 1
    fi

    echo "Running POPU benchmark for dataset=$dataset"
    python -m benchmark \
        --dataset "$dataset" \
        --real \
        --pred popu \
        --boost \
        --boost_fr \
        --dump_file \
        --output_root_dir "$output_root_dir" \
        --boost_preds_dir "$boost_preds_dir" \
        2>&1 | tee "$log_dir/${dataset}.log"

    generated_result="$output_root_dir/$dataset/1/popu.csv"
    flat_result="$output_root_dir/$dataset.csv"
    if [[ ! -f "$generated_result" ]]; then
        echo "Expected result file was not generated: $generated_result" >&2
        exit 1
    fi
    mv "$generated_result" "$flat_result"
    rmdir "$output_root_dir/$dataset/1" "$output_root_dir/$dataset"
done

echo "Completed all datasets. Result tables are stored as $output_root_dir/<dataset>.csv."
