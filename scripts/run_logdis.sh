#!/bin/bash
datasets=("astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")
MAX_JOBS=1

mkdir -p logs/benchmark/oracle
pids=()
for dataset in "${datasets[@]}"; do
    while [ ${#pids[@]} -ge $MAX_JOBS ]; do
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $MAX_JOBS ] && sleep 5
    done

    echo "Running dataset=$dataset"
    python -m benchmark --boost_fr --dataset "$dataset" --oracle --pred oracle_dis --noise_type logdis --dump_file --output_root_dir stat > "logs/benchmark/oracle/${dataset}_logdis.log" 2>&1 &
    pids+=($!)
done

echo "Waiting for ${#pids[@]} jobs to finish..."
wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name logdis --results_dir stat
