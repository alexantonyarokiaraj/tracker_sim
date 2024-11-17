#!/bin/bash
export OPENBLAS_NUM_THREADS=1  # Limits OpenBLAS threads to 1 per process

# Define the total number of events and chunk size
total_events=5000
num_cores=20
chunk_size=$((total_events / num_cores))

# Generate and execute the commands with parallel
for ((i=0; i<num_cores; i++)); do
    start=$((i * chunk_size))
    end=$((start + chunk_size - 1))

    echo "python3 tracks.py 0@3@$start@$end"
done | parallel -u
