#!/bin/bash
export OPENBLAS_NUM_THREADS=1  # Limits OpenBLAS threads to 1 per process

#!/bin/bash

# Generate combinations of ex and cm
# ex_values=(0 5 10)
ex_values=(10)
cm_values=(5)

# Generate input lines for parallel
for ex in "${ex_values[@]}"; do
  for cm in "${cm_values[@]}"; do
    echo "$ex@$cm@1@500"
    echo "$ex@$cm@501@1000"
    echo "$ex@$cm@1001@1500"
    echo "$ex@$cm@1501@2000"
    echo "$ex@$cm@2001@2500"
    echo "$ex@$cm@2501@3000"
    echo "$ex@$cm@3001@3500"
    echo "$ex@$cm@3501@4000"
    echo "$ex@$cm@4001@4500"
    echo "$ex@$cm@4501@5000"
  done
done > schedule_5.txt

# Run parallel with the generated combinations
# cat input_combinations.txt | parallel -u python3 tracks.py {}

