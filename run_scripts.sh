#!/bin/bash
export OPENBLAS_NUM_THREADS=1  # Limits OpenBLAS threads to 1 per process

#!/bin/bash

# Generate combinations of ex and cm
# ex_values=(0 5 10)
ex_values=(30)
cm_values=(1 2 3 4 5)

# Generate input lines for parallel
for ex in "${ex_values[@]}"; do
  for cm in "${cm_values[@]}"; do
    echo "$ex@$cm@1@2500"
  done
done > schedule_5.txt

# Run parallel with the generated combinations
# cat input_combinations.txt | parallel -u python3 tracks.py {}

