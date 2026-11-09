#!/usr/bin/env bash
# Strong-scaling sweep: same problem size, increasing number of MPI processes.
#
#   bash scripts/run_scaling.sh 1024 "1 2 4 8"
#
# Use process counts up to the number of PHYSICAL cores of your machine,
# and make sure N is divisible by every process count.
set -euo pipefail

N="${1:-1024}"
PROCS="${2:-1 2 4 8}"
CSV="results/scaling_n${N}.csv"
rm -f "$CSV"

for p in $PROCS; do
    # one thread per process, so that the process count is what scales
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        mpiexec -n "$p" python scripts/benchmark.py --n "$N" --csv "$CSV"
done

python scripts/plot_scaling.py "$CSV"
