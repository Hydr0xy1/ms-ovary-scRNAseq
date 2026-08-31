#!/usr/bin/env bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ovary_sc

project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Use half the server's CPU cores by default, leaving headroom for the OS and I/O.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-16}"
export NUMBA_THREADING_LAYER="${NUMBA_THREADING_LAYER:-omp}"

cd "$project"
printf 'Activated ovary_sc in %s\n' "$PWD"
