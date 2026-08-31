#!/usr/bin/env bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ovary_sc

project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Use half the server's CPU cores by default, leaving headroom for the OS and I/O.
# AutoDL may provide OMP_NUM_THREADS=0; OpenMP rejects zero, so replace every
# unset, non-integer, or non-positive thread limit with a safe fallback.
thread_default="${OVARY_THREADS:-16}"
if [[ ! "$thread_default" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Warning: invalid OVARY_THREADS=%q; using 16.\n' "$thread_default" >&2
  thread_default=16
fi

for thread_variable in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMBA_NUM_THREADS; do
  thread_value="${!thread_variable:-}"
  if [[ ! "$thread_value" =~ ^[1-9][0-9]*$ ]]; then
    printf -v "$thread_variable" '%s' "$thread_default"
  fi
  export "$thread_variable"
done
# The project installs a current TBB runtime for Numba and PyNNDescent/BBKNN.
# Use it explicitly instead of inheriting the older runtime from the base image.
export NUMBA_THREADING_LAYER="${OVARY_NUMBA_THREADING_LAYER:-tbb}"

cd "$project"
printf 'Activated ovary_sc in %s\n' "$PWD"
printf 'Thread limits: OMP=%s MKL=%s OPENBLAS=%s NUMBA=%s (%s)\n' \
  "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" \
  "$NUMBA_NUM_THREADS" "$NUMBA_THREADING_LAYER"

unset thread_default thread_value thread_variable
