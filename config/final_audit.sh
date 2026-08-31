#!/usr/bin/env bash
set -euo pipefail

project=/root/autodl-tmp/ovary_scRNAseq
env_dir=/root/miniconda3/envs/ovary_sc

echo ENVIRONMENT
/root/miniconda3/bin/conda env list
"$env_dir/bin/python" --version
"$env_dir/bin/jupyter" kernelspec list

echo DATASET
cat "$project/config/sample_inventory.tsv"
awk 'NR > 1 {cells += $3; nnz += $6} END {printf "samples=%d\ntotal_cells=%d\ntotal_nnz=%d\n", NR-1, cells, nnz}' \
  "$project/config/sample_inventory.tsv"

echo ARTIFACTS
test -s "$project/config/requirements-lock.txt"
test -s "$project/config/conda-history.yml"
test -s "$project/logs/environment_smoke_test.h5ad"
grep -q INPUT_READ_TEST_OK "$project/logs/input_read_test.log"
wc -l "$project/config/requirements-lock.txt"
du -sh "$env_dir" "$project"
df -h / /root/autodl-tmp

echo PROCESSES
pgrep -af '[b]ootstrap_env' || true
echo FINAL_AUDIT_OK
