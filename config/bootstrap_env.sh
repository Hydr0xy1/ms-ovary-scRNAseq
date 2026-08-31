#!/usr/bin/env bash
set -euo pipefail

project=/root/autodl-tmp/ovary_scRNAseq
conda=/root/miniconda3/bin/conda
env_dir=/root/miniconda3/envs/ovary_sc
python="$env_dir/bin/python"
mkdir -p /root/autodl-tmp/.conda/pkgs /root/autodl-tmp/.uv-cache "$project/logs"
export NUMBA_THREADING_LAYER=omp

if ! test -x "$python"; then
  CONDA_PKGS_DIRS=/root/autodl-tmp/.conda/pkgs \
    "$conda" create -y -n ovary_sc python=3.12 pip
fi

"$python" -m pip install --upgrade pip uv
UV_CACHE_DIR=/root/autodl-tmp/.uv-cache UV_CONCURRENT_DOWNLOADS=16 \
"$env_dir/bin/uv" pip install --python "$python" \
  -r "$project/config/requirements.in"

"$python" -m ipykernel install --user --name ovary_sc \
  --display-name 'Python (ovary_sc)'

"$env_dir/bin/uv" pip freeze --python "$python" \
  > "$project/config/requirements-lock.txt"
CONDA_PKGS_DIRS=/root/autodl-tmp/.conda/pkgs \
  "$conda" env export -n ovary_sc --from-history \
  > "$project/config/conda-history.yml"

chmod 0755 "$project/config/activate.sh"
cd "$project"
"$python" "$project/config/validate_env.py"
