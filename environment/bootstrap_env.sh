#!/usr/bin/env bash
set -euo pipefail

project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
conda=/root/miniconda3/bin/conda
env_dir=/root/miniconda3/envs/ovary_sc
python="$env_dir/bin/python"
mkdir -p /root/autodl-tmp/.conda/pkgs /root/autodl-tmp/.uv-cache "$project/logs"
export NUMBA_THREADING_LAYER=tbb

if ! test -x "$python"; then
  CONDA_PKGS_DIRS=/root/autodl-tmp/.conda/pkgs \
    "$conda" create -y -n ovary_sc python=3.12 pip
fi

"$python" -m pip install --upgrade pip uv
UV_CACHE_DIR=/root/autodl-tmp/.uv-cache UV_CONCURRENT_DOWNLOADS=16 \
"$env_dir/bin/uv" pip install --python "$python" \
  -r "$project/environment/requirements.in"

# Install this repository as an editable package without duplicating dependency resolution.
"$env_dir/bin/uv" pip install --python "$python" --no-deps -e "$project"

"$python" -m ipykernel install --user --name ovary_sc \
  --display-name 'Python (ovary_sc)'

"$env_dir/bin/uv" pip freeze --python "$python" --exclude-editable \
  > "$project/environment/requirements-lock.txt"
CONDA_PKGS_DIRS=/root/autodl-tmp/.conda/pkgs \
  "$conda" env export -n ovary_sc --from-history \
  > "$project/environment/conda-history.yml"

chmod 0755 "$project/environment/activate.sh"
cd "$project"
"$python" "$project/environment/validate_env.py"
