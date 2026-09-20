#!/usr/bin/env bash
set -euo pipefail

# Keep the isolated environment and package cache on the large data disk.
BASE_PYTHON="${STAGE24_BASE_PYTHON:-/root/miniconda3/envs/ovary_sc/bin/python}"
ENV_DIR="${STAGE24_ENV_DIR:-/root/autodl-tmp/envs/ovary_stage24}"
PROJECT_DIR="${OVARY_PROJECT_ROOT:-/root/autodl-tmp/ovary_scRNAseq}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache-stage24}"
export UV_CACHE_DIR

if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "Base Python not found: ${BASE_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${ENV_DIR}"
fi

"${BASE_PYTHON}" -m uv pip install \
  --python "${ENV_DIR}/bin/python" \
  -r "${PROJECT_DIR}/environment/stage24-requirements.txt"
"${BASE_PYTHON}" -m uv pip install \
  --python "${ENV_DIR}/bin/python" \
  --no-deps -e "${PROJECT_DIR}"

"${ENV_DIR}/bin/python" - <<'PY'
from importlib.metadata import version

import anndata
import scanpy
import scTenifold
import tensorly

for package in ["anndata", "scanpy", "scTenifoldpy", "tensorly"]:
    print(f"{package}=={version(package)}")
PY
