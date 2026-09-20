#!/usr/bin/env bash
set -euo pipefail

# The official R implementation is installed in an independent prefix on the
# data disk. It is used for a focused one-gene cross-implementation check.
CONDA_BIN="${STAGE24_CONDA_BIN:-/root/miniconda3/bin/conda}"
ENV_DIR="${STAGE24_R_ENV_DIR:-/root/autodl-tmp/envs/ovary_stage24_r}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/autodl-tmp/conda-pkgs-stage24}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda not found: ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -x "${ENV_DIR}/bin/Rscript" ]]; then
  "${CONDA_BIN}" create -y -p "${ENV_DIR}" -c conda-forge \
    r-base=4.4 r-matrix r-cli r-igraph r-reshape2 r-remotes \
    r-rspectra r-irlba
fi

# scTenifoldNet/scTenifoldKnk and several CRAN dependencies contain compiled
# C/C++/Fortran code.  A conda R installation refers to the conda compiler
# wrappers in Makeconf, so keep the complete toolchain inside this isolated
# prefix instead of relying on (or modifying) the system compiler setup.
"${CONDA_BIN}" install -y -p "${ENV_DIR}" -c conda-forge \
  c-compiler cxx-compiler fortran-compiler make pkg-config \
  libcurl openssl libxml2

# Rscript is invoked by absolute path rather than through `conda run`, so make
# the prefix-local compiler wrappers and pkg-config metadata visible explicitly.
export PATH="${ENV_DIR}/bin:${PATH}"
export CONDA_PREFIX="${ENV_DIR}"
export PKG_CONFIG_PATH="${ENV_DIR}/lib/pkgconfig:${ENV_DIR}/share/pkgconfig:${PKG_CONFIG_PATH:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export R_MAKEVARS_USER="${STAGE24_R_MAKEVARS:-${SCRIPT_DIR}/stage24_r.Makevars}"
export MAKEFLAGS="${MAKEFLAGS:--j8}"

"${ENV_DIR}/bin/Rscript" - <<'RS'
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 8)
needed <- c("scTenifoldNet", "scTenifoldKnk")
missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing)
for (pkg in needed) cat(pkg, as.character(packageVersion(pkg)), "\n")
RS
