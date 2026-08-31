#!/usr/bin/env bash
set -euo pipefail

project=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$project/environment/activate.sh"

python scripts/01_build_merged.py
python scripts/02_qc.py results/01_merged_counts.h5ad

echo 'QC audit created. Review results/02_qc_summary.tsv and the QC figure.'
echo 'The pipeline intentionally stops before irreversible filtering.'
echo 'After reviewing thresholds, run:'
echo '  python scripts/02_qc.py results/01_merged_counts.h5ad --apply-filter'
