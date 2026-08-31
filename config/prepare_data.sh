#!/usr/bin/env bash
set -euo pipefail

project=/root/autodl-tmp/ovary_scRNAseq
archive=/root/autodl-tmp/Summary.tar.gz
destination="$project/data"
mkdir -p "$destination"

if ! test -d "$destination/Summary"; then
  tar -xzf "$archive" --no-same-owner -C "$destination"
fi

# Keep the input matrices immutable during analysis.
find "$destination/Summary" -type f -exec chmod 0444 {} +
find "$destination/Summary" -type d -exec chmod 0555 {} +

printf 'sample\tgenes\tcells\tmatrix_rows\tmatrix_cols\tnnz\n' \
  > "$project/config/sample_inventory.tsv"
for sample_dir in "$destination"/Summary/*; do
  sample=$(basename "$sample_dir")
  genes=$(gzip -dc "$sample_dir/features.tsv.gz" | wc -l)
  cells=$(gzip -dc "$sample_dir/barcodes.tsv.gz" | wc -l)
  # awk exits after the Matrix Market dimension line; ignore gzip's expected SIGPIPE.
  header=$(gzip -dc "$sample_dir/matrix.mtx.gz" | awk '!/^%/ {print; exit}' || true)
  read -r rows cols nnz <<< "$header"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$sample" "$genes" "$cells" "$rows" "$cols" "$nnz" \
    >> "$project/config/sample_inventory.tsv"
done

cat "$project/config/sample_inventory.tsv"
du -sh "$destination/Summary"
