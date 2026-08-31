#!/usr/bin/env bash
set -euo pipefail

project=/root/autodl-tmp/ovary_scRNAseq
rm -f "$project/logs/bootstrap.exit"
nohup bash -c '
  bash /root/autodl-tmp/ovary_scRNAseq/config/bootstrap_env.sh
  status=$?
  printf "%s\n" "$status" > /root/autodl-tmp/ovary_scRNAseq/logs/bootstrap.exit
  exit "$status"
' > "$project/logs/bootstrap.log" 2>&1 < /dev/null &
printf '%s\n' "$!" | tee "$project/logs/bootstrap.pid"
