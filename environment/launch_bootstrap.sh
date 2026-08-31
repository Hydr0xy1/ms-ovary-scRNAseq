#!/usr/bin/env bash
set -euo pipefail

project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
rm -f "$project/logs/bootstrap.exit"
nohup bash -c "
  bash '$project/environment/bootstrap_env.sh'
  status=\$?
  printf '%s\n' \"\$status\" > '$project/logs/bootstrap.exit'
  exit \"\$status\"
" > "$project/logs/bootstrap.log" 2>&1 < /dev/null &
printf '%s\n' "$!" | tee "$project/logs/bootstrap.pid"
