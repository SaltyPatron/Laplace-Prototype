#!/bin/bash
# Load COPY files from an ingest output directory, with per-table timing.
set -euo pipefail
cd "$1"
P=(psql -h /tmp -p 5439 -U laplace -d laplace -v ON_ERROR_STOP=1 -q)
t0=$(date +%s)
for tb in source entity entity_stats physicality; do
  s=$(date +%s)
  "${P[@]}" -c "\\copy $tb from '$tb.copy'"
  echo "[$(( $(date +%s) - t0 ))s] loaded $tb in $(( $(date +%s) - s ))s: $("${P[@]}" -Atc "select count(*) from $tb") rows"
done
