#!/bin/bash
# Full-grid witness pass over directions x windows x bucket sizes.
# cap=2000, output to ./output/witness/
set -u
cd "$(dirname "$0")"
mkdir -p output/witness

STEPS=(10000000000000000 1000000000000000 100000000000000 10000000000000)
LABELS=("0.01" "0.001" "0.0001" "1e-05")
WINDOWS=(3 7 30 180 0)
DIRECTIONS=("1tok" "kto1")
WORKERS=${WORKERS:-8}

for DIR in "${DIRECTIONS[@]}"; do
  for WD in "${WINDOWS[@]}"; do
    if [ "$WD" = "0" ]; then
      WLBL="all"
    else
      WLBL="${WD}d"
    fi
    for i in 0 1 2 3; do
      S=${STEPS[$i]}
      LBL=${LABELS[$i]}
      OUT=output/witness/witness_${DIR}_${WLBL}_step${LBL}.csv
      if [ -s "$OUT" ]; then
        NR=$(wc -l < "$OUT")
        if [ "$NR" -gt 100 ]; then
          echo "=== have ${DIR} ${WLBL} step=$LBL ETH ($NR rows), skipping ==="
          continue
        fi
      fi
      echo "=== run ${DIR} ${WLBL} step=$LBL ETH workers=$WORKERS cap=2000 $(date -u +%FT%TZ) ==="
      python3 -u witness_entropy_pass.py --direction $DIR --step-wei $S \
          --window-days $WD --workers $WORKERS --cap 2000 --out $OUT
    done
  done
done
