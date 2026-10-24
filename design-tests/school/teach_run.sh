#!/usr/bin/env bash
# TESTING / DESIGN ONLY. day 1 (teach) -> night (sleep) -> day 2 (ask) -> sleep again (ledger).
# usage, from design-tests:  school/teach_run.sh <brain> <scripts-and-outputs dir> [tag] [talk probe] [exam probe]
#   tag names the outputs and the day-2 brain, so runs never overwrite each other.
set -e
B="$1"; S="$2"; TAG="${3:-T}"; PT="${4:-20}"; PE="${5:-116}"; C=./target/release/examples/compose
ARGS="15000 0 0 0 5 2 5 0 128"
before=$(ls ../data/sleep 2>/dev/null | sort)
AVA_LOAD="$B" FLOOR_GB=3 school/guarded.sh "$S/${TAG}_day1.txt" sh -c "$C $ARGS $PT talk < $S/day1.txt"
log=$(comm -13 <(echo "$before") <(ls ../data/sleep | sort) | tail -1)
echo "lessons log: $log"
D2="${B%.brain}-$TAG-day2.brain"
AVA_LOAD="$B" AVA_SLEEP="../data/sleep/$log" AVA_SAVE="$D2" FLOOR_GB=3 school/guarded.sh "$S/${TAG}_night.txt" $C $ARGS $PE
AVA_LOAD="$D2" FLOOR_GB=3 school/guarded.sh "$S/${TAG}_day2.txt" sh -c "$C $ARGS $PT talk < $S/day2.txt"
AVA_LOAD="$D2" AVA_SLEEP="../data/sleep/$log" FLOOR_GB=3 school/guarded.sh "$S/${TAG}_again.txt" $C $ARGS $PE
