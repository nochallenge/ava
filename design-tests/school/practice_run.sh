#!/usr/bin/env bash
# TESTING / DESIGN ONLY. v26 retrieval practice: day 2 (quizzed in a 2nd phrasing, corrected)
# -> night (sleep) -> day 3 (3rd phrasings never seen + untaught questions).
# usage, from design-tests:  school/practice_run.sh <day-2 brain> <dir with day2p.txt, day3.txt> [tag]
set -e
B="$1"; S="$2"; TAG="${3:-P}"; C=./target/release/examples/compose
ARGS="15000 0 0 0 5 2 5 0 128"
before=$(ls ../data/sleep 2>/dev/null | sort)
AVA_LOAD="$B" FLOOR_GB=3 school/guarded.sh "$S/${TAG}_day2p.txt" sh -c "$C $ARGS 148 talk < $S/day2p.txt"
log=$(comm -13 <(echo "$before") <(ls ../data/sleep | sort) | tail -1)
echo "practice log: $log"
D3="${B%.brain}-$TAG-day3.brain"
AVA_LOAD="$B" AVA_SLEEP="../data/sleep/$log" AVA_SAVE="$D3" FLOOR_GB=3 school/guarded.sh "$S/${TAG}_night2.txt" $C $ARGS 148
AVA_LOAD="$D3" FLOOR_GB=3 school/guarded.sh "$S/${TAG}_day3.txt" sh -c "$C $ARGS 148 talk < $S/day3.txt"
