#!/usr/bin/env bash
# TESTING / DESIGN ONLY. one arm of a v27 test: night (sleep one day-1 lessons log into the
# base brain) -> day 2 (ask). the day-1 talk is not rerun: its log holds only the teacher's
# lessons, so every arm sleeps on the very same log.
# usage, from design-tests:
#   school/arm_run.sh <compose exe> <base brain> <sleep log> <day2.txt> <out dir> <tag>
# extra env (AVA_LINKS, AVA_CUE, AVA_LINK_ETA, AVA_LINK_DUMP, ...) passes through.
set -e
C="$1"; B="$2"; LOG="$3"; D2T="$4"; O="$5"; TAG="$6"
ARGS="15000 0 0 0 5 2 5 0 128"
D2="../brain/v27-$TAG-day2.brain"
AVA_LOAD="$B" AVA_SLEEP="$LOG" AVA_SAVE="$D2" FLOOR_GB=3 school/guarded.sh "$O/${TAG}_night.txt" "$C" $ARGS 148
AVA_LOAD="$D2" FLOOR_GB=3 school/guarded.sh "$O/${TAG}_day2.txt" sh -c "\"$C\" $ARGS 148 talk < \"$D2T\""
