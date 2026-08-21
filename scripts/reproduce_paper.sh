#!/usr/bin/env bash
# Reproduce every measured number in the paper, in the order the tables appear.
#
#   bash scripts/reproduce_paper.sh /path/to/Converted_dataset
#
# The argument is the root holding DREAM_real/ and DREAM_to_DREAM_syn/ (see README for how to
# build it). Results are appended to results/summary.tsv as they finish, so the run is safe to
# interrupt and resume by commenting out the splits already recorded.
#
# Runtime: roughly 6-9 hours on one RTX A6000. ORB alone is 32,315 frames.
set -uo pipefail

DATA="${1:?usage: reproduce_paper.sh <Converted_dataset root> [kuka root]}"
# The KUKA splits are read straight from the DREAM download rather than from the index
# (scripts/prepare_dream.py explains why), so this is DREAM's own synthetic/ directory.
KUKA="${2:-$DATA}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
OUT="$ROOT/results"
mkdir -p "$OUT"
TSV="$OUT/summary.tsv"
[ -f "$TSV" ] || printf "table\trobot\tsplit\tocclusion\tadd_auc\n" > "$TSV"

REAL="$DATA/DREAM_real"
SYN="$DATA/DREAM_to_DREAM_syn"

run() {  # $1=table  $2=robot  $3=label  $4=val-dir  $5=occlusion
  local tag="$3${5:+_occ$5}"
  local log="$OUT/$2_$tag.log"
  echo "[$(date +%H:%M)] $2/$tag"
  $PY "$ROOT/scripts/eval.py" --robot "$2" --val-dir "$4" \
      ${5:+--occlude-ratio "$5"} ${6:+--max-frames "$6"} > "$log" 2>&1
  local auc
  auc=$(grep -oE 'ADD-AUC@100mm = [0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')
  printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "${5:-0}" "${auc:-FAIL}" >> "$TSV"
  echo "    ${auc:-FAIL}"
}

# Table 1 (main comparison) and the millimetre table: Panda, full test sets.
run main panda azure     "$REAL/panda-3cam_azure"
run main panda kinect    "$REAL/panda-3cam_kinect360"
run main panda realsense "$REAL/panda-3cam_realsense"
run main panda orb       "$REAL/panda-orb"
run main panda synth_dr    "$SYN/panda_synth_test_dr"
run main panda synth_photo "$SYN/panda_synth_test_photo"

# Table 1, KUKA columns.
run main kuka synth_dr    "$KUKA/kuka_synth_test_dr"
run main kuka synth_photo "$KUKA/kuka_synth_test_photo"

# Occlusion table: the same 300 frames at five occlusion levels.
for r in 0 0.1 0.2 0.3 0.4; do
  run occlusion panda synth_photo "$SYN/panda_synth_test_photo" "$r" 300
done

echo
echo "done — $TSV"
column -t "$TSV"
