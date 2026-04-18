#!/usr/bin/env bash
# Re-fetch MUTCD SVGs from Wikimedia (public domain). Run from repo root: bash printable_signs/download_signs.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
curl -fsSL -o MUTCD_R1-1_stop.svg "https://upload.wikimedia.org/wikipedia/commons/c/c0/MUTCD_R1-1.svg"
curl -fsSL -o MUTCD_R3-5L_left_turn_only.svg "https://upload.wikimedia.org/wikipedia/commons/9/9a/MUTCD_R3-5L.svg"
curl -fsSL -o MUTCD_R3-5R_right_turn_only.svg "https://upload.wikimedia.org/wikipedia/commons/0/09/MUTCD_R3-5R.svg"
curl -fsSL -o MUTCD_R3-6L_left_arrow.svg "https://upload.wikimedia.org/wikipedia/commons/6/62/MUTCD_R3-6L.svg"
curl -fsSL -o MUTCD_R3-6R_right_arrow.svg "https://upload.wikimedia.org/wikipedia/commons/5/5d/MUTCD_R3-6R.svg"
curl -fsSL -o MUTCD_R3-19a_u_turn_only.svg "https://upload.wikimedia.org/wikipedia/commons/2/20/MUTCD_R3-19a.svg"
echo "Done. Files in $DIR"
echo "Note: MUTCD_R1-1_stop_same_size_as_R3-5_sheet.svg is repo-local (scaled stop to R3-5 sheet size)."
