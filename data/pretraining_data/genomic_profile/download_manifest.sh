#!/usr/bin/env bash
# Download bigWig files listed in a manifest TSV.
# Comments are in English.

set -euo pipefail

MANIFEST="manifests/epimap_tracks.tsv"   # manifests/epimap_tracks.tsv
OUTDIR="tracks/epimap_avg"     # tracks/epimap_avg
mkdir -p "${OUTDIR}"

tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r track_id assay group mode url filename; do
  echo "[DL] ${track_id}"
  wget -c -O "${OUTDIR}/${filename}" "${url}"
done
