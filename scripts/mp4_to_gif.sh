#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Convert an MP4 file to a GIF using ffmpeg palette generation.

Usage:
  scripts/mp4_to_gif.sh [input.mp4] [output.gif]

Environment variables:
  FPS    Frames per second for the GIF. Default: 12
  WIDTH  Output width in pixels. Default: 560

Examples:
  scripts/mp4_to_gif.sh
  scripts/mp4_to_gif.sh results/demo.mp4 results/demo.gif
  FPS=15 WIDTH=960 scripts/mp4_to_gif.sh results/demo.mp4
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

input="${1:-results/starling2max_policy.mp4}"
output="${2:-${input%.*}.gif}"
fps="${FPS:-12}"
width="${WIDTH:-560}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$input" ]]; then
  echo "Input video not found: $input" >&2
  exit 1
fi

mkdir -p "$(dirname "$output")"
palette="$(mktemp "${TMPDIR:-/tmp}/mp4_to_gif_palette.XXXXXX.png")"
trap 'rm -f "$palette"' EXIT

ffmpeg -y -i "$input" \
  -vf "fps=${fps},scale=${width}:-1:flags=lanczos,palettegen=stats_mode=diff" \
  "$palette"

ffmpeg -y -i "$input" -i "$palette" \
  -lavfi "fps=${fps},scale=${width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  "$output"

echo "Wrote $output"
