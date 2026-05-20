#!/usr/bin/env bash
# Render the deck to PDF via headless Chrome. Open the PDF and visually scan
# for title/chart overlaps, off-grid text, or broken layouts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DECK="$ROOT/deck"
OUT="$ROOT/results/deck_check.pdf"
PORT="${DECK_CHECK_PORT:-8765}"

if [ -n "${CHROME:-}" ]; then
  CHROME_BIN="$CHROME"
elif command -v google-chrome >/dev/null 2>&1; then
  CHROME_BIN="$(command -v google-chrome)"
elif command -v chromium >/dev/null 2>&1; then
  CHROME_BIN="$(command -v chromium)"
elif [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]; then
  CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
else
  echo "ERROR: Chrome/Chromium not found. Set CHROME=... or install one." >&2
  exit 2
fi

mkdir -p "$ROOT/results"
python3 -m http.server "$PORT" --directory "$DECK" >/dev/null 2>&1 &
SERVER=$!
trap "kill $SERVER 2>/dev/null || true" EXIT
sleep 1

"$CHROME_BIN" --headless --disable-gpu --no-sandbox \
  --virtual-time-budget=4000 \
  --print-to-pdf="$OUT" \
  --print-to-pdf-no-header \
  "http://localhost:$PORT/" >/dev/null 2>&1

echo "Wrote $OUT"
echo "Open: open '$OUT' (macOS) -- scan all pages for overlaps."
