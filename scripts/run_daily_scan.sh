#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/universe_scanner.py --limit "${DAILYEDGE_SCAN_LIMIT:-25}" > /tmp/dailyedge_universe_scanner.out
python3 scripts/watcher_ingest.py > /tmp/dailyedge_watcher_ingest.out
printf 'DailyEdge scanner refreshed and watcher candidates ingested.\n'
if [ -f output/universe_scanner_latest.txt ]; then
  sed -n '1,45p' output/universe_scanner_latest.txt
fi
printf '\nWatcher ingest:\n'
sed -n '1,120p' /tmp/dailyedge_watcher_ingest.out
