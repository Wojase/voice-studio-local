#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
action="${1:-status}"
if [ "$action" = logs ]; then
  exec tail -n 80 -f "${STUDIO_DATA:-data}/panel.log"
fi
exec .venv/bin/python -X utf8 -m studio.server "$action"
