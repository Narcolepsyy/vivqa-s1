#!/usr/bin/env bash
# Apply updated harness files to the already set-up machine (no downloads): copies the .py files shipped next to this script over viso-code. Idempotent.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
overlay_code
echo "done. The scripts run.sh / common.sh / ezy_status.py / vlm_kernel.py next to this file are used directly (no copy needed)."
