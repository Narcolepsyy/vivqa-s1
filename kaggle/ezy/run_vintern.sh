#!/usr/bin/env bash
# Vintern-1B-v3_5 as a VLM baseline on the rented GPU (results/vintern): the only model run with trust_remote_code, pinned to ONE revision (b98f263e...), whose 5 remote .py files were reviewed; the
# harness refuses any other revision and checks their sha256 (printed at load). InternVL's own dynamic tiling (448 px tiles, at most VLM_MAX_TILES = 4 + a thumbnail), fp16, the shared Vietnamese letter prompt in
# the model's chat template, ONE forward pass per item, letter-token logits, 2 option orders for choice, then zero-shot + head-only (3 seeds) on the FULL train, per-type temperature on calib.
#   bash run_vintern.sh                 (no waiting for other jobs: queue it yourself; resumable: run it again after an interruption)
# Environment: VLM_MAX_TILES (default 4), VLM_BATCH (default 8), VLM_WORKERS.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv
python - <<'PY' || { echo "timm is missing: installing it (needed by the InternViT remote code)"; python -m pip install --quiet timm || { echo "could not install timm"; exit 1; }; }
import timm  # noqa: F401
PY
say "overlay: harness files (scripts/vlm_adapters.py, s1_vlm_eval.py ...) over the downloaded viso-code"
overlay_code || die "overlay failed"
KERNEL="$HERE/vlm_kernel.py"
say "pre-downloading the pinned Vintern snapshot (cached after the first run)"
python "$HERE/pull_models.py" "$KERNEL" vintern || die "model download failed"
export VLM_MAX_TILES="${VLM_MAX_TILES:-4}"
bash "$HERE/run.sh" vintern 2>&1 | tee -a "$RESULTS/vintern_run.log"
exit "${PIPESTATUS[0]}"
