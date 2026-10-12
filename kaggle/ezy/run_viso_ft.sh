#!/usr/bin/env bash
# ViSO-FT on the rented GPU: ViSO with the SigLIP 2 vision tower and BamiBERT run LIVE and their top 4 blocks fine-tuned (scripts/train_viso_ft.py, configs/viso_ft_bami.yaml).
#   bash run_viso_ft.sh [seed]            default seed 0; job dir <root>/results/viso-ft-bami-s<seed>
# It WAITS until the rental queue is idle (no run.sh / vlm_kernel.py / s1_vlm_lora.py / s1_vlm_eval.py / run_lora2b.sh process), then trains. Resumable: run the same command again after an interruption
# (per-epoch state, the image cache is reused). No Kaggle token. The paired comparison needs the predictions of the reference runs: put the run dirs of the matrix next to the results,
#   <root>/refs/ABL_bami_s0/{preds_dev,preds_test}.jsonl     <root>/refs/A2_s0/{preds_dev,preds_test}.jsonl     (REFS_DIR overrides <root>/refs)
# Environment: FT_WORKERS (DataLoader workers, default min(8, nproc/4)), FT_EXTRA="--enc-lora 16" (the LoRA variant: NOT run by default), FT_WAIT=0 (do not wait for the queue), FT_BATCH.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv
SEED="${1:-0}"
JOB="viso-ft-bami-s$SEED"
REFS_DIR="${REFS_DIR:-$ROOT/refs}"
CODE_DIR="$(dirname "$(find "$INPUT/viso-code" -name train_viso.py 2>/dev/null | head -1)")"
S1_DIR="$(dirname "$(dirname "$(find "$INPUT/vivqa-s1" -path '*/data/train.jsonl' 2>/dev/null | head -1)")")"
[ -f "$CODE_DIR/train_viso.py" ] && [ -d "$S1_DIR/data" ] || die "viso-code / vivqa-s1 not found under $INPUT: run setup.sh first"
[ -f "$INPUT/coco/.verified" ] || die "the COCO images are not verified (setup.sh)"
NPROC="$(nproc 2>/dev/null || echo 8)"
W="${FT_WORKERS:-$(( NPROC / 4 ))}"; [ "$W" -le 8 ] || W=8; [ "$W" -ge 2 ] || W=2
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" VIVQA_S1_ROOT="$S1_DIR"

if [ "${FT_WAIT:-1}" = 1 ]; then
  say "waiting for the rental queue to be idle (run.sh / vlm_kernel / lora / eval processes)"
  while pgrep -f "[r]un\.sh|[v]lm_kernel\.py|[s]1_vlm_lora\.py|[s]1_vlm_eval\.py|[r]un_lora2b\.sh" >/dev/null; do sleep 60; done
fi
say "overlay: the ViSO-FT code over the downloaded viso-code"
overlay_code || die "overlay failed"
say "pre-downloading the two pinned towers (cached after the first run)"
python - <<'PY' || die "model download failed"
from huggingface_hub import snapshot_download
for repo, rev in (("google/siglip2-base-patch16-256", "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"), ("Qualcomm-AI-Research/BamiBERT", "57bc1340debbe4e348ec549047a763caebe4a977")):
    print(repo, snapshot_download(repo, revision=rev, ignore_patterns=["*.onnx", "*.msgpack", "*.h5", "*.gguf"]))
PY
REFS=()
for r in ABL_bami_s0 A2_s0; do [ -f "$REFS_DIR/$r/preds_test.jsonl" ] && REFS+=("$r=$REFS_DIR/$r") || echo "note: no reference predictions for $r in $REFS_DIR (the paired bootstrap against it is skipped)"; done
SHA="$(sha_list_path "$CODE_DIR")" || die "image sha list not found"
python "$HERE/ezy_status.py" status --results "$RESULTS" --queue "$JOB" --job "$JOB" --status running
T0=$(date +%s)
say "JOB $JOB starts (workers $W)"
( cd "$CODE_DIR" && python -u scripts/train_viso_ft.py --out-dir "$RESULTS" --name "$JOB" --seed "$SEED" --s1-dir "$S1_DIR/data" --image-roots "$INPUT" --sha-list "$SHA" \
    --cache-dir "$ROOT/ft_cache" --workers "$W" ${FT_BATCH:+--batch "$FT_BATCH"} --resume ${REFS[@]:+--refs "${REFS[@]}"} ${FT_EXTRA:-} ) 2>&1 | tee -a "$RESULTS/${JOB}_train.log"
rc=${PIPESTATUS[0]}
SECS=$(( $(date +%s) - T0 ))
if [ "$rc" = 0 ] && [ -f "$RESULTS/$JOB/ft_run.json" ] && [ -f "$RESULTS/$JOB/ft_report/ft_report.json" ]; then
  python - "$RESULTS/summary_$JOB.json" "$SECS" <<'PY'
import json, os, sys
p, secs = sys.argv[1], int(sys.argv[2])
json.dump({"model_key": os.path.basename(p)[8:-5], "status": "complete", "stages": {"train_and_predict": {"returncode": 0, "seconds": secs}}, "elapsed_s": secs}, open(p + ".tmp", "w"), indent=1)
os.replace(p + ".tmp", p)
PY
  python "$HERE/ezy_status.py" status --results "$RESULTS" --job "$JOB" --status complete --seconds "$SECS"
  touch "$STATE/done_$JOB"
  printf '\n=====================================================\n  VISO-FT RUN COMPLETE: %s (%dh%02dm this session)\n  Results: %s/%s (ft_run.json, ft_report/, preds_*.jsonl, calibration/)\n  Keep the rental until your pull (rsync) has copied it.\n=====================================================\n' "$JOB" $((SECS / 3600)) $((SECS % 3600 / 60)) "$RESULTS" "$JOB"
else
  python "$HERE/ezy_status.py" status --results "$RESULTS" --job "$JOB" --status incomplete --seconds "$SECS"
  printf '\n!!! VISO-FT RUN NOT COMPLETE (exit %s): run the same command again, it resumes from the last epoch.\n' "$rc"
fi
