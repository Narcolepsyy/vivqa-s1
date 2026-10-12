#!/usr/bin/env bash
# Run VLM jobs on the rented GPU with the same protocol as the Kaggle kernels (fp16, one forward pass per unit, resumable shards, heartbeat, watchdog),
# Results stay in $RESULTS (<root>/results; RUN_STATUS.json there says per job: status, seconds, files) and are pulled by rsync from the user's PC. Rsync tip: --exclude='*.tmp*' (temp files of files being written).
# Re-running the same command resumes: finished jobs are skipped (state/done_<job>), finished shards are kept, LoRA continues from its last checkpoint (every 1/4 epoch). Results copied back into $RESULTS are picked up the same way.
# OPTIONAL (off by default): KAGGLE_UPLOAD=1 also uploads each finished job to the PRIVATE Kaggle dataset narcolepsyy/vivqa-vlm-results (asks for the Kaggle token; prints SAFE TO END RENTAL).
#   bash run.sh preflight                       10-minute speed test (Intern-Decision-0.8B, calib only) -> projected hours; nothing is uploaded
#   bash run.sh intern jpt onejev               head-only protocol: zero-shot + linear head, 3 seeds (train features are the long part)
#   bash run.sh qwen                            only if the Kaggle vlm-qwen run ended incomplete
#   bash run.sh ref-intern2b ref-intern4b ref-qwen3vl8b     reference tier: zero-shot only (calib / dev / test), fp16, no quantisation
#   bash run.sh vintern                         Vintern-1B-v3_5 (the only trust_remote_code model: pinned revision, reviewed + hash-checked remote code; 4 dynamic tiles): zero-shot + head-only like the small models
#   bash run.sh lora-img lora-txt               LoRA decision fine-tune of Qwen3.5-0.8B (with images / text-only control): 1 epoch, 1 seed, bf16 (see scripts/s1_vlm_lora.py)
#   bash run.sh restore                         only re-create state/done_<job> from complete summaries found in $RESULTS (also done automatically at every start)
#   bash run.sh small | lora | ref | all        groups: small = intern jpt onejev, lora = lora-img lora-txt, ref = the 3 reference models, all = small + lora + ref (in that order)
# Environment: MODEL_CLEANUP=1 (delete a model's Hugging Face cache after its job is uploaded and confirmed: small disks; it is re-downloaded if needed again)  VLM_HEAD_TRAIN_LIMIT=8000 (train subset, from the preflight result; default 0 = all train)  FORCE=1 (rerun finished jobs)  KAGGLE_UPLOAD=1 (+ UPLOAD_SHARDS=1|0)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
KERNEL="${EZY_KERNEL:-$HERE/vlm_kernel.py}"; [ -f "$KERNEL" ] || KERNEL="$HERE/../vlm_kernel.py"
[ -f "$KERNEL" ] || die "vlm_kernel.py not found next to run.sh"
activate_venv

DO_UPLOAD=0
if [ "${KAGGLE_UPLOAD:-0}" = 1 ] || [ "${SKIP_UPLOAD:-1}" = 0 ]; then DO_UPLOAD=1; fi
STATUS_PY="$HERE/ezy_status.py"
[ $# -ge 1 ] || { sed -n 2,19p "$0"; exit 1; }
JOBS=(); RESTORE=0
for j in "$@"; do
  case "$j" in
    small) JOBS+=(intern jpt onejev) ;;
    lora)  JOBS+=(lora-img lora-txt) ;;
    ref)   JOBS+=(ref-intern2b ref-intern4b ref-qwen3vl8b) ;;
    all)   JOBS+=(intern jpt onejev lora-img lora-txt ref-intern2b ref-intern4b ref-qwen3vl8b) ;;
    restore) RESTORE=1 ;;
    preflight|intern|jpt|onejev|qwen|vintern|lora-img|lora-txt|ref-intern2b|ref-intern4b|ref-qwen3vl8b) JOBS+=("$j") ;;
    *) die "unknown job '$j' (see the usage at the top of run.sh)" ;;
  esac
done
if [ "${#JOBS[@]}" -eq 0 ] && [ "$RESTORE" = 1 ]; then
  python "$STATUS_PY" markers --results "$RESULTS" --state "$STATE"; exit 0
fi
[ -f "$INPUT/coco/.verified" ] || die "setup.sh has not finished (COCO images not verified)"
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ -t 1 ] && [ -z "${EZY_NO_TMUX_WARN:-}" ]; then
  echo "WARNING: not inside tmux/screen. If the browser terminal disconnects, the run dies. Recommended:  tmux new -s run   (reattach: tmux attach -t run)"
  echo "Continuing in 10 s (Ctrl-C to stop)..."; sleep 10
fi
HEAD_LIMIT="${VLM_HEAD_TRAIN_LIMIT:-0}"
# DataLoader workers: Kaggle uses 2; with 32 vCPU the GPU starved at 2 (14.9 units/s) and ran 2.3x faster with 8 workers + 4 math threads in the main process, 1 per worker
NPROC="$(nproc 2>/dev/null || echo 8)"
if [ -z "${VLM_WORKERS:-}" ]; then VLM_WORKERS=$(( NPROC / 4 )); [ "$VLM_WORKERS" -le 8 ] || VLM_WORKERS=8; [ "$VLM_WORKERS" -ge 2 ] || VLM_WORKERS=2; fi
export VLM_WORKERS OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
say "jobs: ${JOBS[*]} | VLM_HEAD_TRAIN_LIMIT=$HEAD_LIMIT | workers $VLM_WORKERS (OMP/MKL $OMP_NUM_THREADS) | results: $RESULTS | Kaggle upload: $([ "$DO_UPLOAD" = 1 ] && echo ON || echo off)"
say "restore check (results already in $RESULTS, e.g. copied back from the PC)"
python "$STATUS_PY" markers --results "$RESULTS" --state "$STATE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [ "$DO_UPLOAD" = 1 ] && [ "${JOBS[*]}" != "preflight" ]; then ask_token; fi
export EZY_RUN_STARTED="$(date +%s)"
QUEUE="$(printf '%s,' "${JOBS[@]}")"; QUEUE="${QUEUE%,}"
[ "${JOBS[*]}" = "preflight" ] || python "$STATUS_PY" status --results "$RESULTS" --queue "$QUEUE"

run_info() {   # provenance next to the results
  {
    echo "job: $1"; echo "date: $(date -u +%FT%TZ)"; echo "host: rented container (EZYCLOUDX Docker GPU)"
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
    echo "VLM_HEAD_TRAIN_LIMIT: $HEAD_LIMIT"
    python - <<'PY'
import importlib.metadata as m, sys
print("python", sys.version.split()[0])
for p in ("torch", "transformers", "flash-linear-attention", "fla-core", "triton", "numpy", "pillow"):
    try: print(p, m.version(p))
    except Exception: print(p, "-")
PY
  } > "$UPLOAD/run_info_$1.txt"
}

stage_upload() {   # copy everything worth keeping into $UPLOAD (the dataset content); model dirs become <key>.zip on Kaggle (dir-mode zip)
  mkdir -p "$UPLOAD"
  local f d key
  for f in "$RESULTS"/*; do [ -f "$f" ] && cp -f "$f" "$UPLOAD/"; done
  for d in "$RESULTS"/*/; do
    key="$(basename "$d")"
    [ -n "$key" ] && [ -f "$d/meta.json" ] || continue
    rm -rf "${UPLOAD:?}/${key:?}"
    cp -a "$d" "$UPLOAD/$key"
    [ "${UPLOAD_SHARDS:-1}" = 1 ] || rm -rf "${UPLOAD:?}/${key:?}/shards"
    rm -f "$UPLOAD/$key/best_state.pt"            # the LoRA jobs keep adapter/ + head.pt (best_state.pt is the same weights)
  done
  [ -f "$UPLOAD/dataset-metadata.json" ] || cat > "$UPLOAD/dataset-metadata.json" <<JSON
{"title": "vivqa-vlm-results", "id": "$RESULTS_DS", "licenses": [{"name": "other"}]}
JSON
}

upload() {   # $1 = job. Returns 0 only when Kaggle shows the dataset 'ready' and lists this job's files.
  local job="$1" msg ok=0 tries st
  stage_upload; run_info "$job"
  msg="$job $(date -u +%FT%TZ) head_limit=$HEAD_LIMIT"
  for tries in 1 2 3; do
    if kaggle datasets status "$RESULTS_DS" >/dev/null 2>&1; then
      kaggle datasets version -p "$UPLOAD" -m "$msg" -r zip && { ok=1; break; }
    else
      kaggle datasets create -p "$UPLOAD" -r zip && { ok=1; break; }
    fi
    echo "upload attempt $tries failed; retrying in 30 s"; sleep 30
  done
  [ "$ok" = 1 ] || return 1
  st=""
  for _ in $(seq 1 90); do                                         # wait up to 15 min for 'ready'
    st="$(kaggle datasets status "$RESULTS_DS" 2>&1 || true)"
    case "$(echo "$st" | tr 'A-Z' 'a-z')" in *ready*) break ;; esac
    sleep "${EZY_POLL_SEC:-10}"
  done
  case "$(echo "$st" | tr 'A-Z' 'a-z')" in *ready*) ;; *) echo "dataset status after the upload: $st"; return 1 ;; esac
  kaggle datasets files "$RESULTS_DS" --page-size 200 2>&1 | tee "$STATE/files_$job.txt" | grep -i "$job" >/dev/null || { echo "the uploaded file list does not mention '$job'"; return 1; }
  return 0
}

job_ok() {   # the kernel's summary says every stage returned 0 and all splits are complete
  python - "$RESULTS/summary_$1.json" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception as e:
    print("no summary:", e); sys.exit(1)
bad = {k: v["returncode"] for k, v in s.get("stages", {}).items() if v.get("returncode") != 0}
print("status:", s.get("status"), "| stage return codes:", {k: v["returncode"] for k, v in s.get("stages", {}).items()})
sys.exit(0 if s.get("status") == "complete" and not bad else 1)
PY
}

CODE_DIR="$(dirname "$(find "$INPUT/viso-code" -name train_viso.py 2>/dev/null | head -1)")"
S1_DIR="$(dirname "$(dirname "$(find "$INPUT/vivqa-s1" -path '*/data/train.jsonl' 2>/dev/null | head -1)")")"
[ -f "$CODE_DIR/scripts/s1_vlm_lora.py" ] || die "scripts/s1_vlm_lora.py is not in viso-code: run setup.sh again (step 5 copies it in)"

run_lora() {   # $1 = lora-img | lora-txt ; the recipe is in scripts/s1_vlm_lora.py (defaults = the approved hyper-parameters)
  local job="$1" mode="${1#lora-}" repo rev lic rc t0 nw
  read -r repo rev lic < <(python "$HERE/pull_models.py" "$KERNEL" --print qwen)
  nw="$VLM_WORKERS"
  t0=$(date +%s)
  VIVQA_S1_ROOT="$S1_DIR" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 python -u "$CODE_DIR/scripts/s1_vlm_lora.py" --mode "$mode" --model "$repo" --revision "$rev" --licence "$lic" \
    --model-dir "$RESULTS/$job" --s1-data "$S1_DIR/data" --root "$S1_DIR" --image-sha "$(sha_list_path "$CODE_DIR")" --image-roots "$INPUT" --workers "$nw" --resume ${LORA_ARGS:-} 2>&1 | tee -a "$RESULTS/${job}_train.log"
  rc=${PIPESTATUS[0]}
  python - "$RESULTS/summary_$job.json" "$rc" "$(( $(date +%s) - t0 ))" "$RESULTS/$job/lora_s0/views.json" <<'PY'
import json, os, sys
path, rc, secs, views = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ok = rc == 0 and os.path.isfile(views)
json.dump({"model_key": os.path.basename(path)[8:-5], "status": "complete" if ok else f"failed (rc {rc})", "stages": {"train_and_predict": {"returncode": rc, "seconds": secs}}, "elapsed_s": secs}, open(path + ".tmp", "w"), indent=1)
os.replace(path + ".tmp", path)
PY
  if ! VIVQA_S1_ROOT="$S1_DIR" python "$CODE_DIR/scripts/s1_vlm_eval.py" report --root-models "$RESULTS" > "$RESULTS/report_$job.log" 2>&1; then
    echo "(report stage failed: the per-job results are still in $RESULTS/$job; last lines of $RESULTS/report_$job.log:)"; tail -5 "$RESULTS/report_$job.log"
  fi
  return "$rc"
}

cleanup_model() {   # MODEL_CLEANUP=1: free the job's Hugging Face cache once its results are safely on Kaggle (never the Qwen3.5-0.8B base, which the lora jobs share)
  [ "${MODEL_CLEANUP:-0}" = 1 ] || return 0
  case "$1" in lora-*|qwen) return 0 ;; esac
  local repo dir
  read -r repo _ < <(python "$HERE/pull_models.py" "$KERNEL" --print "$1") || return 0
  [ -n "$repo" ] || return 0
  dir="$HF_HOME/hub/models--${repo//\//--}"
  if [ -d "$dir" ]; then echo "MODEL_CLEANUP: removing $dir ($(du -sh "$dir" | cut -f1))"; rm -rf "${dir:?}"; fi
}

declare -A RESULT TIMES
for job in "${JOBS[@]}"; do
  if [ "$job" = preflight ]; then
    say "PREFLIGHT: Intern-Decision-0.8B calib features for ${VLM_PREFLIGHT_MIN:-10} min"
    PW="$ROOT/preflight"; rm -rf "${PW:?}"; mkdir -p "$PW"
    VLM_MODEL=intern VLM_PREFLIGHT=1 VLM_PREFLIGHT_MIN="${VLM_PREFLIGHT_MIN:-10}" VLM_INPUT_ROOT="$INPUT" VLM_WORK="$PW" VLM_MAX_HOURS=0 python "$KERNEL"
    python - "$PW/intern/preflight.json" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("PREFLIGHT FAILED:", e); sys.exit(1)
print(f"\nSpeed: {r['units_per_s_after_warmup']} units/s | linear-attention path: {r['linear_attention_path']}")
print(f"Projected time PER SMALL MODEL: full train {r['projected_hours_full_run']} h ({r['units_full_run']} units) | with VLM_HEAD_TRAIN_LIMIT=8000: {r['projected_hours_8000_train_subset']} h ({r['units_with_8000_train_subset']} units)")
print("(the 3 small models cost about 3x that; each reference model is zero-shot only = 17.8k units, so about 0.35x of a full small-model run per size step.)")
PY
    RESULT[$job]="preflight (no upload)"
    continue
  fi
  if [ -f "$STATE/done_$job" ] && [ "${FORCE:-0}" != 1 ]; then
    say "$job: already complete (state/done_$job; FORCE=1 to rerun)"; RESULT[$job]="done earlier"
    python "$STATUS_PY" status --results "$RESULTS" --job "$job" --status "done earlier" >/dev/null; continue
  fi
  say "JOB $job starts"
  T0=$(date +%s)
  python "$STATUS_PY" status --results "$RESULTS" --job "$job" --status running
  if [[ "$job" == lora-* ]]; then
    run_lora "$job"; echo "lora exit code: $?"
  else
  VLM_MODEL="$job" VLM_INPUT_ROOT="$INPUT" VLM_WORK="$RESULTS" VLM_SHARED_WORK=1 VLM_MAX_HOURS="${VLM_MAX_HOURS:-0}" VLM_HEAD_TRAIN_LIMIT="$HEAD_LIMIT" python "$KERNEL"
  echo "kernel exit code: $?"
  fi
  TIMES[$job]=$(( $(date +%s) - T0 ))
  if job_ok "$job"; then fin=complete; else fin=INCOMPLETE; fi
  python "$STATUS_PY" status --results "$RESULTS" --job "$job" --status "$([ "$fin" = complete ] && echo complete || echo incomplete)" --seconds "${TIMES[$job]}"
  if [ "$DO_UPLOAD" != 1 ]; then
    if [ "$fin" = complete ]; then touch "$STATE/done_$job"; RESULT[$job]="complete (results in $RESULTS/$job)"
    else RESULT[$job]="INCOMPLETE (re-run the same command to continue where it stopped)"; fi
    continue
  fi
  say "JOB $job $fin: uploading to $RESULTS_DS"
  if upload "$job"; then
    if [ "$fin" = complete ]; then
      RESULT[$job]="complete, uploaded + confirmed"; touch "$STATE/done_$job"
      cleanup_model "$job"
    else
      RESULT[$job]="INCOMPLETE (partial results uploaded; re-run the same command to continue)"
    fi
  else
    RESULT[$job]="$fin, UPLOAD NOT CONFIRMED"
  fi
done

say "SUMMARY"
all_safe=1
for job in "${JOBS[@]}"; do
  t="${TIMES[$job]:-}"; tt=""; [ -n "$t" ] && tt=$(printf ' | measured %dh%02dm' $((t / 3600)) $((t % 3600 / 60)))
  echo "  $job: ${RESULT[$job]:-not run}$tt"
  case "${RESULT[$job]:-}" in *"uploaded + confirmed"*|"done earlier"|"preflight (no upload)"|complete\ \(results*) ;; *) all_safe=0 ;; esac
done
if [ "$DO_UPLOAD" != 1 ] && [ "${JOBS[*]}" != "preflight" ]; then
  if [ "$all_safe" = 1 ]; then
    printf '\n=====================================================\n  RUN COMPLETE: every job above finished.\n  Results: %s (RUN_STATUS.json lists them).\n  Keep the rental until your pull (rsync) has copied everything.\n=====================================================\n' "$RESULTS"
  else
    printf '\n!!! RUN NOT COMPLETE: at least one job is incomplete (see above). Re-run the same command: finished shards / LoRA checkpoints are kept and finished jobs are skipped.\n'
  fi
elif [ "$all_safe" = 1 ] && [ "${JOBS[*]}" != "preflight" ]; then
  printf '\n=====================================================\n  SAFE TO END RENTAL: every job above is uploaded to\n  https://www.kaggle.com/datasets/%s (private) and confirmed.\n=====================================================\n' "$RESULTS_DS"
elif [ "${JOBS[*]}" = "preflight" ]; then
  echo "(preflight only: nothing was uploaded; this is not a final state)"
else
  printf '\n!!! NOT SAFE TO END THE RENTAL: at least one job is incomplete or its upload is not confirmed (see above). Re-run the same command: finished shards are kept and finished jobs are skipped.\n'
fi
if [ "$DO_UPLOAD" = 1 ]; then
  forget_token
  echo "The Kaggle token was removed from this shell. If it was a temporary token, REVOKE it now at https://www.kaggle.com/settings (API tokens)."
fi
