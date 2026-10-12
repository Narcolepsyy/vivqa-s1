#!/usr/bin/env bash
# Shared helpers for setup.sh / run.sh (EZYCLOUDX "Docker GPU" rental). Sourced, not run.
# Layout: everything lives under $ROOT (/workspace/vivqa, else ~/vivqa; NEVER /tmp). EZY_ROOT overrides.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${EZY_ROOT:-}" ]; then ROOT="$EZY_ROOT"
elif [ -d /workspace ] && [ -w /workspace ]; then ROOT=/workspace/vivqa
else ROOT="$HOME/vivqa"; fi
case "$ROOT" in /tmp|/tmp/*) echo "refusing to work under /tmp ($ROOT): it may be RAM-backed" >&2; exit 1;; esac
VENV="$ROOT/venv"
INPUT="$ROOT/input"           # viso-code, vivqa-s1, coco/{train2014,val2014}: the same layout as /kaggle/input in the kernels
RESULTS="$ROOT/results"       # VLM_WORK of every job (shared; files prefixed by the model key)
UPLOAD="$ROOT/upload"         # staging folder = the content of the Kaggle results dataset
STATE="$ROOT/state"
OWNER="narcolepsyy"
RESULTS_DS="$OWNER/vivqa-vlm-results"
mkdir -p "$ROOT" "$INPUT" "$RESULTS" "$STATE" "$ROOT/tmp" "$ROOT/hf" "$ROOT/pip-cache" "$ROOT/triton"
export HF_HOME="$ROOT/hf" TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/pip-cache" TRITON_CACHE_DIR="$ROOT/triton" PYTHONDONTWRITEBYTECODE=1

say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

skip_step() { case " ${EZY_SKIP:-} " in *" $1 "*) return 0 ;; esac; return 1; }

# "12.8" -> cu128 (the newest PyTorch wheel tag the driver can run)
cuda_tag() {
  local major="${1%%.*}" minor="${1#*.}"
  minor="${minor%%.*}"
  local v=$((major * 100 + minor))
  if   [ "$v" -ge 1300 ]; then echo cu130
  elif [ "$v" -ge 1208 ]; then echo cu128
  elif [ "$v" -ge 1206 ]; then echo cu126
  elif [ "$v" -ge 1204 ]; then echo cu124
  elif [ "$v" -ge 1201 ]; then echo cu121
  else echo cu118; fi
}

activate_venv() {
  [ -f "$VENV/bin/activate" ] || die "venv missing: run setup.sh first"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
}

# The Kaggle token is asked with a hidden prompt and exists ONLY as an exported variable of this shell. Never written, never echoed.
ask_token() {
  if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -z "${KAGGLE_KEY:-}" ]; then
    local tok="" usr=""
    printf 'Kaggle API token (typing is hidden; kept only in this shell, never written to disk): ' >&2
    IFS= read -rs tok </dev/tty || die "no terminal to read the token from"
    printf '\n' >&2
    [ -n "$tok" ] || die "empty token"
    case "$tok" in
      KGAT_*) export KAGGLE_API_TOKEN="$tok" ;;
      *) printf 'This looks like a legacy API key. Kaggle username: ' >&2
         IFS= read -r usr </dev/tty
         export KAGGLE_USERNAME="$usr" KAGGLE_KEY="$tok" ;;
    esac
    tok=""
  fi
  if ! kaggle datasets status "$OWNER/viso-code" >/dev/null 2>&1; then
    die "Kaggle did not accept the token (or no network). Check the token and that it can read $OWNER/viso-code."
  fi
  say "Kaggle token accepted (not shown, not saved)"
}

# Kaggle AUTO-DECOMPRESSES uploaded *.gz files: a downloaded viso-code has kaggle/s1_images_sha256.tsv, the repo has .tsv.gz (same content). Prints whichever exists under $1/kaggle.
sha_list_path() {
  local d="$1/kaggle" f
  for f in "$d/s1_images_sha256.tsv.gz" "$d/s1_images_sha256.tsv"; do [ -f "$f" ] && { echo "$f"; return 0; }; done
  return 1
}

# Copy the reference versions of the harness / ViSO-FT files over the downloaded viso-code (the Kaggle copy may be older). Idempotent. The list is overlay_files.txt (paths relative to ViVQAv2-main);
# the files come from <ezy>/overlay/<path> (the bundle) or, in the repository, from ../../<path>.
overlay_code() {
  local code rel src n=0
  code="$(dirname "$(find "$INPUT/viso-code" -name train_viso.py 2>/dev/null | head -1)")"
  [ -d "$code/scripts" ] || { echo "overlay: viso-code not found under $INPUT (run setup.sh first)"; return 1; }
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    src="$HERE/overlay/$rel"; [ -f "$src" ] || src="$HERE/../../$rel"
    if [ -f "$src" ]; then mkdir -p "$code/$(dirname "$rel")"; cp -f "$src" "$code/$rel" && n=$((n + 1)); else echo "overlay: $rel NOT FOUND next to the scripts"; fi
  done < "$HERE/overlay_files.txt"
  echo "overlay: $n files copied into $code"
}

forget_token() { unset KAGGLE_API_TOKEN KAGGLE_KEY KAGGLE_USERNAME; }
