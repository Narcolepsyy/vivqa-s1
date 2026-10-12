#!/usr/bin/env bash
# One-time setup of the rented GPU container (idempotent: safe to run again, finished steps are skipped).
#   bash setup.sh                    all steps
#   EZY_SKIP="coco models" bash setup.sh      skip steps (names: venv torch pip datasets coco models smoke)
# Needs: Linux, an NVIDIA driver, python3 >= 3.10, internet. Asks ONCE for the Kaggle token (hidden; only in this shell).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
KERNEL="$HERE/vlm_kernel.py"; [ -f "$KERNEL" ] || KERNEL="$HERE/../vlm_kernel.py"
[ -f "$KERNEL" ] || die "vlm_kernel.py not found next to setup.sh"

say "1/9 machine"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this is not a GPU container"
nvidia-smi
DRV_CUDA="$(nvidia-smi | grep -oE 'CUDA (UMD )?Version: [0-9.]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"   # newer drivers print 'CUDA UMD Version: 13.3'
echo "driver supports CUDA ${DRV_CUDA:-unknown}"
python3 --version
python3 -c 'import sys; assert sys.version_info >= (3, 10), "python >= 3.10 is required"' || die "python3 is too old"
echo "work root: $ROOT"
df -h "$ROOT" | tail -1
free -g | sed -n 1,2p
FREE_GB="$(df -Pk "$ROOT" | awk 'NR==2{print int($4/1048576)}')"
{ skip_step coco && skip_step models; } || [ "$FREE_GB" -ge 85 ] || die "only ${FREE_GB} GB free under $ROOT (need ~85 GB: one COCO zip at a time 13 GB + images 6 GB + models ~40 GB + venv ~10 GB)"

say "2/9 virtualenv ($VENV)"
if ! skip_step venv; then
  if [ ! -f "$VENV/bin/activate" ]; then
    python3 -m venv "$VENV" 2>/dev/null || { rm -rf "$VENV"; python3 -m pip install --quiet --user virtualenv && python3 -m virtualenv "$VENV"; } || die "cannot create a virtualenv (try: apt-get install -y python3-venv)"
  fi
fi
activate_venv
python -m pip install --quiet --upgrade pip wheel
python --version

say "3/9 PyTorch (CUDA wheel matching the driver)"
if ! skip_step torch; then
  if python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "torch with a working GPU is already installed"
  else
    TAG="$(cuda_tag "${DRV_CUDA:-12.1}")"
    ok=0
    # pass 1: the tested pair (torch 2.14.1) on every tag the driver can run, newest first (it exists on cu130 and cu126); pass 2: whatever the index offers
    for spec in "torch==2.14.1 torchvision==0.29.1" "torch torchvision"; do
      for t in cu130 cu128 cu126 cu124 cu121 cu118; do
        [ "${t#cu}" -le "${TAG#cu}" ] || continue                                  # never a tag newer than the driver's CUDA
        echo "trying $spec from the $t index"
        # shellcheck disable=SC2086
        if python -m pip install --quiet $spec --index-url "https://download.pytorch.org/whl/$t" 2>/dev/null && python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then ok=1; break 2; fi
      done
    done
    [ "$ok" = 1 ] || die "no PyTorch wheel with a working GPU could be installed"
  fi
  python -c 'import torch; print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))'
fi

say "4/9 python packages"
if ! skip_step pip; then
  # Triton (used by flash-linear-attention) JIT-builds a small C launcher: it needs gcc + Python headers, else fla silently falls back to CPU
  # system packages: gcc + python headers (Triton), rsync (the PC pulls the results with rsync over SSH; the rental image has none), tmux (recommended for long runs)
  APT_NEED=""
  command -v gcc >/dev/null || APT_NEED="$APT_NEED build-essential"
  [ -f "/usr/include/python3.$(python -c 'import sys;print(sys.version_info[1])')/Python.h" ] || APT_NEED="$APT_NEED python3-dev"
  command -v rsync >/dev/null || APT_NEED="$APT_NEED rsync"
  command -v tmux >/dev/null || APT_NEED="$APT_NEED tmux"
  if [ -n "$APT_NEED" ]; then
    echo "apt-get install:$APT_NEED"
    # shellcheck disable=SC2086
    DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $APT_NEED >/dev/null || die "could not install:$APT_NEED (gcc/python3-dev are needed by Triton, rsync by the result pull)"
  fi
  python -m pip install --quiet "transformers==5.19.0" "flash-linear-attention" "peft==0.21.2" pytorch_transformers timm einops yacs termcolor tabulate joblib pillow numpy huggingface_hub
  # Kaggle CLI 2.x (KGAT_ tokens) needs Python >= 3.11: own tiny venv via uv, linked into the main venv
  if ! "$ROOT/kvenv/bin/kaggle" --version >/dev/null 2>&1; then
    [ -x "$ROOT/uvbin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/uvbin" sh >/dev/null 2>&1 || die "could not install uv"
    UV_PYTHON_INSTALL_DIR="$ROOT/uvpy" UV_CACHE_DIR="$ROOT/uvcache" "$ROOT/uvbin/uv" venv -q --python 3.11 "$ROOT/kvenv" && \
    UV_CACHE_DIR="$ROOT/uvcache" "$ROOT/uvbin/uv" pip install -q --python "$ROOT/kvenv/bin/python" kaggle || die "could not install the Kaggle CLI"
  fi
  python -m pip uninstall -y -q kaggle >/dev/null 2>&1 || true
  ln -sf "$ROOT/kvenv/bin/kaggle" "$VENV/bin/kaggle"; kaggle --version
  # causal-conv1d: only a prebuilt wheel is acceptable (a source build takes ~20 min); the torch fallback for the conv is cheap
  if timeout 120 python -m pip install --quiet --only-binary=:all: causal-conv1d >/dev/null 2>&1; then echo "causal-conv1d: installed (prebuilt wheel)"; else echo "causal-conv1d: no prebuilt wheel -> skipped (the torch fallback is fine)"; fi
  python -m pip cache purge >/dev/null 2>&1 || true                                  # the wheel cache (several GB) is not needed on a small disk
  python -c 'import transformers, fla, fla.utils._device as d; print("transformers", transformers.__version__, "| fla", fla.__version__, "| fla backend", d.get_available_device())'
  python -c 'import fla.utils._device as d, sys; sys.exit(0 if d.get_available_device() == "cuda" else 1)' 2>/dev/null || die "flash-linear-attention cannot use the GPU (Triton backend is not cuda)"
fi

say "5/9 Kaggle token + datasets viso-code, vivqa-s1"
if ! skip_step datasets; then
  ask_token
  for ds in viso-code vivqa-s1; do
    if [ -f "$INPUT/$ds/.ok" ]; then echo "$ds: already downloaded"; continue; fi
    rm -rf "${INPUT:?}/$ds"; mkdir -p "$INPUT/$ds"
    kaggle datasets download -d "$OWNER/$ds" -p "$INPUT/$ds" --unzip -q
    sums="$INPUT/$ds/SHA256SUMS.txt"
    # Kaggle strips dataset-metadata.json from downloads and auto-decompresses *.gz (s1_images_sha256.tsv.gz -> .tsv). The readers accept both names; here everything
    # else is verified from SHA256SUMS and the sha list against the hash of its CONTENT (the same for the .gz or the .tsv upload).
    SHA_LIST_CONTENT=b064ac1c975965da864f9bcc14da1886791941d8d003c2e4bfeeab8cd4c475e2
    if [ -f "$sums" ]; then
      (cd "$INPUT/$ds" && grep -vE ' (\./)?(dataset-metadata\.json|.*s1_images_sha256\.tsv(\.gz)?)$' SHA256SUMS.txt | sha256sum -c --quiet -) || die "$ds: SHA256SUMS check failed"
      echo "$ds: SHA256SUMS ok"
    else echo "$ds: no SHA256SUMS.txt in the dataset (skipped check)"; fi
    if [ "$ds" = viso-code ]; then
      SHA_LIST="$(sha_list_path "$(dirname "$(find "$INPUT/viso-code" -name train_viso.py | head -1)")")" || die "image sha list not found in viso-code"
      got="$(python -c 'import sys,gzip,hashlib; b=open(sys.argv[1],"rb").read(); b=gzip.decompress(b) if b[:2]==b"\x1f\x8b" else b; print(hashlib.sha256(b).hexdigest())' "$SHA_LIST")"
      [ "$got" = "$SHA_LIST_CONTENT" ] || die "$SHA_LIST: content hash mismatch ($got)"
      echo "image sha list ok: $SHA_LIST"
    fi
    touch "$INPUT/$ds/.ok"
  done
fi
# overlay: the harness + LoRA scripts shipped next to setup.sh are the reference copies (the Kaggle viso-code may be older)
CODE_DIR="$(dirname "$(find "$INPUT/viso-code" -name train_viso.py 2>/dev/null | head -1)")"
[ -d "$CODE_DIR/scripts" ] && overlay_code
SHA_LIST="$(sha_list_path "$CODE_DIR" 2>/dev/null || true)"

say "6/9 COCO 2014 images: download zips, extract the 34,893 benchmark files, verify every sha256"
if ! skip_step coco; then
  [ -n "$SHA_LIST" ] || die "s1_images_sha256.tsv[.gz] not found under $INPUT/viso-code (step datasets)"
  if [ -f "$INPUT/coco/.verified" ]; then
    echo "COCO images already extracted and verified"
  else
    mkdir -p "$ROOT/coco_zips" "$INPUT/coco"
    for sub in train2014 val2014; do
      [ -f "$INPUT/coco/.done_$sub" ] && { echo "$sub: already extracted and verified"; continue; }
      z="$ROOT/coco_zips/$sub.zip"
      n=0
      until python "$HERE/extract_coco.py" --zip "$z" --check >/dev/null 2>&1; do      # complete zip = readable central directory
        n=$((n + 1)); [ "$n" -le 8 ] || die "could not download $sub.zip"
        echo "downloading $sub.zip (attempt $n; resumes where it stopped)"
        curl -fL --retry 5 --retry-delay 5 --retry-all-errors -C - -o "$z" "http://images.cocodataset.org/zips/$sub.zip" || true
      done
      python "$HERE/extract_coco.py" --sha-list "$SHA_LIST" --zip "$z" --sub "$sub" --out "$INPUT/coco" || die "COCO $sub: verification FAILED, aborting (see above)"
      touch "$INPUT/coco/.done_$sub"
      rm -f "$z"                                                                       # free the disk before the next zip (the rental has ~100 GB)
    done
    touch "$INPUT/coco/.verified"
  fi
  echo "images on disk: $(find "$INPUT/coco" -name 'COCO_*.jpg' | wc -l) (expected 34893)"
fi

say "7/9 Hugging Face models (pinned full revisions from vlm_kernel.py)"
if ! skip_step models; then
  python "$HERE/pull_models.py" "$KERNEL"
fi

say "8/9 smoke test: the whole pipeline on 24 items per split (model: ${SMOKE_MODEL:-intern})"
if ! skip_step smoke; then
  SMOKE="$ROOT/smoke"; rm -rf "${SMOKE:?}"; mkdir -p "$SMOKE"
  LOGF="$SMOKE/smoke.log"
  VLM_DRY=1 VLM_MODEL="${SMOKE_MODEL:-intern}" VLM_INPUT_ROOT="$INPUT" VLM_WORK="$SMOKE/work" VLM_MAX_HOURS=0 timeout 900 python "$KERNEL" 2>&1 | tee "$LOGF" | grep -E "^\[|units/s|linear-attention|Traceback|Error|error" | grep -v "^\[transformers\]" | tail -40 || true
  say "smoke test summary"
  grep -E "linear-attention path|versions:" "$LOGF" || echo "(no 'linear-attention path' line: the features stage did not start - read $LOGF)"
  grep -E "units/s\)" "$LOGF" | tail -3 || true
  echo "(the dry run is tiny and includes Triton compilation warm-up: the real speed is in the heartbeat lines or 'bash run.sh preflight')"
  grep -E '"status"' "$SMOKE/work/summary_${SMOKE_MODEL:-intern}.json" 2>/dev/null || grep -E '"status"' "$SMOKE/work/summary.json" 2>/dev/null || echo "(no summary.json: the smoke test failed - read $LOGF)"
  rm -rf "${SMOKE:?}/work"
  say "8b/9 LoRA smoke test: 24 items, 3 optimizer steps, bf16 (checks the training backward pass on this GPU)"
  S1_DIR="$(dirname "$(dirname "$(find "$INPUT/vivqa-s1" -path '*/data/train.jsonl' | head -1)")")"
  read -r REPO_M REV_M LIC_M < <(python "$HERE/pull_models.py" "$KERNEL" --print qwen)
  VIVQA_S1_ROOT="$S1_DIR" timeout 900 python -u "$CODE_DIR/scripts/s1_vlm_lora.py" --mode img --model "$REPO_M" --revision "$REV_M" --licence "$LIC_M" --model-dir "$SMOKE/lora" \
    --s1-data "$S1_DIR/data" --root "$S1_DIR" --image-sha "$(sha_list_path "$CODE_DIR")" --image-roots "$INPUT" --limit 24 --max-steps 3 --batch 4 --accum 2 --workers 2 --eval-items 12 --eval-points 2 2>&1 \
    | grep -E "^model |LoRA \(|EVAL|training finished|Traceback|Error|error" | cut -c1-200 || true
  rm -rf "${SMOKE:?}/lora"
fi

say "9/9 done"
forget_token
echo "Setup finished. Next:  bash run.sh preflight      (optional, 10 min: measures the real speed)"
echo "                       bash run.sh all      (= intern jpt onejev lora-img lora-txt ref-intern2b ref-intern4b ref-qwen3vl8b, in this order)"
