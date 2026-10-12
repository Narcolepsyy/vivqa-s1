#!/usr/bin/env bash
# Plumbing check of the whole pipeline for every experiment-matrix row (NOT results): 64 type-balanced items per
# split, 2 epochs. train -> viso_predict -> viso_calibrate -> vivqa_s1_v2/evaluate.py, logs under output/viso/tiny_*.
#   bash scripts/viso_tiny_runs.sh [python]     (default python: ~/.venvs/viso/bin/python)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-$HOME/.venvs/viso/bin/python}
export PYTHONDONTWRITEBYTECODE=1
COMMON=(MODEL.DEVICE cpu DATASET.FEATURE_DATASET.LIMIT 64 DATASET.FEATURE_DATASET.BATCH_SIZE 16 TRAINING.EPOCHS_SUP 2
        TRAINING.EPOCHS_RL 2 TRAINING.PATIENCE 0 TRAINING.WARMUP_STEPS 2 TRAINING.SEED 0)
VISO=configs/viso_vivqav2.yaml; FLAT=configs/viso_flat_vivqav2.yaml
S1=../vivqa_s1_v2/data

run() {  # run <name> <config> <opts...>
  local name=$1 cfg=$2; shift 2
  echo "=== $name"
  "$PY" train_viso.py --config-file "$cfg" --opts "${COMMON[@]}" MODEL.NAME "tiny_$name" "$@" 2>&1 | grep -E '"stage"|Task done' || true
  "$PY" scripts/viso_predict.py --config-file "$cfg" --opts "${COMMON[@]}" MODEL.NAME "tiny_$name" "$@" > /dev/null 2>&1
  "$PY" scripts/viso_calibrate.py --run-dir "output/viso/tiny_$name" --s1-dir "$S1" > /dev/null
  "$PY" - "$name" <<'PYEOF'
import json, subprocess, sys
name = sys.argv[1]
run = f"output/viso/tiny_{name}"
ids = [json.loads(l)["id"] for l in open(f"{run}/preds_dev.jsonl")]
gold = {json.loads(l)["id"]: l for l in open("../vivqa_s1_v2/data/dev.jsonl")}
open(f"{run}/gold_dev.jsonl", "w").write("".join(gold[i] for i in ids))
for variant, path in (("uncal", f"{run}/preds_dev.jsonl"), ("pertype", f"{run}/calibration/preds_dev_pertype.jsonl")):
    r = subprocess.run([sys.executable, "../vivqa_s1_v2/evaluate.py", "--gold", f"{run}/gold_dev.jsonl", "--pred", path,
                        "--out-json", f"{run}/eval_dev_{variant}.json"], capture_output=True, text=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1"})
    m = json.load(open(f"{run}/eval_dev_{variant}.json"))["primary"]["all"]
    print(f"evaluate.py[{variant}] rc={r.returncode} missing={m['missing']} n={m['n']} acc={m['accuracy']:.4f} nll={m['nll']:.4f} ece={m['ece']:.4f}")
PYEOF
}

run A0 "$FLAT" LOSS.NAME soft_ce
run A1 "$VISO" LOSS.NAME hard_ce
run A2 "$VISO" LOSS.NAME soft_ce
run A3 "$VISO" LOSS.NAME rlcd LOSS.STAGE all          # A4 = A3 + per-type T + escalation head (calibration/ dir of A3)
run A5 "$VISO" LOSS.NAME soft_ce MODEL.ZERO_VISION True
run A6 "$VISO" LOSS.NAME soft_ce MODEL.SET_ATTENTION False
run A2_repeat "$VISO" LOSS.NAME soft_ce               # determinism: identical logits to A2 (sha256 below)
echo "=== determinism (sha256 of logits)"
for s in dev test; do sha256sum output/viso/tiny_A2/logits_$s.npz output/viso/tiny_A2_repeat/logits_$s.npz; done
echo "=== efficiency"
"$PY" -m utils.viso_efficiency --config-file "$VISO" "$FLAT" --out output/viso/tiny_efficiency.json > /dev/null
echo done
