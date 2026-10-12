import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import data_utils.datasets.s1_dataset  # noqa: F401
import models.viso  # noqa: F401
import tasks.typed_decision_task as T
from builders.task_builder import build_task
from train_viso import load_config
from utils import viso_calibration as calib
from utils.viso_s1 import S1_ROOT


def zero_dropout(cfg):
    for k in list(cfg.keys()):
        if hasattr(cfg[k], "keys"):
            zero_dropout(cfg[k])
        elif "DROPOUT" in k:
            cfg[k] = 0.0


def make_task(tmp_path, arch="ViSO", name="soft_ce", limit=32, batch=16, lr_enc=2e-4, epochs=26, dropout=True, **extra):
    base = "configs/viso_flat_vivqav2.yaml" if arch == "FlatS1" else "configs/viso_vivqav2.yaml"
    opts = ["MODEL.DEVICE", "cpu", "MODEL.NAME", f"t_{arch}_{name}", "TRAINING.CHECKPOINT_PATH", str(tmp_path / "ck"),
            "TRAINING.OUTPUT_PATH", str(tmp_path / "out"), "DATASET.FEATURE_DATASET.LIMIT", str(limit),
            "DATASET.FEATURE_DATASET.BATCH_SIZE", str(batch), "TRAINING.LR_ENCODER", str(lr_enc),
            "TRAINING.LR_HEADS", "0.001", "TRAINING.WARMUP_STEPS", "2", "TRAINING.EPOCHS_SUP", str(epochs),
            "TRAINING.EPOCHS_RL", "2", "TRAINING.PATIENCE", "0", "LOSS.NAME", name]
    for k, v in extra.items():
        opts += [k, str(v)]
    cfg = load_config(base, opts)
    if not dropout:
        zero_dropout(cfg.MODEL)
    return build_task(cfg)


# (limit, epochs, lr_enc, lr_heads) per model: settings found by probing so that the 50-ish step budget halves the
# loss deterministically (ViSO: 16 items / 50 steps; FlatS1 has a 3,074-way softmax and needs larger learning rates).
OVERFIT = {"ViSO": (16, 50, 2e-4, 1e-3), "FlatS1": (32, 20, 5e-4, 3e-3)}


@pytest.mark.parametrize("arch", ["ViSO", "FlatS1"])
def test_overfit_tiny_subset_soft_ce_loss_halves(tmp_path, arch):
    limit, epochs, lr_enc, lr_heads = OVERFIT[arch]
    task = make_task(tmp_path, arch, limit=limit, epochs=epochs, lr_enc=lr_enc, dropout=False,
                     **{"TRAINING.LR_HEADS": lr_heads})
    before = task.mean_loss("train", "soft_ce")
    task.start()
    after = task.mean_loss("train", "soft_ce")
    assert np.isfinite(before) and after <= 0.5 * before, (before, after)


def test_rlcd_stage_runs_without_nan_and_logs_reward(tmp_path):
    task = make_task(tmp_path, "ViSO", "rlcd", limit=32, batch=16, epochs=1, **{"LOSS.STAGE": "all"})
    task.start()
    rows = [json.loads(l) for l in open(task.log_file, encoding="utf-8")]
    rl = [r for r in rows if r.get("stage") == "rl"]
    assert len(rl) == 2 and all(np.isfinite(r["train_loss"]) and r["mean_reward"] is not None for r in rl)
    assert os.path.isfile(os.path.join(task.checkpoint_path, "best_model_stage1.pth"))


def test_flat_rejects_rlcd(tmp_path):
    task = make_task(tmp_path, "FlatS1", "soft_ce", limit=16, epochs=1)
    with pytest.raises(ValueError):
        task.item_loss({"logits": None}, None, "rlcd")


def test_predictions_pass_through_s1_evaluate_with_no_missing(tmp_path):
    task = make_task(tmp_path, "ViSO", limit=64, epochs=1)
    task.start()
    task.get_predictions()
    ds = task.datasets["dev"]
    lg = calib.load_logits(os.path.join(task.output_path, "logits_dev.npz"))
    assert lg["logits"].shape == (len(ds), 16) and lg["ids"].tolist() == [it["id"] for it in ds.items]
    probs = calib.probs_from_logits(lg["logits"], lg["space_mask"])
    gold, pred = tmp_path / "gold.jsonl", tmp_path / "preds.jsonl"
    with open(gold, "w", encoding="utf-8") as f:
        for it in ds.items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    calib.write_preds_jsonl(str(pred), ds.items, probs)
    r = subprocess.run([sys.executable, os.path.join(S1_ROOT, "evaluate.py"), "--gold", str(gold), "--pred", str(pred),
                        "--out-json", str(tmp_path / "m.json")], capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, r.stderr
    m = json.load(open(tmp_path / "m.json"))
    assert m["primary"]["all"]["missing"] == 0 and m["primary"]["all"]["n"] == len(ds)
    assert "WARNING" not in r.stderr


def test_logits_are_deterministic_for_the_same_seed(tmp_path):
    outs = []
    for run in range(2):
        d = tmp_path / f"r{run}"
        task = make_task(d, "ViSO", limit=32, batch=16, epochs=2)
        task.start()
        outs.append(T.collect_logits(task.model, task.datasets["dev"], 16, task.device)["logits"])
    assert np.array_equal(outs[0], outs[1])
