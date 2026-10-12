import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import viso_calibrate  # noqa: E402
import viso_predict  # noqa: E402
import viso_report  # noqa: E402

from utils import viso_calibration as C  # noqa: E402
from utils.viso_s1 import S1_ROOT, read_jsonl  # noqa: E402

S1 = "../vivqa_s1_v2/data"


def balanced(split, n_per_type=20):
    by = {"noul": [], "score": [], "choice": []}
    for it in read_jsonl(f"{S1}/{split}.jsonl"):
        t = it["questions"]["q"]["type"]
        if len(by[t]) < n_per_type:
            by[t].append(it)
    return by["noul"] + by["score"] + by["choice"]


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory):
    """A run dir with synthetic logits (gold-aligned signal + noise) for calib/dev/test."""
    run = tmp_path_factory.mktemp("run")
    rng = np.random.RandomState(0)
    for split in ("calib", "dev", "test"):
        items = balanced(split)
        g = C.gold_arrays(items)
        z = g["target"] * 3.0 + rng.randn(*g["target"].shape) * 1.5
        z = np.where(g["space_mask"], z, -1e9)
        C.save_logits(run / f"logits_{split}.npz", g["ids"], g["type_id"], z, g["space_mask"])
    return run


def test_calibrate_script_outputs(fake_run):
    out = viso_calibrate.main(["--run-dir", str(fake_run), "--s1-dir", S1])
    temps = json.load(open(os.path.join(out, "temperatures.json")))
    assert all(0.1 <= t <= 10 for t in temps["per_type_nll"].values()) and 0.1 <= temps["global_nll"] <= 10
    for v in viso_calibrate.VARIANTS:
        for split in ("dev", "test"):
            assert os.path.isfile(os.path.join(out, f"preds_{split}_{v}.jsonl"))
    esc = json.load(open(os.path.join(out, "escalation.json")))
    for part in ("calib_oof", "dev", "test"):
        assert set(esc[part]) >= {"auroc_head", "auroc_max_prob", "selective_head", "selective_max_prob"}
        assert "sel_acc@0.5" in esc[part]["selective_head"] and "aurc" in esc[part]["selective_max_prob"]
    m = json.load(open(os.path.join(out, "calibration_metrics.json")))
    # per-type NLL temperatures are fitted on calib to minimise NLL there: the calibrated calib NLL cannot be worse
    assert m["calib"]["pertype"]["primary"]["all"]["nll"] <= m["calib"]["uncal"]["primary"]["all"]["nll"] + 1e-9


def test_calibrated_predictions_pass_s1_evaluate_and_report(fake_run, tmp_path):
    out = viso_calibrate.main(["--run-dir", str(fake_run), "--s1-dir", S1])
    items = balanced("test")
    gold = tmp_path / "gold.jsonl"
    with open(gold, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    r = subprocess.run([sys.executable, os.path.join(S1_ROOT, "evaluate.py"), "--gold", str(gold), "--pred",
                        os.path.join(out, "preds_test_pertype.jsonl"), "--out-json", str(tmp_path / "m.json")],
                       capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0 and "WARNING" not in r.stderr, r.stderr
    assert json.load(open(tmp_path / "m.json"))["primary"]["all"]["missing"] == 0
    viso_report.main(["--runs", f"X={fake_run}:pertype", f"Y={fake_run}:uncal", "--split", "test", "--s1-dir", S1,
                      "--out", str(tmp_path / "report.md")])
    md = open(tmp_path / "report.md", encoding="utf-8").read()
    assert "| X |" in md and "| Y |" in md and "Sources:" in md


def test_efficiency_profile_keys(vocab):
    import models.viso  # noqa: F401
    from builders.model_builder import build_model
    from tests.viso.test_models import model_cfg
    from utils import viso_efficiency as E
    res = {}
    for arch in ("ViSO", "FlatS1"):
        m = build_model(model_cfg(arch), vocab)
        res[arch] = E.profile(m, vocab, threads=(1,), warmup=1, runs=3)
    v, f = res["ViSO"], res["FlatS1"]
    assert v["params"]["total"] == v["params"]["encoder"] + v["params"]["heads_and_classifier"]
    assert "choice_head" in v["params"] and "flat_classifier" in f["params"]
    assert v["flops_batch1"] > 0 and f["flops_batch1"] > 0 and v["cpu_latency"]["1_threads"]["median_ms"] > 0
    assert f["params"]["flat_classifier"] == 512 * vocab.total_answers + vocab.total_answers


def test_predict_script_writes_logits_and_preds(tmp_path):
    from tests.viso.test_task import make_task
    task = make_task(tmp_path, "ViSO", limit=16, batch=16, epochs=1)
    task.start()
    opts = ["MODEL.DEVICE", "cpu", "MODEL.NAME", "t_ViSO_soft_ce", "TRAINING.CHECKPOINT_PATH", str(tmp_path / "ck"),
            "TRAINING.OUTPUT_PATH", str(tmp_path / "out"), "DATASET.FEATURE_DATASET.LIMIT", "16",
            "DATASET.FEATURE_DATASET.BATCH_SIZE", "16"]
    out = viso_predict.main(["--config-file", "configs/viso_vivqav2.yaml", "--opts", *opts, "--splits", "dev"])
    assert os.path.isfile(os.path.join(out, "logits_dev.npz")) and os.path.isfile(os.path.join(out, "preds_dev.jsonl"))
    rows = read_jsonl(os.path.join(out, "preds_dev.jsonl"))
    assert all(abs(sum(r["probabilities"].values()) - 1) < 1e-6 for r in rows)
