"""Phase 2c report additions (scripts/viso_views_report.py): Image-unseen column, seen-vs-unseen gap, agreement bins, efficiency rows. Uses the real v2 benchmark (read-only) and a
'predict the gold distribution' stand-in run, so every number can be checked against the view sizes."""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import viso_views_report as VR  # noqa: E402

S1V2 = "../vivqa_s1_v2/data"
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(S1V2, "eval_flags.jsonl")), reason="the v2 benchmark is not next to the repo")


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    run = d / "A2_s0"
    run.mkdir()
    with open(run / "preds_test.jsonl", "w", encoding="utf-8") as f:
        for l in open(os.path.join(S1V2, "test.jsonl"), encoding="utf-8"):
            it = json.loads(l)
            f.write(json.dumps({"id": it["id"], "probabilities": it["gold"]["q"]["probabilities"]}) + "\n")
    manifest = {"methods": {"A2": {"family": "A2", "variant": "uncal"}}, "jobs": [{"id": "A2_s0", "method": "A2", "seed": 0}]}
    return manifest, str(d)


def test_view_sizes_and_bins_add_up():
    v = VR.split_gold(S1V2)
    assert len(v["all"]) == 4409 and len(v["core"]) < len(v["all"]) and len(v["image_unseen"]) < len(v["core"])
    assert len(v["core_seen"]) + len(v["image_unseen"]) == len(v["core"])
    assert sum(len(b) for b in VR.bin_items(v["core"])) == len(v["core"])


def test_markdown_has_the_four_sections_and_consistent_numbers(fake_run):
    manifest, runs = fake_run
    veff = {"params_total": 92930304, "params_encoder": 85843200, "params_map_head": 7087104, "flops_batch1": 44405096448, "cpu_latency_ms_median_batch1": 455.1, "cpu_threads": 4}
    eff = {"ViSO": {"params": {"total": 10_000_000, "encoder": 8_000_000, "heads_and_classifier": 2_000_000}, "flops_batch1": 1.0e9}}
    md = VR.views_markdown(manifest, runs, S1V2, "test", veff, eff)
    for h in ("## Views:", "## Seen-vs-unseen gap", "## Agreement bins", "## Efficiency rows"):
        assert h in md
    assert "Core 4335" in md and "92,930,304" in md and "tower + ViSO" in md and "102,930,304" in md
    row = [l for l in md.splitlines() if l.startswith("| A2 |")]
    assert len(row) == 3                                               # views, gap, agreement bins
    acc = float(re.findall(r"\d\.\d+", row[0])[0])
    assert 0.9 < acc <= 1.0                                           # a model that predicts the gold distribution is (almost) always right


def test_missing_runs_give_no_method_rows(tmp_path):
    manifest = {"methods": {"A2": {"family": "A2", "variant": "uncal"}}, "jobs": [{"id": "A2_s0", "method": "A2", "seed": 0}]}
    md = VR.views_markdown(manifest, str(tmp_path), S1V2, "test")
    assert "| A2 |" not in md and "## Agreement bins" in md
