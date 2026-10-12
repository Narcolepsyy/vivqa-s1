"""--view / --flags in scripts/viso_bootstrap.py and scripts/viso_report.py (v1.1 release)."""
import hashlib
import json
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "scripts")))
sys.path.insert(0, HERE)
import viso_bootstrap as B  # noqa: E402
import viso_report as R  # noqa: E402
import view_case as V  # noqa: E402

FLAGS_PY = os.path.join(HERE, "golden", "flags_v11.py")                  # the v1.1 flags.py frozen as a test fixture (the view machinery is tested with the v1.1 flag set; the v1.x folders no longer exist)
needs_flags_py = pytest.mark.skipif(not os.path.isfile(FLAGS_PY), reason="tests/viso/golden/flags_v11.py not found")
GOLDEN = os.path.join(HERE, "golden")


def with_flags(s1_data, rows):
    """Install the v1.1 s1bench/flags.py next to the fake data dir and write eval_flags.jsonl; returns the flags path."""
    root = os.path.dirname(s1_data)
    os.makedirs(os.path.join(root, "s1bench"), exist_ok=True)
    shutil.copy(FLAGS_PY, os.path.join(root, "s1bench", "flags.py"))
    path = os.path.join(s1_data, "eval_flags.jsonl")
    with open(path, "w") as f:
        for i, fl in rows.items():
            f.write(json.dumps({"id": i, "split": "test", "flags": fl, "evidence": {}}) + "\n")
    return path


FLAGGED = {5: ["audit_gold_conflict"], 6: ["audit_gold_conflict", "image_seen"], 7: ["gold_wrong_human"], 8: ["text_answer_llm"],
           9: ["image_seen"], 10: ["image_seen"], 11: ["text_answer_confirmed"]}


def test_default_view_is_byte_identical_to_the_pre_view_output(tmp_path):
    manifest, runs, s1 = V.build_case(tmp_path)
    boot = B.run(manifest, runs, s1, resamples=100)
    out = tmp_path / "bootstrap.json"
    with open(out, "w") as f:
        json.dump(boot, f, indent=1)
    assert out.read_bytes() == open(os.path.join(GOLDEN, "bootstrap_default.json"), "rb").read()
    md = R.results_markdown(str(tmp_path), manifest, boot, {}, s1)
    assert md.encode("utf-8") == open(os.path.join(GOLDEN, "results_default.md"), "rb").read()
    assert "view" not in boot and "View:" not in md


@needs_flags_py
def test_core_view_drops_exactly_the_flagged_ids(tmp_path):
    manifest, runs, s1 = V.build_case(tmp_path)
    fp = with_flags(s1, FLAGGED)
    sha = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    all_ = B.run(manifest, runs, s1, resamples=50)
    core = B.run(manifest, runs, s1, resamples=50, view="core")
    unseen = B.run(manifest, runs, s1, resamples=50, view="image_unseen")
    assert all_["n_items"] == 400 and "view" not in all_
    excluded = {5, 6, 7, 11}                                     # audit_gold_conflict, gold_wrong_human, text_answer_confirmed
    assert core["n_items"] == 400 - len(excluded) and unseen["n_items"] == 400 - len(excluded | {9, 10})
    assert core["view"] == "core" and core["flags_sha256"] == sha and core["n_gold_view"] == 396 and core["n_gold_split"] == 400
    assert core["excluded_by_flag"]["core"] == {"audit_gold_conflict": 2, "text_answer_confirmed": 1, "gold_wrong_human": 1}
    assert unseen["excluded_by_flag"]["image_unseen"] == {"image_seen": 2}      # 6 is already out of Core
    # the scored items are exactly the unflagged ones: recompute the accuracy of one method directly
    gold = {it["id"]: it for it in V.noul_items()}
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(runs, "FA_s0", "preds_test.jsonl"))}
    keep = [gold[i] for i in sorted(set(gold) - excluded)]
    ref = B.point_metrics(B.scores_for(keep, preds))
    assert core["methods"]["A"]["seeds"]["0"]["point"]["accuracy"] == pytest.approx(ref["accuracy"], abs=1e-12)
    assert core["methods"]["A"]["seeds"]["0"]["point"]["n_human"] == ref["n_human"]
    md = R.results_markdown(str(tmp_path), manifest, core, {}, s1)
    assert f"View: **core** (396 of 400 gold items; scored n = 396)" in md and sha in md


@needs_flags_py
def test_empty_flags_make_core_equal_all(tmp_path):
    manifest, runs, s1 = V.build_case(tmp_path)
    with_flags(s1, {})
    a = B.run(manifest, runs, s1, resamples=50)
    c = B.run(manifest, runs, s1, resamples=50, view="core")
    assert a["methods"] == c["methods"] and a["comparisons"] == c["comparisons"] and a["n_items"] == c["n_items"]


def test_core_view_without_flags_file_fails_clearly(tmp_path):
    manifest, runs, s1 = V.build_case(tmp_path)
    with pytest.raises(FileNotFoundError, match="--view core needs a flags file"):
        B.run(manifest, runs, s1, resamples=10, view="core")
    with pytest.raises(FileNotFoundError, match="--view image_unseen"):
        B.run(manifest, runs, s1, resamples=10, view="image_unseen", flags_path=str(tmp_path / "nope.jsonl"))
    man = tmp_path / "m.json"
    man.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit) as e:                           # the CLI turns it into an argparse error (exit code 2)
        B.main(["--manifest", str(man), "--runs-dir", runs, "--s1-dir", s1, "--view", "core", "--out", str(tmp_path / "o.json"), "--resamples", "10"])
    assert e.value.code == 2


def test_missing_flags_py_in_benchmark_is_a_clear_error(tmp_path):
    manifest, runs, s1 = V.build_case(tmp_path)
    (tmp_path / "f.jsonl").write_text("")
    with pytest.raises(ImportError, match="no s1bench/flags.py"):
        B.run(manifest, runs, s1, resamples=10, view="core", flags_path=str(tmp_path / "f.jsonl"))


def test_prediction_ids_outside_the_gold_split_still_raise(tmp_path):
    manifest, runs, s1 = V.build_case(tmp_path)
    with open(os.path.join(runs, "FA_s0", "preds_test.jsonl"), "a") as f:
        f.write(json.dumps({"id": 9999, "probabilities": {"true": 1.0}}) + "\n")
    for fam, s in (("FA", 1), ("FB", 0), ("FB", 1)):
        with open(os.path.join(runs, f"{fam}_s{s}", "preds_test.jsonl"), "a") as f:
            f.write(json.dumps({"id": 9999, "probabilities": {"true": 1.0}}) + "\n")
    with pytest.raises(KeyError, match="missing from the gold split"):
        B.run(manifest, runs, s1, resamples=10)


@needs_flags_py
def test_report_script_scores_only_the_view(tmp_path):
    manifest, runs, s1 = V.build_case(tmp_path)
    with_flags(s1, FLAGGED)
    run_dir = os.path.join(runs, "FA_s0")
    R.main(["--runs", f"X={run_dir}", "--split", "test", "--s1-dir", s1, "--out", str(tmp_path / "all.md")])
    R.main(["--runs", f"X={run_dir}", "--split", "test", "--s1-dir", s1, "--view", "core", "--out", str(tmp_path / "core.md")])
    all_md, core_md = (tmp_path / "all.md").read_text(), (tmp_path / "core.md").read_text()
    assert "| X | 400 |" in all_md and "View:" not in all_md
    assert "| X | 396 |" in core_md and "View: core (396 of 400 gold items)" in core_md
    with pytest.raises(SystemExit) as e:
        R.main(["--runs", f"X={run_dir}", "--split", "test", "--s1-dir", s1, "--view", "core", "--flags", str(tmp_path / "x.jsonl"),
                "--out", str(tmp_path / "bad.md")])
    assert e.value.code == 2
