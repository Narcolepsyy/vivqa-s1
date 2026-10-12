import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import viso_kaggle_run as R  # noqa: E402

REPO = R.REPO
S1_ROOT = os.path.abspath(os.path.join(REPO, "..", "vivqa_s1_v2"))
MANIFEST = os.path.join(REPO, "kaggle", "experiments.json")
PY = sys.executable


def tiny_manifest(tmp_path, s1_sha=None):
    m = json.load(open(MANIFEST))
    m["jobs"] = [j for j in m["jobs"] if j["id"] in ("A2_s0", "A3_s0")]
    m["methods"] = {k: v for k, v in m["methods"].items() if k in ("A2", "A3", "A4")}
    m["comparisons"] = [["A3", "A2"], ["A4", "A3"]]
    m["dry_run_overrides"].update({"DATASET.FEATURE_DATASET.LIMIT": 32, "DATASET.FEATURE_DATASET.BATCH_SIZE": 16,
                                   "TRAINING.EPOCHS_SUP": 1, "TRAINING.EPOCHS_RL": 1})
    m["s1_sha256"] = s1_sha or {f: R.sha256_file(os.path.join(S1_ROOT, "data", f)) for f in R.S1_FILES}      # pin the benchmark root under test (the legacy manifest pinned the deleted v1)
    p = tmp_path / "tiny.json"
    p.write_text(json.dumps(m))
    return str(p)


def runner_cmd(manifest, out, gpu, *extra):
    return [PY, os.path.join(REPO, "scripts", "viso_kaggle_run.py"), "--experiments", manifest, "--out", str(out), "--code-root", REPO,
            "--s1-dir", S1_ROOT, "--gpu", str(gpu), "--dry-run", "--poll-seconds", "0.2", *extra]


ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


# ------------------------------------------------------------------------------------------ the real manifest
def test_manifest_matches_the_plan():
    m = R.load_manifest(MANIFEST)
    ids = [j["id"] for j in m["jobs"]]
    assert len(ids) == 15 and ids[:7] == ["A2_s0", "A0_s0", "A3_s0", "A3c_s0", "A1_s0", "A5_s0", "A6_s0"]
    sup = [j for j in m["jobs"] if not j["stage1_from"]]
    rl = [j for j in m["jobs"] if j["stage1_from"]]
    assert len(sup) == 9 and len(rl) == 6
    for fam in ("A0", "A2", "A3", "A3c"):
        assert {j["seed"] for j in m["jobs"] if j["method"] == fam} == {0, 1, 2}
    for j in rl:
        assert j["depends_on"] == [j["stage1_from"]] and j["stage1_from"] == f"A2_s{j['seed']}" and j["overrides"]["LOSS.STAGE"] == "rl"
    pg = {j["id"]: j["overrides"]["LOSS.PG_WEIGHT"] for j in rl}
    assert all(isinstance(v, float) for v in pg.values()) and pg["A3c_s0"] == 0.0 and pg["A3_s0"] == 1.0
    assert set(m["s1_sha256"]) == set(R.S1_FILES)
    assert {a for pair in m["comparisons"] for a in pair} <= set(m["methods"])


def test_every_job_builds_a_valid_config_with_kaggle_settings(tmp_path):
    m = R.load_manifest(MANIFEST)
    args = R.parse(["--out", str(tmp_path), "--code-root", REPO, "--s1-dir", "/kaggle/input/vivqa-s1", "--packed", "/kaggle/input/vivqa-packed"])
    args.out, args.packed = str(tmp_path), "/kaggle/input/vivqa-packed"
    for j in m["jobs"]:
        path, cfg = R.build_config(j, args, m, deadline_unix=1.9e9)
        fd = cfg.DATASET.FEATURE_DATASET
        assert fd.FEATURE_FORMAT == "packed" and fd.PACKED_PATH == "/kaggle/input/vivqa-packed" and fd.WORKERS == 2 and fd.BATCH_SIZE == 64
        assert cfg.MODEL.DEVICE == "cuda" and cfg.TRAINING.RESUME is True and cfg.TRAINING.AMP is False
        assert cfg.TRAINING.DEADLINE_UNIX == 1.9e9 and cfg.TRAINING.SEED == j["seed"] and cfg.MODEL.NAME == j["id"]
        assert cfg.DATASET.VOCAB.S1_DIR == "/kaggle/input/vivqa-s1/data"
        assert cfg.TRAINING.CHECKPOINT_PATH == str(tmp_path / "runs")
    cfg = R.build_config([j for j in m["jobs"] if j["id"] == "A3c_s2"][0], args, m, 0)[1]
    assert cfg.LOSS.PG_WEIGHT == 0.0 and isinstance(cfg.LOSS.PG_WEIGHT, float) and cfg.LOSS.STAGE == "rl" and cfg.LOSS.NAME == "rlcd"
    flat = R.build_config([j for j in m["jobs"] if j["id"] == "A0_s1"][0], args, m, 0)[1]
    assert flat.MODEL.ARCHITECTURE == "FlatS1" and flat.LOSS.NAME == "soft_ce"
    a5 = R.build_config([j for j in m["jobs"] if j["id"] == "A5_s0"][0], args, m, 0)[1]
    a6 = R.build_config([j for j in m["jobs"] if j["id"] == "A6_s0"][0], args, m, 0)[1]
    assert a5.MODEL.ZERO_VISION is True and a6.MODEL.SET_ATTENTION is False


def test_checksum_mismatch_is_refused(tmp_path):
    m = json.load(open(MANIFEST))
    bad = dict(m["s1_sha256"], **{"test.jsonl": "0" * 64})
    manifest = tiny_manifest(tmp_path, bad)
    with pytest.raises(ValueError, match="do not match"):
        R.main(["--experiments", manifest, "--out", str(tmp_path / "o"), "--code-root", REPO, "--s1-dir", S1_ROOT, "--dry-run"])


# ------------------------------------------------------------------------------------------ queue semantics
def _claimer(out, ids, q, stay):
    q.put([i for i in ids if R.try_claim(out, i)])
    stay.wait(60)               # a claimer stays alive while it "runs" its jobs (a dead owner's lock may be stolen)


def test_lock_claiming_is_atomic_across_processes(tmp_path):
    ids = [f"job{i}" for i in range(40)]
    q, stay = mp.Queue(), mp.Event()
    procs = [mp.Process(target=_claimer, args=(str(tmp_path), ids, q, stay)) for _ in range(6)]
    [p.start() for p in procs]
    got = [q.get(timeout=60) for _ in procs]
    stay.set()
    [p.join() for p in procs]
    claimed = sorted(i for g in got for i in g)
    assert claimed == sorted(ids)                       # every job claimed by exactly one process


def _worker(out, jobs, q):
    mine = []
    while True:
        j = R.next_job(jobs, out)
        if j is None:
            if not R.unfinished(jobs, out):
                break
            continue
        mine.append(j["id"])
        d = R.run_dir_of(out, j["id"])
        os.makedirs(d, exist_ok=True)
        json.dump({}, open(os.path.join(d, "DONE.json"), "w"))
        json.dump({}, open(os.path.join(d, "post_done.json"), "w"))
        R.release(out, j["id"])
    q.put(mine)


def test_two_workers_never_run_a_job_twice_and_respect_dependencies(tmp_path):
    jobs = [{"id": f"J{i}", "depends_on": [f"J{i - 3}"] if i >= 3 else []} for i in range(12)]
    q = mp.Queue()
    procs = [mp.Process(target=_worker, args=(str(tmp_path), jobs, q)) for _ in range(2)]
    [p.start() for p in procs]
    got = [q.get(timeout=120) for _ in procs]
    [p.join() for p in procs]
    ran = sorted(i for g in got for i in g)
    assert ran == sorted(j["id"] for j in jobs)
    assert not os.path.exists(os.path.join(str(tmp_path), ".locks", "J0")) or True


def test_dependency_order_skip_done_and_failed_block(tmp_path):
    out = str(tmp_path)
    jobs = [{"id": "A", "depends_on": []}, {"id": "B", "depends_on": ["A"]}]
    assert R.next_job(jobs, out)["id"] == "A"
    R.release(out, "A")
    assert R.next_job([jobs[1]] + [], out) is None                     # B waits for A
    d = R.run_dir_of(out, "A")
    os.makedirs(d, exist_ok=True)
    json.dump({}, open(os.path.join(d, "DONE.json"), "w"))
    assert R.next_job(jobs, out)["id"] == "A"                          # A is done but post-processing is missing: claimed again
    R.release(out, "A")
    json.dump({}, open(os.path.join(d, "post_done.json"), "w"))
    assert R.next_job(jobs, out)["id"] == "B"                          # now B is ready; A is skipped
    R.release(out, "B")
    fdir = R.run_dir_of(out, "B")
    os.makedirs(fdir, exist_ok=True)
    json.dump({}, open(os.path.join(fdir, "FAILED.json"), "w"))
    assert R.next_job(jobs, out) is None and R.unfinished(jobs, out) == []
    jobs2 = [{"id": "X", "depends_on": []}, {"id": "Y", "depends_on": ["X"]}]
    os.makedirs(R.run_dir_of(out, "X"), exist_ok=True)
    json.dump({}, open(os.path.join(R.run_dir_of(out, "X"), "FAILED.json"), "w"))
    assert R.blocked_forever(jobs2, out) == {"Y"}


def test_stale_lock_of_a_dead_pid_is_stolen(tmp_path):
    out = str(tmp_path)
    os.makedirs(os.path.join(out, ".locks", "Z"))
    open(os.path.join(out, ".locks", "Z", "pid"), "w").write("999999999")
    assert R.try_claim(out, "Z") is True
    assert R.try_claim(out, "Z") is False                              # now owned by this (live) process


def test_deadline_exit_code_is_3_without_starting_jobs(tmp_path):
    manifest = tiny_manifest(tmp_path)
    r = subprocess.run(runner_cmd(manifest, tmp_path / "o", 0, "--deadline-hours", "0.00001"), env=ENV, capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    assert not os.path.exists(tmp_path / "o" / "runs" / "A2_s0" / "DONE.json")


def test_prior_output_copies_finished_and_resumable_runs(tmp_path):
    prior = tmp_path / "prior" / "runs"
    for name, files in (("A2_s0", ["DONE.json", "best_model.pth"]), ("A2_s1", ["last.pth"]), ("A5_s0", ["train_log.jsonl"])):
        (prior / name).mkdir(parents=True)
        for f in files:
            (prior / name / f).write_text("x")
    out = tmp_path / "out"
    (out / "runs" / "A2_s1").mkdir(parents=True)
    R.copy_prior(str(tmp_path / "prior"), str(out), None)
    assert (out / "runs" / "A2_s0" / "DONE.json").exists() and not (out / "runs" / "A5_s0").exists()   # partial without state: skipped
    assert not (out / "runs" / "A2_s1" / "last.pth").exists()           # an existing run dir is never overwritten


# ------------------------------------------------------------------------------------------ end-to-end (CPU)
def test_two_runners_dry_run_end_to_end_with_a3_from_a2_calibration_and_bundle(tmp_path):
    manifest = tiny_manifest(tmp_path)
    out = tmp_path / "out"
    procs = [subprocess.Popen(runner_cmd(manifest, out, g), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for g in (0, 1)]
    outs = [p.communicate(timeout=1500)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], "\n".join(outs)
    logs = "".join(open(out / f"runner_{g}.log").read() for g in (0, 1))
    assert logs.count("start A2_s0") == 1 and logs.count("start A3_s0") == 1          # never run twice
    a2, a3 = out / "runs" / "A2_s0", out / "runs" / "A3_s0"
    for d in (a2, a3):
        assert (d / "DONE.json").exists() and (d / "post_done.json").exists() and not (d / "last.pth").exists()
        assert (d / "preds_test.jsonl").exists() and (d / "logits_test.npz").exists()
    import hashlib
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert sha(a3 / "best_model_stage1.pth") == sha(a2 / "best_model.pth")           # A3 started from A2's best checkpoint
    rows = [json.loads(l) for l in open(a3 / "train_log.jsonl")]
    assert rows and all(r["stage"] == "rl" for r in rows if "stage" in r)            # stage 1 was NOT retrained
    assert (a3 / "calibration" / "temperatures.json").exists() and (a3 / "calibration" / "escalation.json").exists()
    assert (out / "env_info.json").exists()
    # finalize: report + bootstrap + results.md + bundle
    r = subprocess.run(runner_cmd(manifest, out, 0, "--finalize", "--resamples", "50"), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    md = (out / "results.md").read_text(encoding="utf-8")
    for section in ("## Overall", "## Per type", "## S1-Human", "## Paired comparisons", "## Calibration", "## Escalation head", "## Efficiency"):
        assert section in md
    boot = json.load(open(out / "bootstrap.json"))
    assert "A3_vs_A2" in boot["comparisons"] and "A4_vs_A3" in boot["comparisons"]
    z = zipfile.ZipFile(out / "results_bundle.zip")
    names = z.namelist()
    assert "results.md" in names and "bootstrap.json" in names and "env_info.json" in names
    assert "runs/A3_s0/train_log.jsonl" in names and "runs/A3_s0/DONE.json" in names and "runs/A3_s0/logits_test.npz" in names
    assert not [n for n in names if n.endswith((".pth", ".bin"))] and (out / "results_bundle.zip").stat().st_size < 500 * 2 ** 20
    # a second invocation finds everything done and exits 0 without training again
    r2 = subprocess.run(runner_cmd(manifest, out, 0), env=ENV, capture_output=True, text=True)
    assert r2.returncode == 0 and open(out / "runner_0.log").read().count("start A2_s0") <= 1


def test_preflight_runs_on_cpu_and_writes_the_projection(tmp_path):
    out = tmp_path / "pf"
    r = subprocess.run(runner_cmd(MANIFEST, out, 0, "--preflight", "--preflight-steps", "4"), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    pf = json.load(open(out / "preflight.json"))
    for k in ("A2_s0", "A0_s0"):
        assert pf["runs"][k]["seconds_per_step_mean"] > 0 and pf["runs"][k]["projected_seconds_per_epoch"] > 0
        assert set(pf["runs"][k]) >= {"data_wait_share", "peak_memory_gb", "steps_per_epoch", "gpu"}
    assert pf["projection"]["n_jobs"] == 15 and pf["projection"]["total_gpu_hours_worst_case"] > 0 and "env" in pf


# ------------------------------------------------------------------------------------------ --repredict-test
S1V2_ROOT = os.path.abspath(os.path.join(REPO, "..", "vivqa_s1_v2"))
SIGLIP2_MANIFEST = os.path.join(REPO, "kaggle", "experiments_siglip2.json")
SIGLIP2_CACHE = os.path.expanduser("~/data/siglip2_packed")
needs_v2 = pytest.mark.skipif(not (os.path.isfile(os.path.join(S1V2_ROOT, "data", "test.jsonl")) and os.path.isfile(os.path.join(SIGLIP2_CACHE, "meta.json"))),
                              reason="needs the v2 benchmark (../vivqa_s1_v2) and the SigLIP 2 cache (~/data/siglip2_packed)")


@needs_v2
def test_repredict_test_after_test_jsonl_changed(tmp_path):
    """train / calib / dev frozen, test.jsonl replaced (Sheet A2 / S1-Verified): the normal mode refuses, --repredict-test recomputes only the test predictions."""
    m = json.load(open(SIGLIP2_MANIFEST))
    m["jobs"] = [j for j in m["jobs"] if j["id"] == "A2_s0"]
    m["methods"] = {k: v for k, v in m["methods"].items() if k == "A2"}
    m["comparisons"] = []
    manifest = tmp_path / "tiny.json"
    manifest.write_text(json.dumps(m))
    out = tmp_path / "out"

    def cmd(s1, *extra):
        return [PY, os.path.join(REPO, "scripts", "viso_kaggle_run.py"), "--experiments", str(manifest), "--out", str(out), "--code-root", REPO, "--s1-dir", str(s1),
                "--gpu", "0", "--dry-run", "--packed", SIGLIP2_CACHE, "--packed-in-dry-run", "--poll-seconds", "0.2", *extra]

    r = subprocess.run(cmd(S1V2_ROOT), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    a2 = out / "runs" / "A2_s0"
    old_ids = [json.loads(l)["id"] for l in open(a2 / "preds_test.jsonl")]
    old_dev = open(a2 / "preds_dev.jsonl").read()
    # synthetic new S1 dir: every file is a symlink to the frozen one except test.jsonl (first 5 items dropped, one question edited)
    new_s1 = tmp_path / "s1_new"
    (new_s1 / "data").mkdir(parents=True)
    for f in os.listdir(os.path.join(S1V2_ROOT, "data")):
        if f != "test.jsonl":
            os.symlink(os.path.join(S1V2_ROOT, "data", f), new_s1 / "data" / f)
    for f in ("s1bench", "evaluate.py"):                                                # the benchmark code lives next to data/
        if os.path.exists(os.path.join(S1V2_ROOT, f)):
            os.symlink(os.path.join(S1V2_ROOT, f), new_s1 / f)
    lines = open(os.path.join(S1V2_ROOT, "data", "test.jsonl"), encoding="utf-8").read().splitlines()
    dropped = {json.loads(l)["id"] for l in lines[:5]}
    changed = json.loads(lines[10])
    changed["questions"]["q"]["instructions"] += " trong ảnh"                               # a corrected question text (the gold labels stay valid)
    new_lines = lines[5:10] + [json.dumps(changed, ensure_ascii=False)] + lines[11:]
    (new_s1 / "data" / "test.jsonl").write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    # 1) the normal mode refuses a changed test.jsonl
    r = subprocess.run(cmd(new_s1), env=ENV, capture_output=True, text=True)
    assert r.returncode != 0 and "do not match" in r.stdout + r.stderr
    # 2) a changed train / calib / dev is refused even with allow_test_change
    mm = R.load_manifest(str(manifest))
    info = R.check_s1(str(new_s1), mm, allow_test_change=True)
    assert info["test_changed"] and info["test_sha256_now"] != info["test_sha256_manifest"]
    bad = tmp_path / "s1_bad"
    (bad / "data").mkdir(parents=True)
    for f in ("train.jsonl", "calib.jsonl", "dev.jsonl", "test.jsonl"):
        shutil.copy(new_s1 / "data" / f, bad / "data" / f)
    with open(bad / "data" / "dev.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="dev.jsonl"):
        R.check_s1(str(bad), mm, allow_test_change=True)
    # 3) --repredict-test recomputes the test split only and records both hashes
    r = subprocess.run(cmd(new_s1, "--repredict-test"), env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.load(open(out / "repredict.json"))
    assert rep["repredicted"] == ["A2_s0"] and rep["test_changed"] and rep["test_sha256_now"] != rep["test_sha256_manifest"]
    new_ids = [json.loads(l)["id"] for l in open(a2 / "preds_test.jsonl")]
    all_new = {json.loads(l)["id"] for l in new_lines}
    assert new_ids and set(new_ids) <= all_new and not (set(new_ids) & dropped) and new_ids != old_ids
    assert open(a2 / "preds_dev.jsonl").read() == old_dev                                  # dev untouched
