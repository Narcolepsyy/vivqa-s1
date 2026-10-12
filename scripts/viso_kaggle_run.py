"""Resumable experiment runner for Kaggle (one process per GPU, shared job queue). Runs on CPU too (--dry-run).

  python scripts/viso_kaggle_run.py --experiments kaggle/experiments.json --out /kaggle/working/viso_out \\
      --code-root /kaggle/working/code/ViVQAv2-main --s1-dir /kaggle/input/vivqa-s1 --packed /kaggle/input/vivqa-packed \\
      --gpu 0 --deadline-hours 8.5            # a second process with --gpu 1 shares the queue
  ... --preflight                              # 200 steps of A2 and A0 -> preflight.json (seconds/step, memory, projection)
  ... --finalize                               # report + bootstrap + results.md + results_bundle.zip
Queue: a job is claimed by creating <out>/.locks/<id> atomically (os.mkdir; the claimer's pid is stored inside and a
lock of a dead pid is stolen). A job runs only after its `depends_on` have DONE.json; A3 / A3c copy A2's best_model.pth
(same seed) to best_model_stage1.pth and train LOSS.STAGE=rl (A3's stage 1 IS A2's run). Exit codes: 0 all done,
3 stopped by the deadline with resumable state, 1 error / failed jobs. fp32 only (no AMP). No network, no keys."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

S1_FILES = ("train.jsonl", "calib.jsonl", "dev.jsonl", "test.jsonl")
SKIP_BUNDLE_SUFFIX = (".pth", ".pth.tmp", ".bin")
MIN_START_SECONDS = 300          # do not start a new job with less than this left before the deadline


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def code_hash(root):
    """sha256 over the relative path + bytes of every source file under root (git-less code version)."""
    h = hashlib.sha256()
    skip = {"output", "saved_models", "__pycache__", ".pytest_cache", "analysis_results", "plans", "evaluation_results"}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in skip)
        for f in sorted(files):
            if f.endswith((".py", ".yaml", ".yml", ".json", ".md", ".sh", ".txt")) and f != ".env":
                p = os.path.join(d, f)
                h.update(os.path.relpath(p, root).encode()); h.update(sha256_file(p).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- manifest / checks
def load_manifest(path):
    m = json.load(open(path, encoding="utf-8"))
    ids = [j["id"] for j in m["jobs"]]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    for j in m["jobs"]:
        for d in j["depends_on"]:
            assert d in ids, f"{j['id']} depends on unknown job {d}"
    return m


def check_s1(s1_dir, manifest, allow_test_change=False):
    """Abort (ValueError) if the S1 jsonl files differ from the ones the manifest was written for. With allow_test_change (the --repredict-test mode) test.jsonl may differ:
    train / calib / dev are frozen and must match exactly; the function returns {"test_sha256_manifest", "test_sha256_now", "test_changed"}."""
    bad = []
    for f in S1_FILES:
        if allow_test_change and f == "test.jsonl":
            continue
        p = os.path.join(s1_dir, "data", f)
        if not os.path.isfile(p) or sha256_file(p) != manifest["s1_sha256"][f]:
            bad.append(f)
    if bad:
        raise ValueError(f"S1 benchmark files do not match the manifest checksums: {bad} (wrong benchmark version?)")
    now = sha256_file(os.path.join(s1_dir, "data", "test.jsonl"))
    return {"test_sha256_manifest": manifest["s1_sha256"]["test.jsonl"], "test_sha256_now": now, "test_changed": now != manifest["s1_sha256"]["test.jsonl"]}


def env_info(args, manifest):
    info = {"python": platform.python_version(), "platform": platform.platform(), "code_sha256": code_hash(args.code_root),
            "s1_sha256": {f: sha256_file(os.path.join(args.s1_dir, "data", f)) for f in S1_FILES},
            "created_unix": time.time()}
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                     "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    except Exception as e:  # pragma: no cover
        info["torch_error"] = str(e)
    meta = os.path.join(args.packed, "meta.json") if args.packed else None
    if meta and os.path.isfile(meta):
        info["packed_meta_sha256"] = json.load(open(meta))["sha256"]
    return info


# --------------------------------------------------------------------------------------------- configs
def run_dir_of(out, job_id):
    return os.path.join(out, "runs", job_id)


def build_config(job, args, manifest, deadline_unix):
    """Base yaml + job overrides + Kaggle/dry-run overrides, saved as <run dir>/config.yaml; returns its path."""
    from train_viso import load_config
    opts = {}
    opts["TRAINING.SEED"] = job["seed"]
    opts.update(job["overrides"])
    opts.update(manifest["dry_run_overrides"] if args.dry_run else manifest["kaggle_overrides"])
    if args.dry_run and not args.packed_in_dry_run:
        opts["DATASET.FEATURE_DATASET.FEATURE_FORMAT"] = "npy"
    opts.update({"MODEL.NAME": job["id"], "TRAINING.CHECKPOINT_PATH": os.path.join(args.out, "runs"),
                 "TRAINING.OUTPUT_PATH": os.path.join(args.out, "runs"),
                 "DATASET.VOCAB.S1_DIR": os.path.join(args.s1_dir, "data")})
    if args.packed:
        opts["DATASET.FEATURE_DATASET.PACKED_PATH"] = args.packed
    base = os.path.basename(job["base_config"])                       # only the cached-text-encoder configs have a TEXT_CACHE key
    for tag, path in (("tsig", args.text_cache_tsig), ("bami", args.text_cache_bami)):
        if path and f"_{tag}" in base:
            opts["DATASET.FEATURE_DATASET.TEXT_CACHE"] = path
    if args.features:
        opts["DATASET.FEATURE_DATASET.FEATURE_PATH.FEATURES"] = args.features
    if deadline_unix:
        opts["TRAINING.DEADLINE_UNIX"] = float(deadline_unix)
    if args.device:
        opts["MODEL.DEVICE"] = args.device
    flat = []
    for k, v in opts.items():
        flat += [k, v]
    cfg = load_config(os.path.join(args.code_root, job["base_config"]), flat)
    d = run_dir_of(args.out, job["id"])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "run_config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(cfg.dump())
    return path, cfg


# --------------------------------------------------------------------------------------------- queue
def lock_path(out, jid):
    return os.path.join(out, ".locks", jid)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def try_claim(out, jid):
    """Atomic claim (os.mkdir). A lock whose owner pid is dead is stolen. Returns True if we now own the job."""
    os.makedirs(os.path.join(out, ".locks"), exist_ok=True)
    p = lock_path(out, jid)
    try:
        os.mkdir(p)
    except FileExistsError:
        try:
            owner = int(open(os.path.join(p, "pid")).read().strip())
        except (OSError, ValueError):
            return False                         # being created by another process right now
        if owner == os.getpid() or pid_alive(owner):
            return False
        shutil.rmtree(p, ignore_errors=True)     # stale lock of a crashed runner
        try:
            os.mkdir(p)
        except FileExistsError:
            return False
    with open(os.path.join(p, "pid"), "w") as f:
        f.write(str(os.getpid()))
    return True


def release(out, jid):
    shutil.rmtree(lock_path(out, jid), ignore_errors=True)


def is_done(out, jid):
    return os.path.isfile(os.path.join(run_dir_of(out, jid), "DONE.json"))


def is_failed(out, jid):
    return os.path.isfile(os.path.join(run_dir_of(out, jid), "FAILED.json"))


def post_done(out, jid):
    return os.path.isfile(os.path.join(run_dir_of(out, jid), "post_done.json"))


def next_job(jobs, out):
    """First job (manifest order = priority) that is not finished, not locked and whose dependencies are done."""
    for j in jobs:
        jid = j["id"]
        if is_done(out, jid) and post_done(out, jid) or is_failed(out, jid):
            continue
        if not all(is_done(out, d) and post_done(out, d) for d in j["depends_on"]):
            continue
        if try_claim(out, jid):
            return j
    return None


def unfinished(jobs, out):
    return [j for j in jobs if not ((is_done(out, j["id"]) and post_done(out, j["id"])) or is_failed(out, j["id"]))]


def blocked_forever(jobs, out):
    """Unfinished jobs that depend (transitively) on a failed job."""
    failed = {j["id"] for j in jobs if is_failed(out, j["id"])}
    changed = True
    while changed:
        changed = False
        for j in jobs:
            if j["id"] not in failed and any(d in failed for d in j["depends_on"]):
                failed.add(j["id"]); changed = True
    return {j["id"] for j in jobs if j["id"] in failed and not is_failed(out, j["id"])}


# --------------------------------------------------------------------------------------------- one job
def prepare_stage1(job, args):
    """A3 / A3c: copy the donor A2 run's best checkpoint (+ vocab) so the RL stage starts from it."""
    src, dst = run_dir_of(args.out, job["stage1_from"]), run_dir_of(args.out, job["id"])
    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(os.path.join(src, "best_model.pth"), os.path.join(dst, "best_model_stage1.pth"))
    if os.path.isfile(os.path.join(src, "vocab.bin")):
        shutil.copyfile(os.path.join(src, "vocab.bin"), os.path.join(dst, "vocab.bin"))


def post_process(job, args, fh):
    """CPU post-processing of a finished run: preds_<split>.jsonl at T=1, then calibration (+ escalation) if requested."""
    import numpy as np
    from utils import viso_calibration as C
    from utils.viso_s1 import read_jsonl
    d = run_dir_of(args.out, job["id"])
    s1data = os.path.join(args.s1_dir, "data")
    for split in ("dev", "test"):
        lg = C.load_logits(os.path.join(d, f"logits_{split}.npz"))
        by_id = {it["id"]: it for it in read_jsonl(os.path.join(s1data, f"{split}.jsonl"))}
        items = [by_id[int(i)] for i in lg["ids"]]
        C.write_preds_jsonl(os.path.join(d, f"preds_{split}.jsonl"), items, C.probs_from_logits(lg["logits"], lg["space_mask"]))
    if job["post"] == "calibrate":
        import viso_calibrate
        viso_calibrate.main(["--run-dir", d, "--s1-dir", s1data])
    with open(os.path.join(d, "post_done.json"), "w") as f:
        json.dump({"post": job["post"], "unix": time.time()}, f)


def repredict_test(args, manifest, fh, info):
    """--repredict-test: the test split changed after training (Sheet A2 corrections, S1-Verified) while train / calib / dev are frozen. For every finished run with best_model.pth:
    recompute the TEST logits / predictions with the new test.jsonl (viso_predict.py --splits test), then redo the post-processing (preds at T = 1, and for calibrate-jobs the per-type temperature
    and escalation head, fitted again on the FROZEN calib logits, so A0T / A4 are rebuilt consistently). Both test hashes are recorded in <out>/repredict.json."""
    done, skipped = [], []
    for job in manifest["jobs"]:
        d = run_dir_of(args.out, job["id"])
        if not (is_done(args.out, job["id"]) and os.path.isfile(os.path.join(d, "best_model.pth"))):
            skipped.append(job["id"])
            log(f"repredict: {job['id']} skipped (not finished or no best_model.pth in {d})", fh)
            continue
        cfg_path, _ = build_config(job, args, manifest, 0)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VIVQA_S1_ROOT=args.s1_dir)
        if not args.dry_run and args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        with open(os.path.join(d, "stdout_repredict.log"), "a") as so:
            rc = subprocess.run([sys.executable, os.path.join("scripts", "viso_predict.py"), "--config-file", cfg_path, "--splits", "test"], cwd=args.code_root, env=env,
                                stdout=so, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            tail = open(os.path.join(d, "stdout_repredict.log"), errors="replace").read()[-800:]
            log(f"repredict: {job['id']} FAILED (viso_predict exit code {rc}): {tail}", fh)
            skipped.append(job["id"])
            continue
        post_process(job, args, fh)
        done.append(job["id"])
        log(f"repredict: {job['id']} done", fh)
    rep = {**info, "repredicted": done, "skipped": skipped, "unix": time.time(), "note": "train / calib / dev unchanged (checked by sha256); only the test split was re-predicted"}
    json.dump(rep, open(os.path.join(args.out, "repredict.json"), "w"), indent=1)
    return rep


def run_job(job, args, manifest, deadline_unix, fh):
    """Returns 'done' | 'deadline' | 'failed'."""
    d = run_dir_of(args.out, job["id"])
    os.makedirs(d, exist_ok=True)
    if job.get("reuse") and not is_done(args.out, job["id"]):             # a run of an earlier matrix that must have been copied (prior output); never train it here
        json.dump({"returncode": None, "tail": f"reuse job {job['id']}: DONE.json missing: the run was not found in the previous output"}, open(os.path.join(d, "FAILED.json"), "w"))
        log(f"gpu{args.gpu}: {job['id']} is a reuse job but its run directory is missing -> FAILED", fh)
        return "failed"
    if not is_done(args.out, job["id"]):
        if job["stage1_from"] and not os.path.isfile(os.path.join(d, "last.pth")) and not os.path.isfile(os.path.join(d, "best_model_stage1.pth")):
            prepare_stage1(job, args)
        cfg_path, _ = build_config(job, args, manifest, deadline_unix)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VIVQA_S1_ROOT=args.s1_dir)
        if not args.dry_run and args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        log(f"gpu{args.gpu}: start {job['id']}", fh)
        t0 = time.time()
        with open(os.path.join(d, "stdout.log"), "a") as so:
            rc = subprocess.run([sys.executable, "train_viso.py", "--config-file", cfg_path], cwd=args.code_root, env=env,
                                stdout=so, stderr=subprocess.STDOUT).returncode
        log(f"gpu{args.gpu}: {job['id']} exit code {rc} after {time.time() - t0:.0f}s", fh)
        if rc == 3:
            return "deadline"
        if rc != 0 or not is_done(args.out, job["id"]):
            tail = open(os.path.join(d, "stdout.log"), errors="replace").read()[-2000:]
            json.dump({"returncode": rc, "tail": tail}, open(os.path.join(d, "FAILED.json"), "w"))
            return "failed"
    try:
        post_process(job, args, fh)
    except Exception as e:                   # post-processing failures are logged and do not stop other jobs
        log(f"post-processing of {job['id']} failed: {type(e).__name__}: {e}", fh)
        json.dump({"post_error": f"{type(e).__name__}: {e}"}, open(os.path.join(d, "post_done.json"), "w"))
    return "done"


# --------------------------------------------------------------------------------------------- prior output
def copy_prior(prior, out, fh, only=None):
    """Bring finished (DONE.json) and resumable (last.pth) run dirs of an earlier session into --out (only= a set of run ids: copy just those, e.g. the A2 donors of an RL-only re-run)."""
    n = 0
    src_runs = os.path.join(prior, "runs") if os.path.isdir(os.path.join(prior, "runs")) else prior
    for jid in sorted(os.listdir(src_runs)) if os.path.isdir(src_runs) else []:
        s, d = os.path.join(src_runs, jid), os.path.join(out, "runs", jid)
        if only is not None and jid not in only:
            continue
        if os.path.isdir(s) and not os.path.exists(d) and (os.path.isfile(os.path.join(s, "DONE.json")) or os.path.isfile(os.path.join(s, "last.pth"))):
            shutil.copytree(s, d, ignore=shutil.ignore_patterns("FAILED.json", "post_done.json"))
            n += 1
    log(f"prior output: copied {n} run dirs from {prior}", fh)


# --------------------------------------------------------------------------------------------- preflight
def preflight(args, manifest, fh, steps):
    import torch
    import train_viso  # noqa: F401  (registers modules)
    from builders.task_builder import build_task
    res = {"env": env_info(args, manifest), "steps": steps, "runs": {}}
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    jobs = {j["id"]: j for j in manifest["jobs"]}
    for jid in ("A2_s0", "A0_s0"):
        job = json.loads(json.dumps(jobs[jid]))
        job["id"] = f"preflight_{jid}"
        cfg_path, _ = build_config(job, args, manifest, 0)
        from train_viso import load_config
        cfg = load_config(cfg_path, ["TRAINING.RESUME", False])
        task = build_task(cfg)
        task.model.train()
        sched = task._make_scheduler(max(steps, 1))
        loader = iter(task.train_loader)
        waits, times = [], []
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        for i in range(steps):
            t0 = time.perf_counter()
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(task.train_loader)
                batch = next(loader)
            t1 = time.perf_counter()
            batch.to(task.device)
            out = task.model(batch)
            per_item = task.item_loss(out, batch, "soft_ce")
            loss = per_item.mean()
            task.optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(task.model.parameters(), cfg.TRAINING.GRAD_CLIP)
            task.optim.step(); sched.step()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t2 = time.perf_counter()
            waits.append(t1 - t0); times.append(t2 - t0)
        warm = min(10, max(0, steps // 4))
        body = times[warm:] or times
        t_eval = time.perf_counter(); task.evaluate_split("dev"); t_eval = time.perf_counter() - t_eval
        spe = len(task.train_loader)
        mean_step = statistics.mean(body)
        res["runs"][jid] = {
            "seconds_per_step_mean": round(mean_step, 4), "seconds_per_step_median": round(statistics.median(body), 4),
            "data_wait_share": round(sum(waits[warm:]) / max(sum(body), 1e-9), 3), "steps_per_epoch": spe,
            "dev_eval_seconds": round(t_eval, 2), "projected_seconds_per_epoch": round(mean_step * spe + t_eval, 1),
            "peak_memory_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2) if dev.type == "cuda" else None,
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}
        shutil.rmtree(run_dir_of(args.out, job["id"]), ignore_errors=True)
        del task
    # projection: worst case = every supervised run uses all EPOCHS_SUP epochs, RL runs EPOCHS_RL epochs
    sec_epoch = {"FlatS1": res["runs"]["A0_s0"]["projected_seconds_per_epoch"], "ViSO": res["runs"]["A2_s0"]["projected_seconds_per_epoch"]}
    from train_viso import load_config
    total_worst, total_typical = 0.0, 0.0
    for j in manifest["jobs"]:
        cfg = load_config(os.path.join(args.code_root, j["base_config"]), [])
        arch = cfg.MODEL.ARCHITECTURE
        if j["stage1_from"]:
            ep_worst = ep_typ = cfg.TRAINING.EPOCHS_RL
        else:
            ep_worst, ep_typ = cfg.TRAINING.EPOCHS_SUP, 0.6 * cfg.TRAINING.EPOCHS_SUP
        total_worst += ep_worst * sec_epoch[arch]; total_typical += ep_typ * sec_epoch[arch]
    res["projection"] = {"seconds_per_epoch": sec_epoch, "n_jobs": len(manifest["jobs"]),
                         "total_gpu_hours_worst_case": round(total_worst / 3600, 2),
                         "total_gpu_hours_typical_60pct_of_max_epochs": round(total_typical / 3600, 2),
                         "wall_hours_with_two_gpus_worst_case": round(total_worst / 3600 / 2, 2),
                         "note": "RL epochs are timed with the soft-CE step; the K=4 noise samples add a small cost"}
    with open(os.path.join(args.out, "preflight.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("preflight: " + json.dumps(res["projection"]) + " " + json.dumps({k: (v["seconds_per_step_mean"], v["peak_memory_gb"]) for k, v in res["runs"].items()}), fh)
    return res


# --------------------------------------------------------------------------------------------- finalize
def finalize(args, manifest, fh):
    import viso_bootstrap
    import viso_report
    out = args.out
    boot = viso_bootstrap.run(manifest, os.path.join(out, "runs"), os.path.join(args.s1_dir, "data"), split="test",
                              resamples=args.resamples, seed=0)
    with open(os.path.join(out, "bootstrap.json"), "w") as f:
        json.dump(boot, f, indent=1)
    eff = viso_report.static_efficiency(args, manifest)
    md = viso_report.results_markdown(out, manifest, boot, eff, os.path.join(args.s1_dir, "data"))
    try:                                                                       # Phase 2c additions: Image-unseen column, seen-vs-unseen gap, agreement bins, efficiency rows
        import viso_views_report
        veff = None
        vp = os.path.join(args.packed or "", "vision_tower_efficiency.json")
        if args.packed and os.path.isfile(vp):
            veff = json.load(open(vp))
        tpath = os.path.join(args.code_root, "kaggle", "text_tower_efficiency.json")
        teff = json.load(open(tpath)) if os.path.isfile(tpath) else None
        md += viso_views_report.views_markdown(manifest, os.path.join(out, "runs"), os.path.join(args.s1_dir, "data"), "test", veff, eff, teff)
    except Exception as e:                                                     # never lose the main report because of an addition
        md += f"\n(views / agreement-bin additions failed: {type(e).__name__}: {e})\n"
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    bundle = os.path.join(out, "results_bundle.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(out):
            dirs[:] = [x for x in dirs if x not in (".locks", "__pycache__", "preflight_A2_s0", "preflight_A0_s0")]
            for f in sorted(files):
                p = os.path.join(d, f)
                if p == bundle or f.endswith(SKIP_BUNDLE_SUFFIX) or f in ("last.pth",):
                    continue
                z.write(p, os.path.relpath(p, out))
    size = os.path.getsize(bundle)
    log(f"results_bundle.zip: {size / 2 ** 20:.1f} MB", fh)
    if size > 500 * 2 ** 20:
        log("WARNING: bundle is larger than 500 MB", fh)
    return bundle


# --------------------------------------------------------------------------------------------- main
def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default=os.path.join(REPO, "kaggle", "experiments.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-root", default=REPO)
    ap.add_argument("--s1-dir", default=os.path.join(REPO, "..", "vivqa_s1_v2"), help="dir containing data/*.jsonl and s1bench/")
    ap.add_argument("--packed", default=None, help="packed feature dir (FEATURE_FORMAT packed)")
    ap.add_argument("--text-cache-tsig", default=None, help="frozen SigLIP 2 text-tower cache (siglip2-text-cache) for the *_tsig configs: sets DATASET.FEATURE_DATASET.TEXT_CACHE")
    ap.add_argument("--text-cache-bami", default=None, help="frozen BamiBERT cache for the *_bami configs")
    ap.add_argument("--features", default=None, help="npy feature dir (dry-run / non-packed)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", default=None, help="override MODEL.DEVICE (default: from the manifest)")
    ap.add_argument("--deadline-hours", type=float, default=8.5)
    ap.add_argument("--prior-output", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--preflight-steps", type=int, default=200)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--repredict-test", action="store_true", help="test.jsonl changed after training: re-predict the test split of every finished run and redo the post-processing (train / calib / dev must still match the manifest)")
    ap.add_argument("--dry-run", action="store_true", help="CPU, LIMIT 64, 2 epochs (manifest dry_run_overrides)")
    ap.add_argument("--packed-in-dry-run", action="store_true")
    ap.add_argument("--reset-locks", action="store_true", help="remove all locks first (start of a notebook session)")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--resamples", type=int, default=1000)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    args.out, args.code_root = os.path.abspath(args.out), os.path.abspath(args.code_root)
    args.s1_dir = os.path.abspath(args.s1_dir)
    if args.packed:
        args.packed = os.path.abspath(os.path.expanduser(args.packed))
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    sys.path.insert(0, os.path.join(args.code_root, "scripts"))
    os.environ["VIVQA_S1_ROOT"] = args.s1_dir
    fh = open(os.path.join(args.out, f"runner_{args.gpu}.log"), "a")
    manifest = load_manifest(args.experiments)
    s1_info = check_s1(args.s1_dir, manifest, allow_test_change=args.repredict_test)         # refuses a wrong benchmark version (test may differ only with --repredict-test)
    if args.only:
        keep = set(args.only)
        for j in manifest["jobs"]:                                           # keep the dependencies of selected jobs too
            if j["id"] in keep:
                keep.update(j["depends_on"])
        manifest["jobs"] = [j for j in manifest["jobs"] if j["id"] in keep]
    if args.reset_locks:
        shutil.rmtree(os.path.join(args.out, ".locks"), ignore_errors=True)
    info_path = os.path.join(args.out, "env_info.json")
    if not os.path.isfile(info_path):
        json.dump(env_info(args, manifest), open(info_path, "w"), indent=1)
    if args.prior_output:
        copy_prior(args.prior_output, args.out, fh)
    if args.repredict_test:
        rep = repredict_test(args, manifest, fh, s1_info)
        if args.finalize:
            finalize(args, manifest, fh)
        return 1 if not rep["repredicted"] else 0
    if args.preflight:
        preflight(args, manifest, fh, args.preflight_steps)
        return 0
    if args.finalize:
        finalize(args, manifest, fh)
        return 0
    deadline = time.time() + args.deadline_hours * 3600
    jobs = manifest["jobs"]
    while True:
        if time.time() > deadline - MIN_START_SECONDS:
            rc = 3 if unfinished(jobs, args.out) else 0
            log(f"gpu{args.gpu}: deadline reached, exit code {rc}", fh)
            return rc
        job = next_job(jobs, args.out)
        if job is None:
            left = unfinished(jobs, args.out)
            if not left:
                break
            if {j["id"] for j in left} <= blocked_forever(jobs, args.out):
                log(f"gpu{args.gpu}: remaining jobs are blocked by failed dependencies", fh)
                break
            time.sleep(args.poll_seconds)                                    # waiting for another runner / a dependency
            continue
        try:
            status = run_job(job, args, manifest, deadline, fh)
        finally:
            release(args.out, job["id"])
        if status == "deadline":
            log(f"gpu{args.gpu}: {job['id']} stopped by the deadline (resumable)", fh)
            return 3
        log(f"gpu{args.gpu}: {job['id']} -> {status}", fh)
    failed = [j["id"] for j in jobs if is_failed(args.out, j["id"])]
    log(f"gpu{args.gpu}: queue finished; failed jobs: {failed}", fh)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
