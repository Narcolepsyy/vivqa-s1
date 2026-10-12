"""Kaggle script kernel: the MAIN ViSO MATRIX (lean manifest kaggle/experiments_lean.json, 12 jobs) on frozen SigLIP 2 features, ViVQA-S1 v2.0.2.
Final config (planner decision 2026-10-09, KD dropped): main text encoder = T-SIG (the frozen SigLIP 2 text tower) + F2 (FUSION_PRODUCT, VISION_PATH_AS_HEADS, TEXT_PROJ_AS_HEADS), strict text mask,
LR encoder 5e-5 / heads 5e-4, warmup 200, best-dev-NLL checkpoint, soft-CE human labels only (no KD, no type weights); EPOCHS_SUP 12 (= 6000 steps at batch 64, the probes' budget), PATIENCE 5 epochs (a run
therefore never stops before ~3000 steps).
Jobs (priority order): A2_s0, A0_s0, A3_s0 (RL stage from A2_s0's checkpoint), A5_s0, ABL_bami_s0, ABL_scratch_s0, then seeds 1-2 of A2 / A0 / A3. Post-processing as the runner does it: preds at T=1, A0T / A4 = per-type temperature
(+ escalation head for A4) fitted on calib. Predictions (logits) of calib / dev / test are written for every run; the test split is still unfrozen: re-predict later with MATRIX_MODE=repredict (viso_kaggle_run.py --repredict-test).
One runner process per GPU (2 on T4 x2) shares the job queue (atomic lock directories). Live console: every runner line is echoed with a [gpuN] prefix, a heartbeat line every 60 s (jobs done, per GPU: job, last epoch, dev NLL / acc,
age of the last epoch row, GPU util / memory). Deadline MATRIX_BUDGET_HOURS (default 11 h): no new epoch starts after it, state is saved (last.pth) and the run ends 'incomplete'; attach this kernel's output
(viso_out) to the next run and it resumes (finished runs are copied, half-finished ones continue from last.pth).
At the end (also when incomplete): finalize (report, bootstrap, results.md, results_bundle.zip) and the A3c RULE INPUT: paired A3 - A2 differences with 95% CIs per seed (views all / image_unseen / core).
Inputs are found by SEARCHING /kaggle/input (depth 6): viso-code | vivqa-s1 | vivqa-siglip2-packed | vivqa-siglip2-text-cache | vivqa-bamibert-cache | (a previous output: a directory named viso_out with runs/).
Output: /kaggle/working/viso_out/{runs/<job>/..., results.md, results_bundle.zip, bootstrap.json, a3c_rule_input.{json,md}, runner_*.log}, /kaggle/working/summary.json (atomic, rewritten often).
Local dry run: MATRIX_DRY=1 MATRIX_INPUT_ROOT=<dir with the dataset folders> MATRIX_WORK=<disk dir, NOT /tmp> python viso_matrix_kernel.py   (CPU, 64 items, 2 epochs; MATRIX_ONLY="A2_s0 A3_s0" selects jobs)."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time

DRY = os.environ.get("MATRIX_DRY") == "1"
INPUT = os.environ.get("MATRIX_INPUT_ROOT", "/kaggle/input")
WORK = os.path.abspath(os.environ.get("MATRIX_WORK", "/kaggle/working"))
BUDGET_H = float(os.environ.get("MATRIX_BUDGET_HOURS", "11"))
ONLY = os.environ.get("MATRIX_ONLY", "").split()
MODE = os.environ.get("MATRIX_MODE", "run")                        # run | repredict
OUT = os.path.join(WORK, "viso_out")
MANIFEST = os.environ.get("MATRIX_MANIFEST", "kaggle/experiments_lean.json")            # the RL-fix kernel sets kaggle/experiments_rlfix.json
PRIOR_RENAME = dict(x.split("=", 1) for x in os.environ.get("MATRIX_PRIOR_RENAME", "").split())             # old=new: copy a previous run under another name (e.g. ABL_bami_s0=A2bami_s0)
PRIOR_ONLY = os.environ.get("MATRIX_PRIOR_ONLY", "").split()                                # copy only these run ids from the previous output (e.g. the A2 donors)
RULE_KEY = os.environ.get("MATRIX_RULE_KEY", "A3_vs_A2")                                   # the paired comparison reported as the A3c rule input
T0 = time.time()
os.makedirs(WORK, exist_ok=True)
LOCK = threading.Lock()
SUMMARY = {"dry_run": DRY, "mode": MODE, "status": "running", "events": [], "runners": {}}


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_summary():
    with LOCK:
        SUMMARY["elapsed_s"] = round(time.time() - T0)
        tmp = os.path.join(WORK, "summary.json.tmp")
        json.dump(SUMMARY, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, os.path.join(WORK, "summary.json"))


def event(msg):
    say(msg)
    with LOCK:
        SUMMARY["events"].append(f"{time.strftime('%H:%M:%S')} {msg}")
    save_summary()


def find_dir(test, depth=6, root=None):
    stack = [(root or INPUT, 0)]
    while stack:
        d, lvl = stack.pop(0)
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        if test(d):
            return d
        if lvl < depth:
            stack += [(os.path.join(d, n), lvl + 1) for n in names if os.path.isdir(os.path.join(d, n))]
    return None


def find_all(test, depth=6):
    """every directory below INPUT (at most `depth` levels) for which test(path) is true (a match is not descended into)"""
    out, stack = [], [(INPUT, 0)]
    while stack:
        d, lvl = stack.pop(0)
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        if test(d):
            out.append(d)
            continue
        if lvl < depth:
            stack += [(os.path.join(d, n), lvl + 1) for n in names if os.path.isdir(os.path.join(d, n))]
    return out


def meta_model(d):
    try:
        return json.load(open(os.path.join(d, "meta.json"))).get("model", "")
    except (OSError, ValueError):
        return ""


def locate():
    code = find_dir(lambda d: os.path.isfile(os.path.join(d, "train_viso.py")) and os.path.isfile(os.path.join(d, MANIFEST)) and os.path.isfile(os.path.join(d, "scripts", "viso_kaggle_run.py")))
    s1 = find_dir(lambda d: os.path.isfile(os.path.join(d, "data", "train.jsonl")) and os.path.isdir(os.path.join(d, "s1bench")))
    packed = find_dir(lambda d: os.path.isfile(os.path.join(d, "regions_fp16.bin")) and os.path.isfile(os.path.join(d, "meta.json")))
    tsig = find_dir(lambda d: os.path.isfile(os.path.join(d, "q_index.npz")) and "siglip2" in meta_model(d).lower()) if os.environ.get("MATRIX_NEED_TSIG", "1") == "1" else None
    bami = find_dir(lambda d: os.path.isfile(os.path.join(d, "q_index.npz")) and "bami" in meta_model(d).lower()) if os.environ.get("MATRIX_NEED_BAMI", "1") == "1" else None
    miss = [n for n, v in (("viso-code (with kaggle/experiments_lean.json)", code), ("vivqa-s1", s1), ("vivqa-siglip2-packed", packed), ("vivqa-siglip2-text-cache", tsig), ("vivqa-bamibert-cache", bami)) if v is None and not (n == "vivqa-siglip2-text-cache" and os.environ.get("MATRIX_NEED_TSIG", "1") != "1") and not (n == "vivqa-bamibert-cache" and os.environ.get("MATRIX_NEED_BAMI", "1") != "1")]
    if miss:
        raise SystemExit(f"input datasets not found under {INPUT}: {miss}")
    priors = find_all(lambda d: os.path.basename(d) == "viso_out" and os.path.isdir(os.path.join(d, "runs")) and os.path.abspath(d) != OUT)       # every attached previous output
    return code, s1, packed, tsig, bami, priors


def ensure_deps():
    import importlib
    for mod, pkg in (("yacs", "yacs"), ("termcolor", "termcolor"), ("tabulate", "tabulate"), ("joblib", "joblib"), ("sacremoses", "sacremoses"), ("pytorch_transformers", "pytorch_transformers")):
        try:
            importlib.import_module(mod)
        except ImportError:
            r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", pkg], capture_output=True, text=True)
            say(f"pip install {pkg}: {'ok' if r.returncode == 0 else r.stderr[-300:]}")


def gpu_count():
    if os.environ.get("MATRIX_RUNNERS"):                                # testing / manual override
        return int(os.environ["MATRIX_RUNNERS"])
    if DRY:
        return 1
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return max(1, len([ln for ln in r.stdout.splitlines() if ln.startswith("GPU")]))
    except Exception:
        return 1


def gpu_text(i):
    try:
        r = subprocess.run(["nvidia-smi", f"--id={i}", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        u, m, t = [x.strip() for x in r.stdout.strip().split(",")]
        return f"util {u}% mem {float(m) / 1024:.1f}/{float(t) / 1024:.1f} GB"
    except Exception:
        return "gpu ?"


def read_last_row(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 4096))
            lines = f.read().decode("utf-8", "replace").strip().splitlines()
        for ln in reversed(lines):
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            if "epoch" in row and "dev_nll" in row:
                return row, os.path.getmtime(path)
    except OSError:
        pass
    return None, None


def job_ids():
    m = json.load(open(os.path.join(CODE, MANIFEST)))
    ids = [j["id"] for j in m["jobs"]]
    if ONLY:
        keep = set(ONLY)
        for j in m["jobs"]:
            if j["id"] in keep:
                keep.update(j["depends_on"])
        ids = [i for i in ids if i in keep]
    return ids


def progress():
    ids = job_ids()
    done = [i for i in ids if os.path.isfile(os.path.join(OUT, "runs", i, "DONE.json")) and os.path.isfile(os.path.join(OUT, "runs", i, "post_done.json"))]
    failed = [i for i in ids if os.path.isfile(os.path.join(OUT, "runs", i, "FAILED.json"))]
    return ids, done, failed


def heartbeat(procs, stop):
    while not stop.wait(60.0):
        ids, done, failed = progress()
        parts = []
        locks = os.path.join(OUT, ".locks")
        pid2gpu = {p.pid: g for g, p in procs.items()}
        running = {}
        for jid in sorted(os.listdir(locks)) if os.path.isdir(locks) else []:
            try:
                pid = int(open(os.path.join(locks, jid, "pid")).read().strip())
            except (OSError, ValueError):
                continue
            if pid in pid2gpu:
                running[pid2gpu[pid]] = jid
        for g in sorted(procs):
            jid = running.get(g)
            if jid is None:
                parts.append(f"gpu{g}: idle/finishing ({gpu_text(g) if not DRY else 'cpu'})")
                continue
            row, mt = read_last_row(os.path.join(OUT, "runs", jid, "train_log.jsonl"))
            ep = f"epoch {row['epoch']} {row.get('stage', '')} dev NLL {row['dev_nll']:.4f} acc {row['dev_acc']:.3f} ({(time.time() - mt) / 60:.0f} min ago)" if row else "no epoch finished yet"
            parts.append(f"gpu{g}: {jid} {ep} ({gpu_text(g) if not DRY else 'cpu'})")
        say(f"HB {(time.time() - T0) / 3600:.2f} h | done {len(done)}/{len(ids)}{' | FAILED ' + ','.join(failed) if failed else ''} | " + " | ".join(parts))
        SUMMARY["jobs"] = {"expected": ids, "done": done, "failed": failed}
        save_summary()


def pump(proc, tag, logf):
    for line in proc.stdout:
        sys.stdout.write(f"[{tag}] {line}")
        sys.stdout.flush()
        logf.write(line)
        logf.flush()


def runner_cmd(extra):
    c = [sys.executable, "-u", "-B", os.path.join(CODE, "scripts", "viso_kaggle_run.py"), "--experiments", os.path.join(CODE, MANIFEST), "--out", OUT, "--code-root", CODE, "--s1-dir", S1,
         "--packed", PACKED] + (["--text-cache-bami", BAMI] if BAMI else []) + (["--text-cache-tsig", TSIG] if TSIG else [])
    if DRY:
        c += ["--dry-run", "--packed-in-dry-run"]
    if ONLY:
        c += ["--only", *ONLY]
    return c + extra


def a3c_rule_input():
    """paired A3 - A2 (per seed) with 95% CIs, for the views all / image_unseen / core: the input of the A3c decision rule (A3c only if A3 beat A2 with a CI excluding 0)"""
    sys.path.insert(0, os.path.join(CODE, "scripts"))
    sys.path.insert(0, CODE)
    os.environ["VIVQA_S1_ROOT"] = S1
    import viso_bootstrap as B
    manifest = json.load(open(os.path.join(CODE, MANIFEST)))
    lower_better = {"nll", "brier", "ece", "aurc"}
    out, md = {}, [f"# A3c rule input: {RULE_KEY.replace('_vs_', ' - ')} (paired bootstrap 95% CI, test split)", "", "A3c is added only if the RL variant beats A2 with a CI that excludes 0 (accuracy higher is better; NLL / Brier / ECE / AURC lower is better).", ""]
    for view in ("all", "image_unseen", "core"):
        try:
            res = B.run(manifest, os.path.join(OUT, "runs"), os.path.join(S1, "data"), split="test", resamples=1000, seed=0, view=view)
        except Exception as e:
            md.append(f"view {view}: failed ({type(e).__name__}: {e})\n")
            continue
        comp = res.get("comparisons", {}).get(RULE_KEY)
        if not comp:
            md.append(f"view {view}: no finished pair for {RULE_KEY} yet (n_items {res.get('n_items')})\n")
            continue
        out[view] = {"n_items": res["n_items"], "per_seed": comp["per_seed"], "diff_mean_std_over_seeds": comp["diff_mean_std_over_seeds"]}
        md += [f"## view {view} (n = {res['n_items']})", "", "| seed | metric | difference | 95% CI | verdict |", "|---|---|---|---|---|"]
        for seed, per in sorted(comp["per_seed"].items()):
            for k, v in per["primary"].items():
                lo, hi = v["ci95"]
                excl = lo > 0 or hi < 0
                better = (v["diff"] < 0) if k in lower_better else (v["diff"] > 0)
                verdict = ("first method better, CI excludes 0" if better else "first method WORSE, CI excludes 0") if excl else "n.s. (CI includes 0)"
                md.append(f"| {seed} | {k} | {v['diff']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {verdict} |")
        md.append("")
    json.dump(out, open(os.path.join(OUT, "a3c_rule_input.json"), "w"), indent=1)
    open(os.path.join(OUT, "a3c_rule_input.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    return out


def main():
    global CODE, S1, PACKED, TSIG, BAMI
    code_in, S1, PACKED, TSIG, BAMI, priors = locate()
    prior = priors[0] if priors else None
    event(f"inputs: code {code_in}, s1 {S1}, packed {PACKED}, tsig {TSIG}, bami {BAMI}, previous outputs {priors}")
    CODE = os.path.join(WORK, "code", "ViVQAv2-main")
    shutil.rmtree(os.path.join(WORK, "code"), ignore_errors=True)
    shutil.copytree(code_in, CODE)
    SUMMARY["inputs"] = {"code": code_in, "s1": S1, "packed": PACKED, "tsig": TSIG, "bami": BAMI, "prior_outputs": priors}
    os.environ["VIVQA_S1_ROOT"] = S1
    ensure_deps()
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    shutil.rmtree(os.path.join(OUT, ".locks"), ignore_errors=True)               # a new session: stale locks of a dead session must not block jobs
    n_gpu = gpu_count()
    r = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"], capture_output=True, text=True)
    event(f"torch/GPU: {r.stdout.strip() or r.stderr.strip()[-200:]} -> {n_gpu} runner process(es)")
    SUMMARY["gpus"] = n_gpu
    sys.path.insert(0, os.path.join(CODE, "scripts"))
    sys.path.insert(0, CODE)
    for prior in priors:                                                          # every attached previous output (the first one that has a run wins when two have it)
        import viso_kaggle_run as R
        with open(os.path.join(OUT, "runner_matrix.log"), "a") as fh:
            R.copy_prior(prior, OUT, fh, only=set(PRIOR_ONLY) or None)           # once, here: two runners copying at the same time would race
        src_runs = os.path.join(prior, "runs") if os.path.isdir(os.path.join(prior, "runs")) else prior
        for old, new in PRIOR_RENAME.items():                                     # the same finished run under its family name; the alias is recorded next to it
            s_dir, d_dir = os.path.join(src_runs, old), os.path.join(OUT, "runs", new)
            if os.path.isdir(s_dir) and not os.path.exists(d_dir):
                shutil.copytree(s_dir, d_dir, ignore=shutil.ignore_patterns("FAILED.json", "post_done.json"))
                json.dump({"alias_of": old, "note": f"{new} IS the run {old} of the previous matrix (identical configuration, seed 0); copied under the family name"}, open(os.path.join(d_dir, "ALIAS.json"), "w"))
                event(f"alias: {old} copied as {new}")
        event(f"previous output copied from {prior}: {sorted(os.listdir(os.path.join(OUT, 'runs')))}")
    remaining_h = max(0.05, BUDGET_H - (time.time() - T0) / 3600)

    if MODE == "repredict":
        rc = subprocess.run(runner_cmd(["--repredict-test", "--gpu", "0"]), cwd=CODE).returncode
        event(f"repredict-test finished with exit code {rc}")
        SUMMARY["repredict_rc"] = rc
    else:
        procs, stop = {}, threading.Event()
        for g in range(n_gpu):
            p = subprocess.Popen(runner_cmd(["--gpu", str(g), "--deadline-hours", f"{remaining_h:.3f}"]), cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
            procs[g] = p
            threading.Thread(target=pump, args=(p, f"gpu{g}", open(os.path.join(OUT, f"stdout_gpu{g}.log"), "a")), daemon=True).start()
            event(f"runner gpu{g} started (pid {p.pid}), deadline in {remaining_h:.2f} h")
            time.sleep(3)
        threading.Thread(target=heartbeat, args=(procs, stop), daemon=True).start()
        rcs = {g: p.wait() for g, p in procs.items()}
        stop.set()
        SUMMARY["runners"] = {f"gpu{g}": rc for g, rc in rcs.items()}
        event(f"runners finished: {SUMMARY['runners']} (0 all done, 3 deadline/resumable, 1 failed jobs)")

    ids, done, failed = progress()
    SUMMARY["jobs"] = {"expected": ids, "done": done, "failed": failed}
    save_summary()
    try:
        rc = subprocess.run(runner_cmd(["--finalize"]), cwd=CODE).returncode
        event(f"finalize exit code {rc}")
    except Exception as e:
        event(f"finalize failed: {type(e).__name__}: {e}")
    try:
        SUMMARY["a3c_rule_input"] = a3c_rule_input()
    except Exception as e:
        event(f"A3c rule input failed: {type(e).__name__}: {e}")
    complete = len(done) == len(ids) and not failed
    SUMMARY["status"] = "complete" if complete else (f"failed jobs: {failed}" if failed else "incomplete: attach this kernel's output to the next run to continue")
    event(f"status: {SUMMARY['status']} ({len(done)}/{len(ids)} jobs done)")


if __name__ == "__main__":
    main()
