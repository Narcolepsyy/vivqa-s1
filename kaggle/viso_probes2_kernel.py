"""Kaggle script kernel, round 2 (F2 = FUSION_PRODUCT + VISION_PATH_AS_HEADS is the base config; strict text mask; low LR enc 5e-5 / heads 5e-4, warmup 200). 2 GPUs, one queue.
Inputs found by SEARCHING /kaggle/input (depth 6): viso-code | vivqa-s1 | vivqa-siglip2-packed | vivqa-bamibert-cache.
Jobs: Y1_F2_8000 (8000 steps, NO early stop, dev every 500: does yes/no bootstrap late?), Y2_F2_8000_w3 (Y1 + yes/no loss weight 3), B1c_bami_std_F2 (BamiBERT with per-dimension standardised cached
vectors + F2 + the 768 -> 512 projection in LR_HEADS, 4000 steps, no early stop). Then, on Y1 and Y2 (both GPUs in parallel): Y4 = viso_vision_check.py (normal / image- / question-shuffled dev per
type, paired-bootstrap CIs, yes/no dev accuracy split by question form with the image-shuffle drop CI per form) and Y3 = viso_h_probe.py (logistic regression on the frozen fused representation h:
train yes/no -> dev yes/no). Output: /kaggle/working/probes/<job>.txt, Y4_<job>.txt, Y3_<job>.txt, ckpt/, summary.json (rewritten after every job; includes the per-500-step yes/no curve of every probe).
Local dry run: PROBES_DRY=1 PROBES_INPUT_ROOT=<dir with the 4 dataset folders> PROBES_WORK=<dir> python viso_probes2_kernel.py   (CPU, a few steps)."""
import ast
import json
import os
import queue
import subprocess
import sys
import threading
import time

DRY = os.environ.get("PROBES_DRY") == "1"
INPUT = os.environ.get("PROBES_INPUT_ROOT", "/kaggle/input")
WORK = os.path.abspath(os.environ.get("PROBES_WORK", "/kaggle/working/probes"))
BUDGET_S = float(os.environ.get("PROBES_BUDGET_HOURS", "10.5")) * 3600        # no new job after this (Kaggle sessions end at 12 h)
BAR = {"noul": 0.56, "score": 0.33, "choice": 0.38}
T0 = time.time()
os.makedirs(WORK, exist_ok=True)


def find_dir(test, depth=6):
    """first directory below INPUT (at most `depth` levels) for which test(path) is true"""
    stack = [(INPUT, 0)]
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


def locate():
    code = find_dir(lambda d: os.path.isfile(os.path.join(d, "train_viso.py")) and os.path.isfile(os.path.join(d, "scripts", "viso_probe.py")))
    s1 = find_dir(lambda d: os.path.isfile(os.path.join(d, "data", "train.jsonl")) and os.path.isdir(os.path.join(d, "s1bench")))
    packed = find_dir(lambda d: os.path.isfile(os.path.join(d, "regions_fp16.bin")) and os.path.isfile(os.path.join(d, "meta.json")))
    bami = find_dir(lambda d: os.path.isfile(os.path.join(d, "q_index.npz")) and os.path.isfile(os.path.join(d, "c_vecs.npy")))
    miss = [n for n, v in (("viso-code", code), ("vivqa-s1", s1), ("vivqa-siglip2-packed", packed), ("vivqa-bamibert-cache", bami)) if v is None]
    if miss:
        raise SystemExit(f"input datasets not found under {INPUT}: {miss}")
    return code, s1, packed, bami


def ensure_deps():
    import importlib
    for mod, pkg in (("yacs", "yacs"), ("termcolor", "termcolor"), ("tabulate", "tabulate"), ("joblib", "joblib"), ("sacremoses", "sacremoses"), ("pytorch_transformers", "pytorch_transformers")):
        try:
            importlib.import_module(mod)
        except ImportError:
            r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", pkg], capture_output=True, text=True)
            print(f"pip install {pkg}: {'ok' if r.returncode == 0 else r.stderr[-300:]}", flush=True)


CODE, S1, PACKED, BAMI = locate()
SUMMARY = {"inputs": {"code": CODE, "s1": S1, "packed": PACKED, "bami": BAMI}, "dry_run": DRY, "bar": BAR, "probes": {}, "events": []}
LOCK = threading.Lock()


def save_summary():
    with LOCK:
        SUMMARY["elapsed_s"] = round(time.time() - T0)
        tmp = os.path.join(WORK, "summary.json.tmp")
        json.dump(SUMMARY, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, os.path.join(WORK, "summary.json"))              # atomic: a reader never sees a half-written file


def event(msg):
    with LOCK:
        SUMMARY["events"].append(f"{time.strftime('%H:%M:%S')} {msg}")
    print(msg, flush=True)
    save_summary()


F2 = ["TRAINING.LR_ENCODER", "0.00005", "TRAINING.LR_HEADS", "0.0005", "TRAINING.WARMUP_STEPS", "200", "MODEL.FUSION_PRODUCT", "True", "TRAINING.VISION_PATH_AS_HEADS", "True"]
BASE = {"viso": "configs/viso_siglip2.yaml", "bami": "configs/viso_siglip2_bami.yaml"}
# job id -> (config key, opts, steps, early-stop step (0 = none))
JOBS = {
    "Y1_F2_8000": ("viso", F2, 8000, 0),
    "Y2_F2_8000_w3": ("viso", F2 + ["LOSS.TYPE_WEIGHTS", "[3.0, 1.0, 1.0]"], 8000, 0),
    "B1c_bami_std_F2": ("bami", F2 + ["TRAINING.TEXT_PROJ_AS_HEADS", "True"], 4000, 0),
}


def common_opts(cfg_key):
    o = ["DATASET.FEATURE_DATASET.PACKED_PATH", PACKED, "DATASET.VOCAB.S1_DIR", os.path.join(S1, "data"), "DATASET.FEATURE_DATASET.WORKERS", "0" if DRY else "2"]
    if cfg_key == "bami":                                   # only the BamiBERT configs have a TEXT_CACHE key
        o += ["DATASET.FEATURE_DATASET.TEXT_CACHE", BAMI]
    if DRY:
        o += ["MODEL.DEVICE", "cpu", "DATASET.FEATURE_DATASET.LIMIT", "96", "DATASET.FEATURE_DATASET.BATCH_SIZE", "16"]
    else:
        o += ["MODEL.DEVICE", "cuda"]
    return o


def parse_log(path):
    """-> (last dev result, all dev results, yes/no curve [{step, train_noul, dev_noul_acc, dev_noul_nll}] at every dev evaluation)"""
    evals, curve, step, train_noul = [], [], 0, None
    for line in open(path, errors="replace"):
        if line.startswith("step ") and "per-type train loss" in line:
            try:
                step = int(line.split()[1])
                train_noul = ast.literal_eval(line.split("per-type train loss", 1)[1].strip().split("}")[0] + "}").get("noul")
            except (ValueError, SyntaxError):
                pass
        elif "dev (acc, nll):" in line:
            try:
                ev = ast.literal_eval(line.split("dev (acc, nll):", 1)[1].strip())
            except (ValueError, SyntaxError):
                continue
            evals.append(ev)
            curve.append({"step": step, "train_noul_loss": train_noul, "dev_noul_acc": ev.get("noul", [None, None])[0], "dev_noul_nll": ev.get("noul", [None, None])[1],
                          "dev_score_acc": ev.get("score", [None])[0], "dev_choice_acc": ev.get("choice", [None])[0]})
    return (evals[-1] if evals else None), evals, curve


def passes(res):
    return {t: bool(res and t in res and res[t][0] >= BAR[t]) for t in BAR}


def run_probe(jid, gpu):
    cfg_key, opts, steps, es = JOBS[jid]
    log = os.path.join(WORK, f"{jid}.txt")
    st = ["--steps", "12", "--eval-every", "6", "--grad-at", "4"] if DRY else ["--steps", str(steps), "--eval-every", "500", "--grad-at", "500"]
    if es and not DRY:
        st += ["--early-stop-step", str(es)]
    cmd = [sys.executable, "-B", os.path.join(CODE, "scripts", "viso_probe.py"), "--config-file", os.path.join(CODE, BASE[cfg_key]), *st, "--work", os.path.join(WORK, "work_" + jid),
           "--ckpt-dir", os.path.join(WORK, "ckpt"), "--ckpt-name", jid, "--opts", *opts, *common_opts(cfg_key)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VIVQA_S1_ROOT=S1)
    if gpu is not None and not DRY:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t = time.time()
    event(f"start {jid} on gpu {gpu}")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    final, evals, curve = parse_log(log)
    with LOCK:
        SUMMARY["probes"][jid] = {"returncode": rc, "seconds": round(time.time() - t), "final_dev": final, "yes_no_curve": curve, "passes_bar": passes(final), "opts": opts, "steps": steps, "config": BASE[cfg_key]}
    event(f"done {jid} rc={rc} final dev {final}")
    return final


def run_pool(job_ids, n_gpu):
    q = queue.Queue()
    for j in job_ids:
        q.put(j)

    def worker(gpu):
        while True:
            try:
                jid = q.get_nowait()
            except queue.Empty:
                return
            if time.time() - T0 > BUDGET_S:
                event(f"budget exhausted, skipping {jid}")
                continue
            try:
                run_probe(jid, gpu)
            except Exception as e:                            # a failed job must not stop the others
                event(f"{jid} crashed: {type(e).__name__}: {e}")

    ts = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpu)]
    [t.start() for t in ts]
    [t.join() for t in ts]


def analyse(jid, gpu):
    """Y4 (shuffle drops with CIs + yes/no by question form) and Y3 (logistic-regression probe on h) of one finished probe"""
    ck = os.path.join(WORK, "ckpt", f"ckpt_{jid}.pth")
    if not os.path.isfile(ck):
        return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VIVQA_S1_ROOT=S1)
    if gpu is not None and not DRY:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    dev = "cpu" if DRY else "cuda"
    lim = ["--limit", "96"] if DRY else []
    lim_tr = ["--limit-train", "300"] if DRY else []
    for name, script, extra in (("Y4", "viso_vision_check.py", lim), ("Y3", "viso_h_probe.py", lim_tr + ["--out", os.path.join(WORK, f"Y3_{jid}.json")])):
        with open(os.path.join(WORK, f"{name}_{jid}.txt"), "w") as f:
            subprocess.run([sys.executable, "-B", os.path.join(CODE, "scripts", script), "--ckpt", ck, "--device", dev, *extra], cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT)
        event(f"{name} done for {jid}")


def main():
    ensure_deps()
    import torch
    n_gpu = 2 if DRY else max(1, torch.cuda.device_count())
    SUMMARY["env"] = {"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "n_workers": n_gpu}
    event(f"inputs: {SUMMARY['inputs']} | workers: {n_gpu}")
    run_pool(["Y1_F2_8000", "Y2_F2_8000_w3", "B1c_bami_std_F2"], n_gpu)
    P = SUMMARY["probes"]
    done = [j for j in ("Y1_F2_8000", "Y2_F2_8000_w3") if j in P and P[j]["final_dev"]]
    ts = [threading.Thread(target=analyse, args=(j, g % n_gpu)) for g, j in enumerate(done)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    event("all done")


if __name__ == "__main__":
    main()
