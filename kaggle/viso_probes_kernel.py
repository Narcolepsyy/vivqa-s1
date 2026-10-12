"""Kaggle script kernel: ViSO collapse / fusion probes on 2 GPUs (T4 x2). One queue, one worker per GPU.
Inputs (private datasets, found by SEARCHING /kaggle/input/*/ - no hard-coded nesting):
  viso-code (ViVQAv2-main/train_viso.py) | vivqa-s1 (data/*.jsonl + s1bench/) | vivqa-siglip2-packed (meta.json + regions_fp16.bin) | vivqa-bamibert-cache (q_index.npz)
Stage 1 (parallel): D5_long_S3, F1_product, F2_product_visheads. Stage 2: F3 (= F2 + non-uniform pooling init) only if neither F1 nor F2 reaches the bar. Stage 3: B1b = BamiBERT + the winning F config
with its 768 -> 512 projection in the LR_HEADS group. Then the D1 image-shuffle / question-shuffle evaluation of the best run. Every probe: 4000 steps, dev every 500, early stop at step 2000 if the
yes/no train loss is still >= 0.685, per-module gradient norms at step 500. Output: /kaggle/working/probes/<id>.txt, ckpt/, summary.json (rewritten after every job).
BAR (dev, uncalibrated, final eval): yes/no >= 0.56, score >= 0.33, choice >= 0.38.
Local dry run: PROBES_DRY=1 PROBES_INPUT_ROOT=<dir with the 4 dataset folders> PROBES_WORK=<dir> python viso_probes_kernel.py   (CPU, a few steps)."""
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


LR = ["TRAINING.LR_ENCODER", "0.00005", "TRAINING.LR_HEADS", "0.0005", "TRAINING.WARMUP_STEPS", "200"]
BASE = {"viso": "configs/viso_siglip2.yaml", "bami": "configs/viso_siglip2_bami.yaml"}
JOBS = {
    "D5_long_S3": ("viso", LR),
    "F1_product": ("viso", LR + ["MODEL.FUSION_PRODUCT", "True"]),
    "F2_product_visheads": ("viso", LR + ["MODEL.FUSION_PRODUCT", "True", "TRAINING.VISION_PATH_AS_HEADS", "True"]),
    "F3_product_visheads_attrinit": ("viso", LR + ["MODEL.FUSION_PRODUCT", "True", "TRAINING.VISION_PATH_AS_HEADS", "True", "MODEL.ATTR_REDUCE_INIT_STD", "0.3"]),
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


def parse_final(path):
    """-> (last dev result {type: (acc, nll)}, all dev results) from a probe log"""
    evals = []
    for line in open(path, errors="replace"):
        if "dev (acc, nll):" in line:
            try:
                evals.append(ast.literal_eval(line.split("dev (acc, nll):", 1)[1].strip()))
            except (ValueError, SyntaxError):
                pass
    return (evals[-1] if evals else None), evals


def passes(res):
    return {t: bool(res and t in res and res[t][0] >= BAR[t]) for t in BAR}


def score_key(res):
    p = passes(res)
    return (sum(p.values()), res["all"][0] if res and "all" in res else -1)


def run_probe(jid, cfg_key, opts, gpu):
    log = os.path.join(WORK, f"{jid}.txt")
    steps = ["--steps", "12", "--eval-every", "6", "--early-stop-step", "8", "--grad-at", "4"] if DRY else ["--steps", "4000", "--eval-every", "500", "--early-stop-step", "2000", "--grad-at", "500"]
    cmd = [sys.executable, "-B", os.path.join(CODE, "scripts", "viso_probe.py"), "--config-file", os.path.join(CODE, BASE[cfg_key]), *steps, "--work", os.path.join(WORK, "work_" + jid),
           "--ckpt-dir", os.path.join(WORK, "ckpt"), "--ckpt-name", jid, "--opts", *opts, *common_opts(cfg_key)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VIVQA_S1_ROOT=S1)
    if gpu is not None and not DRY:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t = time.time()
    event(f"start {jid} on gpu {gpu}")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    final, evals = parse_final(log)
    with LOCK:
        SUMMARY["probes"][jid] = {"returncode": rc, "seconds": round(time.time() - t), "final_dev": final, "dev_history": evals, "passes_bar": passes(final), "early_stopped": "EARLY STOP" in open(log, errors="replace").read(),
                                  "opts": opts, "config": BASE[cfg_key]}
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
                cfg_key, opts = JOBS[jid]
                run_probe(jid, cfg_key, opts, gpu)
            except Exception as e:                            # a failed job must not stop the others
                event(f"{jid} crashed: {type(e).__name__}: {e}")

    ts = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpu)]
    [t.start() for t in ts]
    [t.join() for t in ts]


def main():
    ensure_deps()
    import torch
    n_gpu = 2 if DRY else max(1, torch.cuda.device_count())
    SUMMARY["env"] = {"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "n_workers": n_gpu}
    event(f"inputs: {SUMMARY['inputs']} | workers: {n_gpu}")
    run_pool(["D5_long_S3", "F1_product", "F2_product_visheads"], n_gpu)
    P = SUMMARY["probes"]
    f_ok = [j for j in ("F1_product", "F2_product_visheads") if j in P and any(passes(P[j]["final_dev"]).values()) and all(passes(P[j]["final_dev"]).values())]
    if not f_ok:                                              # neither F1 nor F2 reaches the bar -> F3
        run_pool(["F3_product_visheads_attrinit"], n_gpu)
    cands = [j for j in ("F1_product", "F2_product_visheads", "F3_product_visheads_attrinit") if j in P and P[j]["final_dev"]]
    if cands:
        win = max(cands, key=lambda j: score_key(P[j]["final_dev"]))
        SUMMARY["winner_F"] = win
        JOBS["B1b_bami_" + win] = ("bami", JOBS[win][1] + ["TRAINING.TEXT_PROJ_AS_HEADS", "True"])
        run_pool(["B1b_bami_" + win], n_gpu)
    done = {j: v for j, v in P.items() if v["final_dev"]}
    if done:
        best = max(done, key=lambda j: score_key(done[j]["final_dev"]))
        SUMMARY["best_run"] = best
        ck = os.path.join(WORK, "ckpt", f"ckpt_{best}.pth")
        if os.path.isfile(ck):
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VIVQA_S1_ROOT=S1)
            cmd = [sys.executable, "-B", os.path.join(CODE, "scripts", "viso_vision_check.py"), "--ckpt", ck, "--device", "cpu" if DRY else "cuda"] + (["--limit", "96"] if DRY else [])
            with open(os.path.join(WORK, f"D1_shuffle_{best}.txt"), "w") as f:
                subprocess.run(cmd, cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT)
            SUMMARY["d1_shuffle_file"] = f"D1_shuffle_{best}.txt"
    event("all done")


if __name__ == "__main__":
    main()
