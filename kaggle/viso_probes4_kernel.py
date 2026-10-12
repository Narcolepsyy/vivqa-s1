"""Kaggle script kernel, round 4 = KNOWLEDGE DISTILLATION probe (planner decisions 2026-10-09). Main encoder = T-SIG (the frozen SigLIP 2 text tower; standardised cached features), base config = F2 + T-SIG
(FUSION_PRODUCT, VISION_PATH_AS_HEADS, TEXT_PROJ_AS_HEADS, low LR enc 5e-5 / heads 5e-4, warmup 200, strict text mask), 6000 steps, dev every 500, NO early stop, best-dev-NLL checkpoints.
Jobs: KD5 (loss = 0.5 CE(human soft label) + 0.5 CE(teacher distribution) on train), KD8 (0.2 / 0.8), TSIG_s1 (the same T-SIG config WITHOUT distillation, seed 1: the noise reference for the round-3 TSIG seed 0).
Teacher targets: TRAIN only (the 'vivqa-teacher-kd' dataset: id + probability list per item; items the teacher refused keep the human loss only); never used for selection or calibration (dev NLL against our gold).
Analysis: viso_vision_check.py (Y4: shuffle drops with paired-bootstrap CIs + yes/no by question form) on the best KD run at its best-overall-NLL checkpoint, and per-item dev dumps (peritem_<job>_{all,noul}.npz) of
all three runs for paired bootstraps against the round-3 TSIG run (scripts/paired_bootstrap.py).
Inputs are found by SEARCHING /kaggle/input (depth 6): viso-code | vivqa-s1 | vivqa-siglip2-packed | vivqa-siglip2-text-cache | vivqa-teacher-kd.
Output: /kaggle/working/probes/<job>.txt, ckpt/<job>_best_{all,noul,score,choice}.pth (+ .json), <job>_history.json, Y4_<job>_*.txt, peritem_*.npz, summary.json (rewritten after every job; atomic).
Local dry run: PROBES_DRY=1 PROBES_INPUT_ROOT=<dir with the 5 dataset folders> PROBES_WORK=<disk dir, NOT /tmp> python viso_probes4_kernel.py   (CPU, a few steps)."""


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


def _meta_model(d):
    try:
        return json.load(open(os.path.join(d, "meta.json"))).get("model", "")
    except (OSError, ValueError):
        return ""


def locate():
    code = find_dir(lambda d: os.path.isfile(os.path.join(d, "train_viso.py")) and os.path.isfile(os.path.join(d, "scripts", "viso_probe.py")))
    s1 = find_dir(lambda d: os.path.isfile(os.path.join(d, "data", "train.jsonl")) and os.path.isdir(os.path.join(d, "s1bench")))
    packed = find_dir(lambda d: os.path.isfile(os.path.join(d, "regions_fp16.bin")) and os.path.isfile(os.path.join(d, "meta.json")))
    tsig = find_dir(lambda d: os.path.isfile(os.path.join(d, "q_index.npz")) and "siglip2" in _meta_model(d).lower())
    teacher = find_dir(lambda d: os.path.isfile(os.path.join(d, "teacher_train.jsonl")))
    miss = [n for n, v in (("viso-code", code), ("vivqa-s1", s1), ("vivqa-siglip2-packed", packed), ("vivqa-siglip2-text-cache", tsig), ("vivqa-teacher-kd", teacher)) if v is None]
    if miss:
        raise SystemExit(f"input datasets not found under {INPUT}: {miss}")
    return code, s1, packed, tsig, os.path.join(teacher, "teacher_train.jsonl")


def ensure_deps():
    import importlib
    for mod, pkg in (("yacs", "yacs"), ("termcolor", "termcolor"), ("tabulate", "tabulate"), ("joblib", "joblib"), ("sacremoses", "sacremoses"), ("pytorch_transformers", "pytorch_transformers")):
        try:
            importlib.import_module(mod)
        except ImportError:
            r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", pkg], capture_output=True, text=True)
            print(f"pip install {pkg}: {'ok' if r.returncode == 0 else r.stderr[-300:]}", flush=True)


CODE, S1, PACKED, TSIG, TEACHER = locate()
SUMMARY = {"inputs": {"code": CODE, "s1": S1, "packed": PACKED, "tsig": TSIG, "teacher": TEACHER}, "dry_run": DRY, "bar": BAR, "probes": {}, "events": []}
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
TSIG_BASE = F2 + ["TRAINING.TEXT_PROJ_AS_HEADS", "True"]
BASE = {"tsig": "configs/viso_siglip2_tsig.yaml"}
STEPS = 6000
# job id -> (config key, opts, family)
JOBS = {
    "KD5": ("tsig", TSIG_BASE + ["LOSS.KD_WEIGHT", "0.5", "DATASET.FEATURE_DATASET.TEACHER_TRAIN", TEACHER], "kd"),
    "KD8": ("tsig", TSIG_BASE + ["LOSS.KD_WEIGHT", "0.8", "DATASET.FEATURE_DATASET.TEACHER_TRAIN", TEACHER], "kd"),
    "TSIG_s1": ("tsig", TSIG_BASE + ["TRAINING.SEED", "1"], "ref"),
}
ORDER = ["KD5", "KD8", "TSIG_s1"]


def common_opts(cfg_key):
    o = ["DATASET.FEATURE_DATASET.PACKED_PATH", PACKED, "DATASET.VOCAB.S1_DIR", os.path.join(S1, "data"), "DATASET.FEATURE_DATASET.WORKERS", "0" if DRY else "2"]
    if cfg_key == "tsig":
        o += ["DATASET.FEATURE_DATASET.TEXT_CACHE", TSIG]
    if DRY:
        o += ["MODEL.DEVICE", "cpu", "DATASET.FEATURE_DATASET.LIMIT", "96", "DATASET.FEATURE_DATASET.BATCH_SIZE", "16"]
    else:
        o += ["MODEL.DEVICE", "cuda"]
    return o


def read_history(jid):
    p = os.path.join(WORK, "ckpt", f"{jid}_history.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def run_probe(jid, gpu):
    cfg_key, opts, family = JOBS[jid]
    log = os.path.join(WORK, f"{jid}.txt")
    st = ["--steps", "12", "--eval-every", "6", "--grad-at", "4"] if DRY else ["--steps", str(STEPS), "--eval-every", "500", "--grad-at", "500"]
    cmd = [sys.executable, "-B", os.path.join(CODE, "scripts", "viso_probe.py"), "--config-file", os.path.join(CODE, BASE[cfg_key]), *st, "--work", os.path.join(WORK, "work_" + jid),
           "--ckpt-dir", os.path.join(WORK, "ckpt"), "--ckpt-name", jid, "--opts", *opts, *common_opts(cfg_key)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VIVQA_S1_ROOT=S1)
    if gpu is not None and not DRY:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t = time.time()
    event(f"start {jid} on gpu {gpu}")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    h = read_history(jid)
    rec = {"returncode": rc, "seconds": round(time.time() - t), "family": family, "opts": opts, "config": BASE[cfg_key], "steps": STEPS}
    if h:
        rec.update(final_dev=h["final"]["dev"] if h.get("final") else None, best={k: {"step": v["step"], "nll": v["nll"], "acc": v["acc"], "dev_at_best": v["dev"]} for k, v in h["best"].items()},
                   yes_no_curve=[{"step": e["step"], "train_noul_loss": (e.get("train_loss") or {}).get("noul"), "dev_noul_acc": e["dev"].get("noul", [None, None])[0], "dev_noul_nll": e["dev"].get("noul", [None, None])[1],
                                  "dev_score_acc": e["dev"].get("score", [None])[0], "dev_choice_acc": e["dev"].get("choice", [None])[0], "dev_all_nll": e["dev"].get("all", [None, None])[1]} for e in h["history"]])
    with LOCK:
        SUMMARY["probes"][jid] = rec
    event(f"done {jid} rc={rc}; best overall dev NLL {rec.get('best', {}).get('all', {}).get('nll')} at step {rec.get('best', {}).get('all', {}).get('step')}")


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


def analyse(jid, gpu, key="all"):
    """Y4 (shuffle drops with CIs + yes/no by question form) on the best-dev-NLL checkpoint of one run (key = all | noul | score | choice) + the per-item dev results for paired comparisons"""
    ck = os.path.join(WORK, "ckpt", f"ckpt_{jid}_best_{key}.pth")
    if not os.path.isfile(ck):
        return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VIVQA_S1_ROOT=S1)
    if gpu is not None and not DRY:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    extra = ["--limit", "96"] if DRY else []
    with open(os.path.join(WORK, f"Y4_{jid}_{key}.txt"), "w") as f:
        subprocess.run([sys.executable, "-B", os.path.join(CODE, "scripts", "viso_vision_check.py"), "--ckpt", ck, "--device", "cpu" if DRY else "cuda", "--dump-peritem", os.path.join(WORK, f"peritem_{jid}_{key}.npz"), *extra],
                       cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT)
    event(f"Y4 done for {jid} (best {key})")


def peritem_only(jid, gpu, key):
    """per-item dev results of one checkpoint (cheap): peritem_<job>_<key>.npz"""
    ck = os.path.join(WORK, "ckpt", f"ckpt_{jid}_best_{key}.pth")
    if not os.path.isfile(ck):
        return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VIVQA_S1_ROOT=S1)
    if gpu is not None and not DRY:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    extra = ["--limit", "96"] if DRY else []
    with open(os.path.join(WORK, f"peritem_{jid}_{key}.log"), "w") as f:
        subprocess.run([sys.executable, "-B", os.path.join(CODE, "scripts", "viso_vision_check.py"), "--ckpt", ck, "--device", "cpu" if DRY else "cuda", "--dump-peritem", os.path.join(WORK, f"peritem_{jid}_{key}.npz"),
                        "--only-peritem", *extra], cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT)


def main():
    ensure_deps()
    import torch
    n_gpu = 2 if DRY else max(1, torch.cuda.device_count())
    SUMMARY["env"] = {"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "n_workers": n_gpu}
    event(f"inputs: {SUMMARY['inputs']} | workers: {n_gpu}")
    run_pool(ORDER, n_gpu)
    P = SUMMARY["probes"]
    kd = [j for j in ("KD5", "KD8") if j in P and P[j].get("best")]
    best_kd = min(kd, key=lambda j: P[j]["best"]["all"]["nll"]) if kd else None
    SUMMARY["best_kd_run"] = best_kd
    for g, (j, k) in enumerate([(j, k) for j in ORDER if j in P and P[j].get("best") for k in ("all", "noul")]):
        peritem_only(j, g % n_gpu, k)
    if best_kd:
        analyse(best_kd, 0, "all")
        analyse(best_kd, 0, "noul")
    event("all done")


if __name__ == "__main__":
    main()
