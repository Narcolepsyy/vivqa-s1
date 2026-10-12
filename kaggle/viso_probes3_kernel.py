"""Kaggle script kernel, round 3 (planner decisions after round 2). Base config = F2 (FUSION_PRODUCT + VISION_PATH_AS_HEADS, low LR enc 5e-5 / heads 5e-4, warmup 200, strict text mask) + the frozen
standardised BamiBERT features (= B1c). 6000 steps, dev every 500, NO early stop; the BEST-dev-NLL checkpoint (overall and per type) is saved and the metrics AT the best step are reported next to the final.
Jobs: B1c_ref | B1c_reg (B1c + weight decay 0.05 + dropout 0.2 on the fusion vector and the heads) | TSIG (same, with the frozen SigLIP 2 TEXT tower features) | BUTD (B-BUTD baseline: GRU + question-guided
attention + Hadamard fusion + the typed heads of ViSO, one LR 5e-4 for all parameters) | F2_scratch_ref (from-scratch text encoder). 5 jobs on a worker queue (one worker per GPU).
Then Y4 (viso_vision_check.py: shuffle drops with paired-bootstrap CIs + yes/no by question form) on the BEST-NLL checkpoint of the best run of each family (BamiBERT, SigLIP text, BUTD, scratch).
Inputs are found by SEARCHING /kaggle/input (depth 6): viso-code | vivqa-s1 | vivqa-siglip2-packed | vivqa-bamibert-cache | vivqa-siglip2-text-cache (told apart by meta.json 'model').
Output: /kaggle/working/probes/<job>.txt, ckpt/<job>_best_{all,noul,score,choice}.pth (+ .json), <job>_history.json, Y4_<job>.txt, summary.json (rewritten after every job).
Local dry run: PROBES_DRY=1 PROBES_INPUT_ROOT=<dir with the 5 dataset folders> PROBES_WORK=<dir> python viso_probes3_kernel.py   (CPU, a few steps)."""

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
    bami = find_dir(lambda d: os.path.isfile(os.path.join(d, "q_index.npz")) and "bami" in _meta_model(d).lower())
    xyn = find_dir(lambda d: os.path.isfile(os.path.join(d, "xyn_partial_train.jsonl")))
    tsig = find_dir(lambda d: os.path.isfile(os.path.join(d, "q_index.npz")) and "siglip2" in _meta_model(d).lower())
    miss = [n for n, v in (("viso-code", code), ("vivqa-s1", s1), ("vivqa-siglip2-packed", packed), ("vivqa-bamibert-cache", bami)) if v is None]     # tsig / xyn are checked in main() for the jobs that need them
    if miss:
        raise SystemExit(f"input datasets not found under {INPUT}: {miss}")
    return code, s1, packed, bami, tsig, os.path.join(xyn, "xyn_partial_train.jsonl") if xyn else None


def ensure_deps():
    import importlib
    for mod, pkg in (("yacs", "yacs"), ("termcolor", "termcolor"), ("tabulate", "tabulate"), ("joblib", "joblib"), ("sacremoses", "sacremoses"), ("pytorch_transformers", "pytorch_transformers")):
        try:
            importlib.import_module(mod)
        except ImportError:
            r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", pkg], capture_output=True, text=True)
            print(f"pip install {pkg}: {'ok' if r.returncode == 0 else r.stderr[-300:]}", flush=True)


CODE, S1, PACKED, BAMI, TSIG, XYN = locate()
SUMMARY = {"inputs": {"code": CODE, "s1": S1, "packed": PACKED, "bami": BAMI, "tsig": TSIG, "xyn": XYN}, "dry_run": DRY, "bar": BAR, "probes": {}, "events": []}
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
B1C = F2 + ["TRAINING.TEXT_PROJ_AS_HEADS", "True"]
BASE = {"viso": "configs/viso_siglip2.yaml", "bami": "configs/viso_siglip2_bami.yaml", "tsig": "configs/viso_siglip2_tsig.yaml", "butd": "configs/viso_butd_siglip2.yaml"}
STEPS = 6000
# job id -> (config key, opts, family)
JOBS = {
    "B1c_ref": ("bami", B1C, "bami"),
    "B1c_reg": ("bami", B1C + ["TRAINING.WEIGHT_DECAY", "0.05", "MODEL.HEAD_DROPOUT", "0.2", "MODEL.FUSION_DROPOUT", "0.2"], "bami"),
    "TSIG": ("tsig", B1C, "tsig"),
    "BUTD": ("butd", ["TRAINING.LR_HEADS", "0.0005", "TRAINING.WARMUP_STEPS", "200"], "butd"),
    "F2_scratch_ref": ("viso", F2, "scratch"),
    # data-effect runs on the B1c config: yes/no train subsets (learning curve; the 100% point is B1c_ref) and the direct addition of the translated real VQA v2 yes/no questions
    "LC25": ("bami", B1C + ["DATASET.FEATURE_DATASET.NOUL_TRAIN_FRACTION", "0.25"], "lc"),
    "LC50": ("bami", B1C + ["DATASET.FEATURE_DATASET.NOUL_TRAIN_FRACTION", "0.5"], "lc"),
    "X7K": ("bami", B1C + ["DATASET.FEATURE_DATASET.EXTRA_TRAIN", XYN], "x7k"),
}
SET = os.environ.get("PROBES_SET", "ALL")                          # ALL | A (B1c_ref + learning curve + X7K) | B (the rest): two kernels if one is too long
ORDER = {"ALL": ["B1c_ref", "X7K", "TSIG", "LC50", "BUTD", "LC25", "B1c_reg", "F2_scratch_ref"], "A": ["B1c_ref", "X7K", "LC50", "LC25"], "B": ["TSIG", "BUTD", "B1c_reg", "F2_scratch_ref"]}[SET]
YES_NO_STEP = ("B1c_ref", "LC25", "LC50", "X7K")                   # also analysed at the best yes/no-NLL step (+ per-item dumps for paired comparisons)


def common_opts(cfg_key):
    o = ["DATASET.FEATURE_DATASET.PACKED_PATH", PACKED, "DATASET.VOCAB.S1_DIR", os.path.join(S1, "data"), "DATASET.FEATURE_DATASET.WORKERS", "0" if DRY else "2"]
    if cfg_key == "bami":
        o += ["DATASET.FEATURE_DATASET.TEXT_CACHE", BAMI]
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


def main():
    if "TSIG" in ORDER and TSIG is None:
        raise SystemExit("input dataset not found: vivqa-siglip2-text-cache (needed by TSIG)")
    if "X7K" in ORDER and XYN is None:
        raise SystemExit("input dataset not found: vivqa-xyn-partial (needed by X7K)")
    ensure_deps()
    import torch
    n_gpu = 2 if DRY else max(1, torch.cuda.device_count())
    SUMMARY["env"] = {"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "n_workers": n_gpu}
    event(f"inputs: {SUMMARY['inputs']} | workers: {n_gpu}")
    run_pool(ORDER, n_gpu)
    P = SUMMARY["probes"]
    best_of = {}
    for jid, rec in P.items():                                # the best run of each family = lowest best-overall dev NLL
        nll = rec.get("best", {}).get("all", {}).get("nll")
        if nll is not None and (rec["family"] not in best_of or nll < P[best_of[rec["family"]]]["best"]["all"]["nll"]):
            best_of[rec["family"]] = jid
    SUMMARY["best_run_per_family"] = best_of
    jobs_an = [(j, "all") for j in best_of.values()] + [(j, "noul") for j in YES_NO_STEP if j in P and P[j].get("best")]
    ts = [threading.Thread(target=analyse, args=(j, g % n_gpu, k)) for g, (j, k) in enumerate(jobs_an)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    event("all done")


if __name__ == "__main__":
    main()
