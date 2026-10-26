"""Kaggle script kernel: ONE external VLM baseline in System One mode (scripts/s1_vlm_eval.py): frozen model, ONE forward pass per item, Vietnamese prompt with lettered options, letter-token logits;
zero-shot + head-only (3 seeds), per-type temperature on calib, Image-unseen headline. One kernel per model (VLM_MODEL, set in the kernel folder's copy of this file).
Inputs found by SEARCHING /kaggle/input (depth 6): viso-code, vivqa-s1, and ANY public COCO 2014 dataset (files named COCO_train2014_%012d.jpg / COCO_val2014_%012d.jpg anywhere below /kaggle/input): every image
read is verified against kaggle/s1_images_sha256.tsv.gz (the benchmark images are the original COCO files); an image that is missing is downloaded from images.cocodataset.org (internet on) and verified the same way;
a mismatch stops the run. A previous run's output attached as an input (kernel source) is resumed: its shards are copied in and skipped.
Order of the features stage: calib, dev, test, then train (zero-shot is possible as soon as calib/dev/test are done; head-only needs train). The time budget (VLM_MAX_HOURS, default 10.5 h) stops the stage
after the current shard; the run then ends with status 'incomplete' (attach this kernel's output to the next run to continue).
Output: /kaggle/working/vlm/<key>/ {meta.json, shards/, zeroshot/, head_s*/}, vlm_report.md, summary.json (rewritten after every stage).
Observability: every stage streams to the console AND /kaggle/working/vlm/<stage>.log (line by line); the features stage prints startup facts (versions, whether fla's Triton kernels or the slow torch fallback run, GPU, image counts),
a heartbeat every 60 s (units done / expected, units/s, ETA, GPU memory) and a watchdog (no unit for 10 min -> stack dump + stage exit rc 3).
VLM_PREFLIGHT=1: speed test only (calib features for VLM_PREFLIGHT_MIN minutes, default 15) -> preflight.json with units/s and the projected hours of the full run; no other stage runs.
VLM_HEAD_TRAIN_LIMIT=N (default 0 = all): features + head use a seeded type-stratified subset of N train items.
Local dry run: VLM_DRY=1 VLM_INPUT_ROOT=<dir with the datasets> VLM_WORK=<dir> python vlm_kernel.py  (24 items per split)."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

MODELS = {
    "onejev": ("OmniJev/OneJev-0.8B", "c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8", "apache-2.0"),
    "intern": ("internlm/Intern-Decision-0.8B", "85a0cc5a99d67ea8d56dfe98115689212867171d", "apache-2.0"),
    "jpt": ("kirp/jpt-0.8b", "1431c0509bbc10772cd59964cc7af5835c8720d6", "CC-BY-NC-4.0 (non-commercial)"),
    "qwen": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "apache-2.0"),
    "minicpm": ("openbmb/MiniCPM-V-4.7-1B", "6fde9ecee3e59c4b0ce555a9c774e179d46d19cc", "apache-2.0"),
    "d3-lite": ("vllm-sr/d3-lite", "99e5b1ce5a718454cd112cfe00caa8d2ae8df2ff", "apache-2.0"),
    "d3-nano": ("vllm-sr/d3-nano", "80820624a8170104679aa99897a6e9026eae0790", "apache-2.0"),
    # reference tier (zero-shot only: calib / dev / test, no train features, no head); run on a rented 24 GB GPU (kaggle/ezy), fp16, no quantisation
    "ref-intern2b": ("internlm/Intern-Decision-2B", "8797836c65fc91a2435b1fb6850b5f0aabd75cc3", "apache-2.0"),
    "ref-intern4b": ("internlm/Intern-Decision-4B", "0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd", "apache-2.0"),
    "ref-qwen3vl8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "apache-2.0"),
    # the ONLY model run with trust_remote_code (user decision): pinned revision, the sha256 of its 5 remote .py files are checked against a reviewed table (scripts/vlm_adapters.py); InternVL dynamic tiling, 448 px tiles
    "vintern": ("5CD-AI/Vintern-1B-v3_5", "b98f263eab246eb5269ade64edbdca8a887dc44d", "MIT"),
}
NO_FLA = {"vintern"}                                                  # models without linear-attention layers: the fla backend check does not apply
ZERO_SHOT_ONLY = {k for k in MODELS if k.startswith("ref-")}
BATCH = {"ref-intern2b": 8, "ref-intern4b": 4, "ref-qwen3vl8b": 4}
KEY = os.environ.get("VLM_MODEL", "qwen")
DRY = os.environ.get("VLM_DRY") == "1"
INPUT = os.environ.get("VLM_INPUT_ROOT", "/kaggle/input")
WORK = os.path.abspath(os.environ.get("VLM_WORK", "/kaggle/working/vlm"))
MAX_H = float(os.environ.get("VLM_MAX_HOURS", "10.5"))
WORKERS = os.environ.get("VLM_WORKERS", "2")                       # DataLoader workers (image decode + processor): Kaggle 2; a many-core rental sets more (kaggle/ezy/run.sh)
PREFLIGHT = os.environ.get("VLM_PREFLIGHT") == "1"
PREFLIGHT_MIN = float(os.environ.get("VLM_PREFLIGHT_MIN", "15"))
HEAD_LIMIT = int(os.environ.get("VLM_HEAD_TRAIN_LIMIT", "0"))
SHARED = os.environ.get("VLM_SHARED_WORK") == "1"                    # several models share one VLM_WORK: stage logs / summary are prefixed with the model key
PFX = f"{KEY}_" if SHARED else ""
T0 = time.time()
MODEL_DIR = os.path.join(WORK, KEY)
os.makedirs(MODEL_DIR, exist_ok=True)
SUMMARY = {"model_key": KEY, "model": MODELS[KEY][0], "revision": MODELS[KEY][1], "licence": MODELS[KEY][2], "dry_run": DRY, "preflight": PREFLIGHT, "head_train_limit": HEAD_LIMIT, "stages": {}, "status": "running"}


def save():
    SUMMARY["elapsed_s"] = round(time.time() - T0)
    name = f"summary_{KEY}.json" if SHARED else "summary.json"
    tmp = os.path.join(WORK, name + ".tmp")
    json.dump(SUMMARY, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, os.path.join(WORK, name))


def find_dir(test, depth=6):
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


def ensure_compiler():
    """Triton JIT-builds a C launcher: without gcc + Python.h flash-linear-attention silently falls back to CPU. Try to install them (root + internet on Kaggle) before fla is imported."""
    import sysconfig
    has_h = os.path.isfile(os.path.join(sysconfig.get_paths()["include"], "Python.h"))
    if shutil.which("gcc") and has_h:
        say("C compiler: gcc + Python.h present")
        return
    say(f"C compiler: gcc {'present' if shutil.which('gcc') else 'MISSING'}, Python.h {'present' if has_h else 'MISSING'} -> apt-get install build-essential python3-dev")
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    for cmd in (["apt-get", "update", "-qq"], ["apt-get", "install", "-y", "-qq", "build-essential", "python3-dev"]):
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode:
            say(f"  {' '.join(cmd[:2])} failed: {r.stderr[-200:]}")
            return


def check_fla():
    """fla must report the 'cuda' backend on a GPU run; otherwise stop before hours are wasted (VLM_ALLOW_SLOW_FLA=1 overrides)"""
    try:
        from fla.utils._device import get_available_device
        be = str(get_available_device())
    except Exception as e:
        be = f"unavailable ({type(e).__name__})"
    SUMMARY["fla_backend"] = be
    say(f"fla backend: {be}")
    gpu = __import__("torch").cuda.is_available()
    if gpu and KEY not in NO_FLA and not be.startswith("unavailable") and be != "cuda" and os.environ.get("VLM_ALLOW_SLOW_FLA") != "1":
        SUMMARY["status"] = f"failed: fla backend is '{be}' (no gcc / Python.h for Triton?)"
        save()
        raise SystemExit(f"STOP: fla backend is '{be}', not 'cuda': the run would be hours slower. See summary.json.")


def ensure_deps():
    import importlib
    wanted = [("transformers", "transformers==5.19.0"), ("fla", "flash-linear-attention"), ("einops", "einops"), ("timm", "timm"), ("pytorch_transformers", "pytorch_transformers"), ("yacs", "yacs"), ("termcolor", "termcolor"), ("tabulate", "tabulate"), ("joblib", "joblib")]
    for mod, pkg in wanted:
        try:
            m = importlib.import_module(mod)
            if mod == "transformers" and m.__version__ != "5.19.0":
                raise ImportError("pin")
        except ImportError:
            r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", pkg], capture_output=True, text=True)
            print(f"pip install {pkg}: {'ok' if r.returncode == 0 else r.stderr[-300:]}", flush=True)


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_list():
    """the image sha list: .tsv.gz in the repo, but Kaggle AUTO-DECOMPRESSES uploaded .gz files, so a mounted dataset has the plain .tsv"""
    base = os.path.join(CODE, "kaggle", "s1_images_sha256.tsv")
    for p in (base + ".gz", base):
        if os.path.isfile(p):
            return p
    raise SystemExit(f"image sha list not found: {base}[.gz]")


def run_stage(stage, extra, log_name):
    log_name = PFX + log_name
    """child process output goes to the console AND the log file, line by line (the Kaggle console stays alive)"""
    cmd = [sys.executable, "-u", "-B", os.path.join(CODE, "scripts", "s1_vlm_eval.py"), stage, "--model", MODELS[KEY][0], "--revision", MODELS[KEY][1], "--licence", MODELS[KEY][2], "--model-dir", MODEL_DIR,
           "--s1-data", os.path.join(S1, "data"), "--root", S1, "--root-models", WORK, "--image-sha", sha_list(), "--image-roots", INPUT,
           "--image-cache", os.path.join(WORK, "coco_cache"), "--download-images", *extra]
    if KEY == "vintern" and stage == "features":
        cmd += ["--trust-remote-code", "--max-tiles", os.environ.get("VLM_MAX_TILES", "4")]
    if HEAD_LIMIT and stage in ("features", "head"):
        cmd += ["--head-train-limit", str(HEAD_LIMIT)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", VIVQA_S1_ROOT=S1, HF_HUB_DISABLE_PROGRESS_BARS="1")
    say(f"=== stage {stage} starts: {' '.join(extra)} (log {log_name})")
    t = time.time()
    with open(os.path.join(WORK, log_name), "a") as f:
        proc = subprocess.Popen(cmd, cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line); f.flush()
        rc = proc.wait()
    SUMMARY["stages"][stage] = {"returncode": rc, "seconds": round(time.time() - t), "log": log_name}
    say(f"=== stage {stage} ended rc={rc} after {time.time() - t:.0f}s")
    save()
    return rc


def startup_report():
    """facts that explain a slow or failing run, printed before anything heavy starts"""
    say(f"kernel start: model key {KEY} ({MODELS[KEY][0]}@{MODELS[KEY][1][:8]}), dry={DRY}, preflight={PREFLIGHT}, max_hours={MAX_H}, head_train_limit={HEAD_LIMIT}")
    say(f"python {sys.version.split()[0]}; work dir {WORK}; input root {INPUT}")
    try:
        say("input datasets: " + ", ".join(sorted(os.listdir(INPUT))))
        for d in sorted(os.listdir(INPUT)):
            sub = os.path.join(INPUT, d)
            if os.path.isdir(sub):
                say(f"  {d}: {', '.join(sorted(os.listdir(sub))[:6])}")
    except OSError as e:
        say(f"cannot list {INPUT}: {e}")
    say(f"viso-code found at {CODE}; vivqa-s1 found at {S1}")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        say("gpus: " + r.stdout.strip().replace("\n", " | "))
    except Exception as e:
        say(f"nvidia-smi failed: {e}")
    try:
        du = shutil.disk_usage(WORK)
        say(f"disk at work dir: {du.free / 2**30:.0f} GB free of {du.total / 2**30:.0f} GB")
    except OSError:
        pass


def main():
    global CODE, S1
    CODE = find_dir(lambda d: os.path.isfile(os.path.join(d, "train_viso.py")) and os.path.isfile(os.path.join(d, "scripts", "s1_vlm_eval.py")))
    S1 = find_dir(lambda d: os.path.isfile(os.path.join(d, "data", "train.jsonl")) and os.path.isdir(os.path.join(d, "s1bench")))
    if CODE is None or S1 is None:
        raise SystemExit(f"input datasets not found under {INPUT}: viso-code={CODE}, vivqa-s1={S1}")
    SUMMARY["inputs"] = {"code": CODE, "s1": S1}
    os.environ["VIVQA_S1_ROOT"] = S1                               # utils.viso_s1 (imported by the harness) finds s1bench there
    # resume: shards of a previous run of the SAME model that were attached as an input
    prev = []
    for d, ds, fs in os.walk(INPUT):
        if os.path.basename(d) == "shards" and os.path.basename(os.path.dirname(d)) == KEY:
            prev.append(d)
        if d.count(os.sep) - INPUT.count(os.sep) > 7:
            ds[:] = []
    n_copied = 0
    for d in prev:
        os.makedirs(os.path.join(MODEL_DIR, "shards"), exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.npz")):
            dst = os.path.join(MODEL_DIR, "shards", os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy2(f, dst); n_copied += 1
    SUMMARY["resumed_shards"] = n_copied
    startup_report()
    say(f"resumed shards copied from attached outputs: {n_copied}")
    ensure_compiler()
    ensure_deps()
    check_fla()
    sys.path.insert(0, os.path.join(CODE, "scripts"))
    sys.path.insert(0, CODE)
    import s1_vlm_eval as V
    lim = ["--limit", "24"] if DRY else []
    zs_only = KEY in ZERO_SHOT_ONLY or os.environ.get("VLM_ZS_ONLY") == "1"
    batch = os.environ.get("VLM_BATCH") or str(BATCH.get(KEY, 8))
    sets = ("calib", "dev", "test") if zs_only else ("calib", "dev", "test", "train")
    SUMMARY["zero_shot_only"] = zs_only
    feat = ["--splits", *sets, "--batch", "4" if DRY else batch, "--workers", "0" if DRY else WORKERS, "--shard", "16" if DRY else "512", "--device", "cuda" if not DRY else ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
            "--dtype", "fp16", "--max-hours", str(MAX_H), *lim]
    if PREFLIGHT:                                                  # speed test only: calib features for a few minutes, nothing else
        feat = ["--splits", "calib", "--batch", feat[feat.index("--batch") + 1], "--workers", WORKERS, "--shard", "512", "--device", "cuda", "--dtype", "fp16", "--preflight-minutes", str(PREFLIGHT_MIN)]
        rc = run_stage("features", feat, "features.log")
        pf = os.path.join(MODEL_DIR, "preflight.json")
        SUMMARY["preflight_result"] = json.load(open(pf)) if os.path.isfile(pf) else None
        SUMMARY["status"] = "preflight done" if rc == 0 and SUMMARY["preflight_result"] else f"preflight failed (rc {rc})"
        save()
        say(f"PREFLIGHT SUMMARY: {json.dumps(SUMMARY['preflight_result'])}")
        return
    rc = run_stage("features", feat, "features.log")
    status = {}
    for sp in sets:
        got, want = V.shards_complete(MODEL_DIR, os.path.join(S1, "data"), sp, 24 if DRY else 0, HEAD_LIMIT)
        status[sp] = {"units": got, "expected": want, "complete": got >= want}
    SUMMARY["features_status"] = status
    meta = os.path.join(MODEL_DIR, "meta.json")
    if os.path.isfile(meta):
        m = json.load(open(meta))
        SUMMARY["params"] = m.get("params")
        SUMMARY["params_over_1B"] = bool(m.get("params") and m["params"] > 1.0e9)
        SUMMARY["seconds_per_unit"] = m.get("seconds_per_unit")
    save()
    if all(status[s]["complete"] for s in ("calib", "dev", "test")):
        run_stage("zeroshot", lim, "zeroshot.log")
    if not zs_only and status["train"]["complete"] and status["calib"]["complete"]:
        run_stage("head", [*lim, "--seeds", "2" if DRY else "3", "--epochs", "2" if DRY else "30"], "head.log")
    run_stage("report", [], "report.log")
    SUMMARY["status"] = "complete" if all(s["complete"] for s in status.values()) else "incomplete: attach this kernel's output to the next run to continue"
    save()


if __name__ == "__main__":
    main()
