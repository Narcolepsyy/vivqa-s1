#!/usr/bin/env python3
"""Run bookkeeping for run.sh (stdlib only; every file is written atomically: temp + rename).
  ezy_status.py status  --results R [--job J --status S [--seconds N]] [--queue a,b,c]    update R/RUN_STATUS.json (per job: status, seconds, files)
  ezy_status.py markers --results R --state ST                                           (re)create ST/done_<job> from every COMPLETE summary_<job>.json whose result files exist;
                                                                                          remove a done_<job> whose files are gone. Prints one line per job.
A job is complete when its summary says status 'complete', every stage returned 0, and the files below exist (shards are NOT required: a finished job needs only its results)."""
import argparse
import json
import os
import sys
import time


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def required(job, summary):
    if job.startswith("viso-ft-"):
        return ["ft_run.json", "calibration/temperatures.json", "preds_test.jsonl", "ft_report/ft_report.json", "best_model.pth"]
    if job.startswith("lora-"):
        return ["lora_s0/views.json", "adapter/adapter_model.safetensors", "head.pt", "meta.json"]
    if job.startswith("ref-") or (summary or {}).get("zero_shot_only"):
        return ["zeroshot/views.json", "meta.json"]
    return ["zeroshot/views.json", "head_s0/views.json", "meta.json"]


def job_complete(results, job):
    """(complete?, reason)"""
    s = load(os.path.join(results, f"summary_{job}.json"))
    if not s and job.startswith("lora-"):                    # a LoRA job started outside run.sh (e.g. /root/run_lora2b.sh) has no summary: its last file is lora_s0/views.json (written after the predictions)
        missing = [r for r in required(job, None) if not os.path.exists(os.path.join(results, job, r))]
        return (not missing), ("complete by its result files (no summary_<job>.json)" if not missing else f"missing files {missing}")
    if not s:
        return False, "no summary"
    if s.get("status") != "complete":
        return False, f"summary status: {s.get('status')}"
    bad = {k: v.get("returncode") for k, v in s.get("stages", {}).items() if v.get("returncode") != 0}
    if bad:
        return False, f"stage return codes {bad}"
    missing = [r for r in required(job, s) if not os.path.exists(os.path.join(results, job, r))]
    if missing:
        return False, f"missing files {missing}"
    return True, "ok"


def files_of(results, job):
    d = os.path.join(results, job)
    info = {"n_files": 0, "bytes": 0, "shards": {"n": 0, "bytes": 0}, "files": []}
    for root, _, fs in os.walk(d):
        for f in sorted(fs):
            if ".tmp" in f or f.endswith(".bad"):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, d)
            size = os.path.getsize(p)
            info["n_files"] += 1
            info["bytes"] += size
            if rel.startswith("shards" + os.sep):
                info["shards"]["n"] += 1
                info["shards"]["bytes"] += size
            elif len(info["files"]) < 300:
                info["files"].append({"path": rel, "bytes": size})
    top = [f for f in sorted(os.listdir(results)) if (f.startswith(job + "_") or f in (f"summary_{job}.json", f"run_info_{job}.txt")) and ".tmp" not in f] if os.path.isdir(results) else []
    info["top_level_files"] = top
    return info


def cmd_status(a):
    path = os.path.join(a.results, "RUN_STATUS.json")
    st = load(path) or {"jobs": {}}
    if a.queue:
        st["queue"] = a.queue.split(",")
    if a.job:
        j = st["jobs"].setdefault(a.job, {})
        j["status"] = a.status
        if a.status == "running":
            j["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            j.pop("finished_at", None)
        else:
            j["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if a.seconds is not None:
            j["seconds"] = a.seconds
        ok, why = job_complete(a.results, a.job)
        j["results_complete"], j["results_check"] = ok, why
        j.update(files_of(a.results, a.job))
    for q in st.get("queue", []):
        st["jobs"].setdefault(q, {"status": "pending"})
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    st["all_complete"] = bool(st.get("queue")) and all(st["jobs"].get(q, {}).get("status") in ("complete", "done earlier") for q in st["queue"])
    os.makedirs(a.results, exist_ok=True)
    atomic_json(path, st)


def cmd_markers(a):
    os.makedirs(a.state, exist_ok=True)
    jobs = sorted({f[len("summary_"):-len(".json")] for f in os.listdir(a.results) if f.startswith("summary_") and f.endswith(".json")}) if os.path.isdir(a.results) else []
    if os.path.isdir(a.results):                                     # LoRA jobs started outside run.sh have no summary file: judge them by their result files
        jobs += [d for d in os.listdir(a.results) if d.startswith("lora-") and os.path.isdir(os.path.join(a.results, d)) and d not in jobs]
    for f in os.listdir(a.state):                                    # markers without any summary: stale
        if f.startswith("done_") and f[5:] not in jobs:
            jobs.append(f[5:])
    for job in sorted(set(jobs)):
        ok, why = job_complete(a.results, job)
        marker = os.path.join(a.state, f"done_{job}")
        if ok and not os.path.exists(marker):
            open(marker, "w").close()
            print(f"{job}: complete -> done marker created")
        elif ok:
            print(f"{job}: complete (marker present)")
        elif os.path.exists(marker):
            os.remove(marker)
            print(f"{job}: marker removed ({why})")
        else:
            print(f"{job}: not complete ({why})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--results", required=True)
    s.add_argument("--job")
    s.add_argument("--status")
    s.add_argument("--seconds", type=int)
    s.add_argument("--queue")
    m = sub.add_parser("markers")
    m.add_argument("--results", required=True)
    m.add_argument("--state", required=True)
    a = ap.parse_args()
    {"status": cmd_status, "markers": cmd_markers}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
