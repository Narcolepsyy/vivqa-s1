"""Write kaggle/experiments_rlfix.json: the RL-stage-only re-run of A3 from the finished A2 checkpoints of the main matrix (planner decision after the A3 collapse; cause in scripts/viso_rl_gradcheck.py:
the policy-gradient gradient is 4-32x the soft-CE gradient, ~1/sigma, and noisy at K=4).
  python scripts/make_rlfix_manifest.py [--pg-weight 0.05] [--samples 16] [--lr-scale 0.1] [--warmup 50] [--with-control]
Jobs: A3fix_s0..s2 = LOSS.NAME rlcd, LOSS.STAGE rl, started from runs/A2_s<seed>/best_model.pth (copied from the previous output of the main matrix: stage1_from, no depends_on), PG_WEIGHT 0.05, NUM_SAMPLES 16,
LR enc / heads = the main matrix's x 0.1, warmup 50, 5 RL epochs, best-dev-NLL selection, post = calibrate (A4fix = A3fix + per-type temperature + escalation head), same data / config otherwise.
A3fixS_s0 (one seed) = the same with the PG term multiplied by sigma (LOSS.PG_SIGMA_SCALE: the PG gradient scale no longer grows as sigma is annealed 0.4 -> 0.1) and PG_WEIGHT 0.2 (about the same average
gradient ratio as 0.05 without the scaling at the start of the anneal, constant afterwards). --with-control adds A3cfix_s0..s2 (PG_WEIGHT 0: the same extra epochs without the PG term = the A3c control)."""
import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def control_only(a, src, rl, job, est):
    """A3c control run: the RL stage with the PG term switched off (same epochs / LRs / warmup / selection / seeds), next to the reused A3fix and A2 runs"""
    out_path = a.out if a.out.endswith("experiments_rlctl.json") else os.path.join(os.path.dirname(a.out), "experiments_rlctl.json")
    jobs = [job("A3cfix", s, {"LOSS.PG_WEIGHT": 0.0}) for s in (0, 1, 2)]
    base = src["jobs"][0]["base_config"]
    for s in (0, 1, 2):
        jobs.append({"id": f"A3fix_s{s}", "method": "A3fix", "base_config": base, "seed": s, "overrides": rl, "depends_on": [], "stage1_from": f"A2_s{s}", "post": "calibrate", "est_minutes": 1, "reuse": True})
    for s in (0, 1, 2):
        j = next(x for x in src["jobs"] if x["id"] == f"A2_s{s}")
        jobs.append({**json.loads(json.dumps(j)), "post": "calibrate", "depends_on": [], "est_minutes": 1, "reuse": True})
    m = src["methods"]
    methods = {"A2": m["A2"], "A2T": m["A2T"],
               "A3fix": {"family": "A3fix", "variant": "uncal", "description": "A2 + RLCD stage, fixed (PG weight 0.05, K = 16, RL LRs x 0.1): reused from viso-rlfix"},
               "A4fix": {"family": "A3fix", "variant": "pertype", "description": "A3fix + per-type temperature"},
               "A3cfix": {"family": "A3cfix", "variant": "uncal", "description": "A3c control: the same RL-stage epochs / LRs / warmup / selection from A2 with PG weight 0 (soft-CE only)"},
               "A4cfix": {"family": "A3cfix", "variant": "pertype", "description": "A3cfix + per-type temperature"}}
    comps = [["A3fix", "A3cfix"], ["A3cfix", "A2"], ["A3fix", "A2"], ["A4fix", "A2T"], ["A4cfix", "A2T"], ["A4fix", "A4cfix"]]
    out = {**{k: v for k, v in src.items() if k not in ("jobs", "methods", "comparisons", "note", "common_overrides")}, "version": 5,
           "note": "A3c control: is the A3fix gain from the RL term or from extra low-LR soft-CE epochs? A3cfix is trained here (PG weight 0); A3fix is copied from the viso-rlfix output, A2 from viso-matrix.",
           "common_overrides": src["common_overrides"], "jobs": jobs, "methods": methods, "comparisons": comps}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}: {[(j['id'], 'reuse' if j.get('reuse') else 'train') for j in jobs]}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(REPO, "kaggle", "experiments_lean.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "kaggle", "experiments_rlfix.json"))
    ap.add_argument("--pg-weight", type=float, default=0.05)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--lr-scale", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--variant-weight", type=float, default=0.2)
    ap.add_argument("--with-control", action="store_true")
    ap.add_argument("--control-only", action="store_true", help="kaggle/experiments_rlctl.json: ONLY the A3c control (A3cfix_s0..s2, PG weight 0) is trained; A3fix (viso-rlfix output) and A2 (viso-matrix output) are reused")
    a = ap.parse_args(argv)
    src = json.load(open(a.src))
    common = dict(src["common_overrides"])
    base = next(j["base_config"] for j in src["jobs"] if j["id"] == "A3_s0")
    rl = {**common, "LOSS.NAME": "rlcd", "LOSS.STAGE": "rl", "LOSS.PG_WEIGHT": a.pg_weight, "LOSS.NUM_SAMPLES": a.samples,
          "TRAINING.LR_ENCODER": common["TRAINING.LR_ENCODER"] * a.lr_scale, "TRAINING.LR_HEADS": common["TRAINING.LR_HEADS"] * a.lr_scale, "TRAINING.WARMUP_STEPS": a.warmup}
    est = next(j["est_minutes"] for j in src["jobs"] if j["id"] == "A3_s0")

    def job(method, seed, over):
        return {"id": f"{method}_s{seed}", "method": method, "base_config": base, "seed": seed, "overrides": {**rl, **over}, "depends_on": [], "stage1_from": f"A2_s{seed}", "post": "calibrate",
                "est_minutes": est}
    jobs = [job("A3fix", s, {}) for s in (0, 1, 2)] + [job("A3fixS", 0, {"LOSS.PG_SIGMA_SCALE": True, "LOSS.PG_WEIGHT": a.variant_weight})]
    methods = {"A2": src["methods"]["A2"],
               "A3fix": {"family": "A3fix", "variant": "uncal", "description": f"A2 + RLCD stage, fixed: PG weight {a.pg_weight}, K = {a.samples}, RL LRs x {a.lr_scale}, warmup {a.warmup}"},
               "A4fix": {"family": "A3fix", "variant": "pertype", "description": "A3fix + per-type temperature (+ escalation head)"},
               "A3fixS": {"family": "A3fixS", "variant": "uncal", "description": f"A3fix with the PG term scaled by sigma (LOSS.PG_SIGMA_SCALE), PG weight {a.variant_weight} (1 seed)"}}
    comps = [["A3fix", "A2"], ["A4fix", "A3fix"], ["A3fixS", "A2"]]
    if a.with_control:
        jobs += [job("A3cfix", s, {"LOSS.PG_WEIGHT": 0.0}) for s in (0, 1, 2)]
        methods["A3cfix"] = {"family": "A3cfix", "variant": "uncal", "description": "A3c control of A3fix: the same RL-stage epochs with PG weight 0 (soft-CE only)"}
        comps += [["A3fix", "A3cfix"], ["A3cfix", "A2"]]
    if a.control_only:
        return control_only(a, src, rl, job, est)
    out = {**{k: v for k, v in src.items() if k not in ("jobs", "methods", "comparisons", "note", "common_overrides")}, "version": 3,
           "note": "RL-FIX re-run: RL stage only from the A2 checkpoints of the main matrix (attach viso-matrix; the A2 runs are found by name in its viso_out). Cause of the original A3 collapse: scripts/viso_rl_gradcheck.py.",
           "common_overrides": common, "jobs": jobs, "methods": methods, "comparisons": comps}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {[j['id'] for j in jobs]}")
    return out


if __name__ == "__main__":
    main()
