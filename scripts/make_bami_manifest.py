"""Write kaggle/experiments_bami.json: BamiBERT becomes the MAIN ViSO encoder (planner / user decision after the matrix: statistically tied with T-SIG, 233 M vs 413 M parameters on the device).
  python scripts/make_bami_manifest.py
Jobs (kaggle kernel narcolepsyy/viso-bami):
  A2bami_s1, A2bami_s2   TRAINED here: exactly the ABL_bami_s0 configuration of the matrix (frozen BamiBERT standardised cache, F2, TEXT_PROJ_AS_HEADS, strict mask, low LR, 12 epochs / patience 5, best-dev-NLL), post = calibrate
  A2bami_s0              REUSE: the matrix run ABL_bami_s0 copied under this name (same configuration, seed 0; the kernel writes ALIAS.json next to it), calibrated here
  A2_s0..s2, A0_s0..s2, A5_s0, ABL_scratch_s0   REUSE: the matrix runs (T-SIG based), copied from the previous output; A2 is calibrated here (A2T + escalation rows); they are never trained in this kernel
Report: main row = A2bami (3 seeds); A2 (T-SIG) = the encoder comparison; A0 / A5 / ABL_scratch stay as T-SIG-based ablations; paired A2bami vs A2."""
import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(REPO, "kaggle", "experiments_lean.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "kaggle", "experiments_bami.json"))
    a = ap.parse_args(argv)
    src = json.load(open(a.src))
    by_id = {j["id"]: j for j in src["jobs"]}
    abl = by_id["ABL_bami_s0"]

    def clone(jid, new_id, method, seed, post, reuse):
        j = json.loads(json.dumps(by_id[jid]))
        j.update({"id": new_id, "method": method, "seed": seed, "post": post, "depends_on": [], "stage1_from": None, "est_minutes": 40})
        if reuse:
            j["reuse"] = True
        return j
    jobs = [clone("ABL_bami_s0", f"A2bami_s{s}", "A2bami", s, "calibrate", False) for s in (1, 2)]
    jobs += [clone("ABL_bami_s0", "A2bami_s0", "A2bami", 0, "calibrate", True)]
    jobs += [clone(f"A2_s{s}", f"A2_s{s}", "A2", s, "calibrate", True) for s in (0, 1, 2)]
    jobs += [clone(f"A0_s{s}", f"A0_s{s}", "A0", s, "calibrate", True) for s in (0, 1, 2)]
    jobs += [clone("A5_s0", "A5_s0", "A5", 0, "predict", True), clone("ABL_scratch_s0", "ABL_scratch_s0", "ABL_scratch", 0, "predict", True)]
    assert all(j["base_config"] == abl["base_config"] and j["overrides"] == abl["overrides"] for j in jobs if j["method"] == "A2bami")
    m = src["methods"]
    methods = {"A2bami": {"family": "A2bami", "variant": "uncal", "description": "ViSO with the frozen BamiBERT text encoder (standardised cache) = the MAIN model"},
               "A2bamiT": {"family": "A2bami", "variant": "pertype", "description": "A2bami + per-type temperature (+ escalation head)"},
               "A2": {**m["A2"], "description": "ViSO with the frozen SigLIP 2 text tower (T-SIG): the encoder comparison"},
               "A2T": {**m["A2T"], "description": "A2 (T-SIG) + per-type temperature"},
               "A0": {**m["A0"], "description": "FlatS1 classifier, T-SIG-based ablation (restricted at evaluation)"},
               "A0T": {**m["A0T"], "description": "A0 (T-SIG-based) + per-type temperature"},
               "A5": {**m["A5"], "description": "ViSO, ZERO_VISION (question-only), T-SIG-based ablation"},
               "ABL_scratch": {**m["ABL_scratch"], "description": "A2 recipe with the from-scratch text encoder (T-SIG-based ablation row; 1 seed)"}}
    comps = [["A2bami", "A2"], ["A2bamiT", "A2bami"], ["A2bami", "A0"], ["A2bami", "A5"], ["A2bami", "ABL_scratch"], ["A2T", "A2"], ["A0T", "A0"], ["A5", "A2"], ["ABL_scratch", "A2"]]
    out = {**{k: v for k, v in src.items() if k not in ("jobs", "methods", "comparisons", "note")}, "version": 4, "main_encoder": "bami",
           "note": "BamiBERT = MAIN encoder (A2bami, 3 seeds: seed 0 = the matrix run ABL_bami_s0 under its family name). T-SIG-based rows (A2, A0, A5, ABL_scratch) are reused matrix runs, kept as the encoder comparison / ablations.",
           "jobs": jobs, "methods": methods, "comparisons": comps}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {[(j['id'], 'reuse' if j.get('reuse') else 'train') for j in jobs]}")
    return out


if __name__ == "__main__":
    main()
