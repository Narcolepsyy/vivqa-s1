"""Write kaggle/experiments_lean.json: the LEAN ViSO matrix (plan addendum 10) from the full manifest's structure.
    python scripts/make_lean_manifest.py --main bami --common TRAINING.LR_ENCODER=5e-5 TRAINING.LR_HEADS=5e-4 TRAINING.WARMUP_STEPS=200 MODEL.STRICT_TEXT_MASK=true
Jobs (priority order, so a partial run is still useful): A2_s0, A0_s0, A3_s0, A5_s0, ABL_<encoder>_s0 (A2 with each of the OTHER two text encoders, 1 seed), then seeds 1-2 of A2 / A0 / A3.
Methods: A0, A0T (= A0 + per-type temperature), A2, A3, A4 (= A3 + per-type temperature + escalation head), A5 (no vision), ABL_<encoder> (text-encoder ablations). No A1 / A6 / A3c; add A3c with
`--with-a3c` only after A3 beat A2 with a paired CI excluding 0. `--common` overrides apply to EVERY job (the winning config of the collapse probes); values are parsed as JSON when possible.
`--main` is the text encoder of the main matrix: scratch (from-scratch word embeddings + LSTM) or bami (frozen BamiBERT)."""
import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASES = {"scratch": ("configs/viso_siglip2.yaml", "configs/viso_flat_siglip2.yaml"), "bami": ("configs/viso_siglip2_bami.yaml", "configs/viso_flat_siglip2_bami.yaml"),
         "tsig": ("configs/viso_siglip2_tsig.yaml", "configs/viso_flat_siglip2_tsig.yaml")}
NAMES = {"scratch": "from-scratch text encoder", "bami": "frozen BamiBERT (standardised)", "tsig": "frozen SigLIP 2 text tower (T-SIG)"}


def parse_common(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", choices=sorted(BASES), required=True, help="text encoder of the main matrix; the other two become one-seed ablation rows (ABL_<encoder>_s0)")
    ap.add_argument("--common", nargs="*", default=[])
    ap.add_argument("--src", default=os.path.join(REPO, "kaggle", "experiments_siglip2.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "kaggle", "experiments_lean.json"))
    ap.add_argument("--with-a3c", action="store_true")
    a = ap.parse_args(argv)
    src = json.load(open(a.src))
    common = parse_common(a.common)
    others = [k for k in ("bami", "scratch", "tsig") if k != a.main]
    est = {j["method"]: j["est_minutes"] for j in src["jobs"]}

    def job(method, seed, base, over, post, dep=None, jid=None):
        return {"id": jid or f"{method}_s{seed}", "method": method, "base_config": base, "seed": seed, "overrides": {**common, **over}, "depends_on": [dep] if dep else [],
                "stage1_from": dep, "post": post, "est_minutes": est.get(method.rstrip("L"), est["A2"])}

    v, f = BASES[a.main]
    a3 = {"LOSS.NAME": "rlcd", "LOSS.STAGE": "rl", "LOSS.PG_WEIGHT": 1.0}
    jobs = [job("A2", 0, v, {"LOSS.NAME": "soft_ce"}, "calibrate"),                # A2 is calibrated too: A2T = A2 + per-type temperature, escalation head fitted on A2 (A2+esc)
            job("A0", 0, f, {"LOSS.NAME": "soft_ce"}, "calibrate"),
            job("A3", 0, v, a3, "calibrate", "A2_s0"),
            job("A5", 0, v, {"LOSS.NAME": "soft_ce", "MODEL.ZERO_VISION": True}, "predict"),
            *[job(f"ABL_{o}", 0, BASES[o][0], {"LOSS.NAME": "soft_ce"}, "predict", jid=f"ABL_{o}_s0") for o in others]]       # method = family = ABL_<encoder> = the run dir prefix
    for j in jobs:
        if j["id"].startswith("ABL_"):
            j["est_minutes"] = est["A2"]
    for s in (1, 2):
        jobs += [job("A2", s, v, {"LOSS.NAME": "soft_ce"}, "calibrate"), job("A0", s, f, {"LOSS.NAME": "soft_ce"}, "calibrate"), job("A3", s, v, a3, "calibrate", f"A2_s{s}")]
    if a.with_a3c:
        jobs += [job("A3c", s, v, {**a3, "LOSS.PG_WEIGHT": 0.0}, "predict", f"A2_s{s}") for s in (0, 1, 2)]
    methods = {k: src["methods"][k] for k in ("A0", "A0T", "A2", "A3", "A4", "A5") + (("A3c",) if a.with_a3c else ())}
    methods = {**{k: methods[k] for k in ("A0", "A0T", "A2")}, "A2T": {"family": "A2", "variant": "pertype", "description": "A2 + per-type temperature (+ escalation head fitted on A2: A2+esc)"},
               **{k: v_ for k, v_ in methods.items() if k not in ("A0", "A0T", "A2")}}
    for o in others:
        methods[f"ABL_{o}"] = {"family": f"ABL_{o}", "variant": "uncal", "description": f"A2 with the {NAMES[o]} (ablation of the main {NAMES[a.main]}; 1 seed)"}
    comps = [["A3", "A2"], ["A2", "A0"], ["A2T", "A2"], ["A4", "A3"], ["A0T", "A0"], ["A5", "A2"]] + [[f"ABL_{o}", "A2"] for o in others] + ([["A3", "A3c"]] if a.with_a3c else [])
    out = {**{k: v_ for k, v_ in src.items() if k not in ("jobs", "methods", "comparisons", "note")}, "version": 2,
           "note": f"LEAN ViSO matrix (plan addendum 10): main text encoder = {a.main}; common overrides (winning collapse-fix config) = {common}. No A1 / A6" + ("" if a.with_a3c else " / A3c")
                   + ". Regenerate with scripts/make_lean_manifest.py (test.jsonl is pinned by sha256: regenerate when the v2 test split changes).",
           "common_overrides": common, "jobs": jobs, "methods": methods, "comparisons": comps}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(jobs)} jobs, main={a.main}, ablations={others}")
    return out


if __name__ == "__main__":
    main()
