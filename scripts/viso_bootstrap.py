"""Paired bootstrap over test items + mean +- std over seeds, from saved predictions.
    python scripts/viso_bootstrap.py --manifest kaggle/experiments.json --runs-dir <out>/runs --s1-dir ../vivqa_s1_v2/data --out bootstrap.json
Per method and seed the point metrics come from s1bench.metrics (evaluate = item_scores + aggregates); every comparison
pair of the manifest gets, per seed, a paired bootstrap (1,000 resamples of the test items, seed 0, the SAME resampled
indices for all methods) of the difference in primary accuracy, NLL, Brier, ECE and AURC with a 95% percentile CI, and the
same for the S1-Human metrics (soft accuracy, soft Brier, KL) on the human10 items only. Seed aggregation: mean +- std
(sample std, null for one seed) of every metric and of every paired difference. Deterministic.
Views (v1.1 release): --view {all,core,image_unseen} (default all) with --flags PATH (default <s1-dir>/eval_flags.jsonl) scores only the
gold items of that test view; the views are defined by the s1bench/flags.py of the benchmark given by --s1-dir (found at
<s1-dir>/../s1bench/flags.py, so the benchmark's own view flags are used). Items are scored on sorted(view gold ids
intersect the ids every method predicted). The default (all, no flags) output is byte-identical to the pre-view version."""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from utils.viso_s1 import read_jsonl  # noqa: E402,F401  (puts s1bench on sys.path)
from s1bench import metrics as M  # noqa: E402

PRIMARY = ("accuracy", "nll", "brier", "ece", "aurc")
HUMAN = ("soft_accuracy", "brier_soft", "kl")


def load_flags_module(s1_data):
    """Import s1bench/flags.py of the benchmark that owns `s1_data` (<root>/data) without touching sys.path / the v1 s1bench."""
    path = os.path.join(os.path.abspath(os.path.join(s1_data, "..")), "s1bench", "flags.py")
    if not os.path.isfile(path):
        raise ImportError(f"{path} not found: the benchmark given by --s1-dir has no s1bench/flags.py (the v1.1 release is required "
                          "for --view core / image_unseen)")
    spec = importlib.util.spec_from_file_location("s1bench_flags_of_s1_dir", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_view(gold_all, s1_data, view="all", flags_path=None):
    """-> (gold of the view as {id: item}, info or None). info is None for the default (all, no flags) so that output is unchanged."""
    if view == "all" and not flags_path:
        return gold_all, None
    fp = flags_path or os.path.join(s1_data, "eval_flags.jsonl")
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"--view {view} needs a flags file but {fp} does not exist "
                                "(pass --flags PATH, or put eval_flags.jsonl next to the split files)")
    F = load_flags_module(s1_data)
    views, exc = F.split_views(list(gold_all.values()), F.load_flags(fp))
    chosen = {"all": "all", "core": "core", "image_unseen": "image_unseen"}[view]
    gold = {it["id"]: it for it in views[chosen]}
    with open(fp, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    info = {"view": view, "flags_file": os.path.basename(fp), "flags_sha256": sha, "n_gold_split": len(gold_all),
            "n_gold_view": len(gold), "excluded_by_flag": exc}
    return gold, info


def scores_for(items, preds):
    """Item-level s1bench scores, in the order of `items`."""
    return [M.item_scores(it, preds.get(it["id"])) for it in items]


def point_metrics(scores):
    out = M.aggregate(scores)
    out.update({k: v for k, v in M.aggregate_human(scores).items() if k != "n"})
    out["n_human"] = M.aggregate_human(scores)["n"]
    return out


def resample_indices(n, resamples, seed):
    return np.random.RandomState(seed).randint(0, n, size=(resamples, n))


def boot_primary(scores, idx):
    """Bootstrap distribution of the primary metrics; ECE / AURC use the s1bench aggregate functions."""
    corr = np.array([s["correct"] for s in scores])
    nll = np.array([s["nll"] for s in scores])
    brier = np.array([s["brier"] for s in scores])
    out = {"accuracy": corr[idx].mean(1), "nll": nll[idx].mean(1), "brier": brier[idx].mean(1)}
    ece, aurc = np.empty(len(idx)), np.empty(len(idx))
    for r, row in enumerate(idx):
        sub = [scores[i] for i in row]
        ece[r], aurc[r] = M.ece(sub), M.aurc(sub)
    out["ece"], out["aurc"] = ece, aurc
    return out


def boot_human(scores, idx):
    h = [s for s in scores if s["human"]]
    if not h:
        return {k: np.full(len(idx), np.nan) for k in HUMAN}
    return {"soft_accuracy": np.array([s["soft_acc"] for s in h])[idx].mean(1),
            "brier_soft": np.array([s["brier_soft"] for s in h])[idx].mean(1),
            "kl": np.array([s["kl"] for s in h])[idx].mean(1)}


def paired(a_boot, b_boot, a_point, b_point, keys):
    """a - b: point difference, 95% percentile CI of the bootstrap differences, share of resamples with a > b."""
    out = {}
    for k in keys:
        d = a_boot[k] - b_boot[k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[k] = {"diff": float(a_point[k] - b_point[k]), "ci95": [float(lo), float(hi)], "boot_mean_diff": float(d.mean()),
                  "share_a_gt_b": float((d > 0).mean())}
    return out


def mean_std(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(statistics.mean(vals)), "std": float(statistics.stdev(vals)) if len(vals) > 1 else None, "n": len(vals)}


def preds_path(runs_dir, family, seed, variant, split):
    d = os.path.join(runs_dir, f"{family}_s{seed}")
    return os.path.join(d, f"preds_{split}.jsonl") if variant == "uncal" else os.path.join(d, "calibration", f"preds_{split}_{variant}.jsonl")


def load_method(manifest, runs_dir, gold, method, seed, split):
    meth = manifest["methods"][method]
    p = preds_path(runs_dir, meth["family"], seed, meth["variant"], split)
    if not os.path.isfile(p):
        return None
    preds = {r["id"]: r for r in read_jsonl(p)}
    return preds


def run(manifest, runs_dir, s1_data, split="test", resamples=1000, seed=0, view="all", flags_path=None):
    gold_full = {it["id"]: it for it in read_jsonl(os.path.join(s1_data, f"{split}.jsonl"))}
    gold_all, view_info = select_view(gold_full, s1_data, view, flags_path)
    seeds = sorted({j["seed"] for j in manifest["jobs"]})
    loaded = {}                                   # (method, seed) -> preds
    for m in manifest["methods"]:
        for s in seeds:
            p = load_method(manifest, runs_dir, gold_all, m, s, split)
            if p is not None:
                loaded[(m, s)] = p
    if not loaded:
        return {"error": "no predictions found", "split": split}
    common = set.intersection(*[set(p) for p in loaded.values()])
    if common - set(gold_full):
        raise KeyError(f"prediction ids missing from the gold split: {sorted(common - set(gold_full))[:5]} ...")
    ids = sorted(common & set(gold_all))                                      # all methods are scored on the same items (of the view)
    items = [gold_all[i] for i in ids]
    idx = resample_indices(len(items), resamples, seed)
    h_pos = [k for k, it in enumerate(items) if it["human_subset"]]
    idx_h = resample_indices(len(h_pos), resamples, seed + 1) if h_pos else np.zeros((resamples, 0), int)
    res = {"split": split, "n_items": len(items), "n_human_items": len(h_pos), "resamples": resamples, "seed": seed,
           "methods": {}, "comparisons": {}}
    if view_info:
        res.update(view_info)
    cache = {}
    for (m, s), preds in loaded.items():
        sc = scores_for(items, preds)
        cache[(m, s)] = {"scores": sc, "point": point_metrics(sc), "boot": boot_primary(sc, idx),
                         "boot_h": boot_human([sc[k] for k in h_pos], idx_h) if h_pos else boot_human([], idx_h)}
        res["methods"].setdefault(m, {"seeds": {}})["seeds"][str(s)] = {
            "point": cache[(m, s)]["point"], "evaluate": M.evaluate(items, {i: preds[i] for i in ids})}
    for m, d in res["methods"].items():
        pts = [v["point"] for v in d["seeds"].values()]
        d["mean_std"] = {k: mean_std([p.get(k) for p in pts]) for k in PRIMARY + HUMAN}
    for a, b in manifest["comparisons"]:
        key, per_seed = f"{a}_vs_{b}", {}
        for s in seeds:
            if (a, s) in cache and (b, s) in cache:
                A, B = cache[(a, s)], cache[(b, s)]
                per_seed[str(s)] = {"primary": paired(A["boot"], B["boot"], A["point"], B["point"], PRIMARY),
                                    "s1_human": paired(A["boot_h"], B["boot_h"], A["point"], B["point"], HUMAN)}
        if per_seed:
            agg = {k: mean_std([v["primary"][k]["diff"] for v in per_seed.values()]) for k in PRIMARY}
            agg.update({k: mean_std([v["s1_human"][k]["diff"] for v in per_seed.values()]) for k in HUMAN})
            res["comparisons"][key] = {"per_seed": per_seed, "diff_mean_std_over_seeds": agg}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--s1-dir", default=os.path.join(REPO, "..", "vivqa_s1_v2", "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--view", choices=("all", "core", "image_unseen"), default="all",
                    help="test view from eval_flags.jsonl (default all = every item, no flags needed)")
    ap.add_argument("--flags", help="eval_flags.jsonl (default <s1-dir>/eval_flags.jsonl)")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        res = run(json.load(open(a.manifest)), a.runs_dir, a.s1_dir, a.split, a.resamples, a.seed, a.view, a.flags)
    except (FileNotFoundError, ImportError) as e:
        ap.error(str(e))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("methods", "comparisons")}))
    return res


if __name__ == "__main__":
    main()
