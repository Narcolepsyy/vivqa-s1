"""System One -> System Two CASCADE analysis from saved predictions (CPU, seconds; no model runs).
System One = ViSO (A2 seeds, per-type calibrated predictions); it answers the items it is most CONFIDENT about (calibrated max_prob) and escalates the rest to System Two:
  (a) a VLM (default: the Qwen3.5-0.8B LoRA-img run: <lora run>/calibration/preds_test_pertype.jsonl),
  (b) the decision-API teacher (a REFERENCE, never a baseline; ~/data/decisions_teacher/test.jsonl converted as in vivqa_s1_verify/v2/teacher_audit.py; a refusal has no distribution: that item keeps System One's answer, counted).
For coverage x = 0 .. 100 % (System One answers its x % most confident test items, ties by id; the rest go to System Two) it reports test accuracy / NLL / ECE (All, Core, Image-unseen),
the System-Two call rate (1 - x), the largest coverage at which the cascade still reaches 95 % / 99 % of System Two's accuracy, an ORACLE escalation (System One answers exactly the items it gets right
and System Two the rest: the upper bound of any escalation signal) and a relative-compute column. Mean ± sd over the System-One seeds.
Per-type ROUTING (--no-routing to skip): per-type confidence thresholds chosen on DEV (dynamic programme: for every overall System-One coverage the split of the budget between noul / score / choice that
maximises the dev cascade accuracy), applied unchanged to TEST, plus the simple policy "always escalate yes/no, global threshold on the rest". Both are reported next to the global-threshold curve at the
SAME actual test coverage, with the 95 % / 99 % match points and the compute column.
  python scripts/viso_cascade.py --viso-runs <runs/A2_s0> <runs/A2_s1> <runs/A2_s2> --vlm-run <ezy_results/lora-img/lora_s0> --teacher ~/data/decisions_teacher/test.jsonl --out-dir <dir>"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S1ROOT = os.path.abspath(os.path.join(REPO, "..", "vivqa_s1_v2"))
sys.path.insert(0, S1ROOT)
sys.path.insert(0, REPO)
from s1bench.flags import load_flags  # noqa: E402
from s1bench.metrics import aggregate, item_scores  # noqa: E402
from s1bench.schema import label_space  # noqa: E402

VIEWS = ("all", "core", "image_unseen")
SPLIT_89 = (66.0, 40.9)


def rd(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def teacher_preds(items, path):
    """-> ({id: pred}, n_refusals); same conversion as the teacher audit (predicate -> P(true), score labels -> levels with the 8-9 mass split by the train prior)"""
    ans = {r["id"]: r["answer"] for r in rd(path)}
    preds, ref = {}, 0
    for it in items:
        a = ans.get(it["id"])
        if a is None or a["type"] == "refusal":
            ref += 1
            continue
        sp = label_space(it["questions"]["q"])
        if a["type"] == "predicate":
            p = min(1.0, max(0.0, float(a["probability"])))
            v = [1 - p, p]
        elif a["type"] == "choice":
            vals = {x["value"]: float(x["probability"]) for x in a["probabilities"]}
            v = [vals.get(k, 0.0) for k in sp]
            s = sum(v)
            v = [x / s for x in v] if s > 0 else [1 / len(sp)] * len(sp)
        else:
            lv = [0.0] * 11
            for x in a["probabilities"]:
                lab, p = x["label"], float(x["probability"])
                if lab == "8-9":
                    lv[8] += p * SPLIT_89[0] / sum(SPLIT_89)
                    lv[9] += p * SPLIT_89[1] / sum(SPLIT_89)
                elif lab == "10+":
                    lv[10] += p
                else:
                    lv[int(lab)] += p
            s = sum(lv)
            v = [x / s for x in lv] if s > 0 else [1 / 11] * 11
        preds[it["id"]] = {"probabilities": dict(zip(sp, v))}
    return preds, ref


def view_masks(items, flags_path):
    spec = importlib.util.spec_from_file_location("s1_evaluate", os.path.join(S1ROOT, "evaluate.py"))
    EV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(EV)
    views, _ = EV.split_views(items, load_flags(flags_path))
    ids = {n: {it["id"] for it in views[n]} for n in ("core", "image_unseen")}
    return {"all": np.ones(len(items), bool), "core": np.array([it["id"] in ids["core"] for it in items]), "image_unseen": np.array([it["id"] in ids["image_unseen"] for it in items])}


def curve(s1, s2, order, masks, grid):
    """metrics of the cascade at each coverage in grid (fractions): items order[:k] keep System One's scores, the rest take System Two's"""
    n = len(s1)
    out = {v: {k: [] for k in ("accuracy", "nll", "ece")} for v in VIEWS}
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    for x in grid:
        k = int(round(x * n))
        use1 = rank < k
        mixed = [s1[i] if use1[i] else s2[i] for i in range(n)]
        for v in VIEWS:
            agg = aggregate([mixed[i] for i in np.flatnonzero(masks[v])])
            for m in out[v]:
                out[v][m].append(agg[m])
    return {v: {m: np.array(a) for m, a in d.items()} for v, d in out.items()}


def match_points(grid, acc, acc_s2, fracs=(0.95, 0.99)):
    """largest System-One coverage whose accuracy is >= frac * System Two's accuracy"""
    res = {}
    for f in fracs:
        ok = np.flatnonzero(acc >= f * acc_s2)
        res[str(f)] = float(grid[ok.max()]) if len(ok) else None
    return res


TYPE_NAMES = ("noul", "score", "choice")


def concave_envelope(cum_gains):
    """Least concave majorant of cumulative gain curve cum_gains[0..n] via Pool Adjacent Violators Algorithm (PAVA)
    on the marginal increments diff(cum_gains) (monotone non-increasing increments). Eliminates random-walk
    spikes on dev and guarantees strictly monotonic per-type budget allocations k_t(K)."""
    inc = np.diff(np.asarray(cum_gains, dtype=np.float64))
    if len(inc) == 0:
        return np.array([0.0], dtype=np.float64)
    sums = inc.copy().tolist()
    counts = [1] * len(inc)
    i = 0
    while i < len(sums) - 1:
        if sums[i] / counts[i] < sums[i + 1] / counts[i + 1]:
            sums[i] += sums.pop(i + 1)
            counts[i] += counts.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    smooth_inc = np.repeat([s / c for s, c in zip(sums, counts)], counts)
    return np.concatenate([[0.0], np.cumsum(smooth_inc)])


def dp_budget(gains, n_total):
    """gains[t] = cumulative gain of answering the k most confident items of type t with System One (array of len n_t + 1). Returns, for every total budget K = 0..n_total, the best split (k_t) maximising the
    summed gain (exact dynamic programme over the three types)."""
    f = [gains[0]]
    choice = []
    for t in (1, 2):
        g, prev = gains[t], f[-1]
        m_max = min(n_total, len(prev) - 1 + len(g) - 1)
        new = np.full(m_max + 1, -np.inf)
        arg = np.zeros(m_max + 1, int)
        for m in range(m_max + 1):
            ks = np.arange(0, min(m, len(g) - 1) + 1)
            prev_idx = m - ks
            ok = prev_idx <= len(prev) - 1
            ks, prev_idx = ks[ok], prev_idx[ok]
            if not len(ks):
                continue
            v = prev[prev_idx] + g[ks]
            j = int(np.argmax(v))
            new[m], arg[m] = v[j], ks[j]
        f.append(new)
        choice.append(arg)
    out = {}
    for K in range(0, len(f[2])):
        k2 = int(choice[1][K])
        rest = K - k2
        k1 = int(choice[0][rest]) if rest < len(choice[0]) else 0
        out[K] = (rest - k1, k1, k2)                                   # (noul, score, choice)
    return out


def thresholds_from(ks, conf_by_type_sorted):
    """k_t -> confidence threshold per type (items with conf >= tau are answered by System One): +inf for k = 0, -inf for k = n_t"""
    tau = []
    for t, k in enumerate(ks):
        c = conf_by_type_sorted[t]
        tau.append(np.inf if k <= 0 else (-np.inf if k >= len(c) else float(c[k - 1])))
    return tau


def routing_curves(s1_dev, s2_dev, type_dev, s1_test, s2_test, type_test, masks, grid_pct):
    """-> {policy: list of {target, test_coverage, per_type_dev_coverage, metrics per view}}"""
    n_dev = len(s1_dev)
    conf_dev = np.array([s["conf"] for s in s1_dev])
    c1 = np.array([s["correct"] for s in s1_dev])
    c2 = np.array([s["correct"] for s in s2_dev])
    ids_dev = np.arange(n_dev)
    gains, sorted_conf, n_t = [], [], []
    for t in range(3):
        idx = np.flatnonzero(type_dev == t)
        o = idx[np.lexsort((ids_dev[idx], -conf_dev[idx]))]
        raw_g = np.concatenate([[0.0], np.cumsum(c1[o] - c2[o])])
        gains.append(concave_envelope(raw_g))
        sorted_conf.append(conf_dev[o])
        n_t.append(len(idx))
    plan = dp_budget(gains, n_dev)
    conf_test = np.array([s["conf"] for s in s1_test])
    out = {"dev_dp": [], "yes_no_to_system_two": []}
    for pct in grid_pct:
        K = int(round(pct / 100 * n_dev))
        # policy 1: dev-chosen per-type thresholds
        ks = plan[min(K, max(plan))]
        tau = thresholds_from((ks[0], ks[1], ks[2]), sorted_conf)
        use = np.array([conf_test[i] >= tau[type_test[i]] for i in range(len(s1_test))])
        out["dev_dp"].append(eval_use(use, s1_test, s2_test, masks, pct, ks, n_t))
        # policy 2: yes/no always escalated; one global threshold on score + choice chosen on dev for the same budget
        rest = np.flatnonzero(type_dev != 0)
        o = rest[np.lexsort((ids_dev[rest], -conf_dev[rest]))]
        k = min(K, len(o))
        tau2 = np.inf if k <= 0 else (-np.inf if k >= len(o) else float(conf_dev[o][k - 1]))
        use2 = np.array([(type_test[i] != 0) and conf_test[i] >= tau2 for i in range(len(s1_test))])
        out["yes_no_to_system_two"].append(eval_use(use2, s1_test, s2_test, masks, pct, (0, None, None), n_t))
    return out


def eval_use(use, s1, s2, masks, pct, ks, n_t):
    mixed = [s1[i] if use[i] else s2[i] for i in range(len(s1))]
    row = {"target_dev_coverage": pct, "test_coverage": float(use.mean()), "per_type_dev_coverage": [None if k is None else round(k / max(n, 1), 3) for k, n in zip(ks, n_t)]}
    for v in VIEWS:
        agg = aggregate([mixed[i] for i in np.flatnonzero(masks[v])])
        row[v] = {m: agg[m] for m in ("accuracy", "nll", "ece")}
    return row


def s2_dev_missing(routing, name, i):
    return bool(routing["s2_dev"][name][i]["missing"])


def ms(a, nd=3):
    return f"{np.mean(a):.{nd}f}" + (f" ± {np.std(a, ddof=1):.{nd}f}" if len(a) > 1 else "")


def vlm_flops(tokens_text=70):
    """rough FLOPs of ONE Qwen3.5-0.8B forward unit (2 x matmul params x tokens; attention terms ignored): language model non-embedding params x (image + text tokens) + vision tower params x patches"""
    try:
        from huggingface_hub import get_safetensors_metadata
        md = get_safetensors_metadata("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17")
        lm = vis = 0
        for f in md.files_metadata.values():
            for name, t in f.tensors.items():
                n = int(np.prod(t.shape))
                if "embed_tokens" in name or "lm_head" in name:
                    continue
                if ".visual." in name:
                    vis += n
                elif "language_model" in name:
                    lm += n
        img_tokens, patches = 196, 784                               # 448 x 448 px cap, patch 16, 2x2 merge
        return {"language_model_params_without_embeddings": lm, "vision_tower_params": vis, "assumed_image_tokens": img_tokens, "assumed_patches": patches, "assumed_text_tokens": tokens_text,
                "flops": 2.0 * lm * (img_tokens + tokens_text) + 2.0 * vis * patches}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--viso-runs", nargs="+", required=True, help="run dirs of System One (each with calibration/preds_test_pertype.jsonl)")
    ap.add_argument("--vlm-run", default=None, help="run dir of the VLM System Two (calibration/preds_test_pertype.jsonl)")
    ap.add_argument("--vlm-name", default="Qwen3.5-0.8B LoRA-img")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--s1-data", default=os.path.join(S1ROOT, "data"))
    ap.add_argument("--efficiency-text", default=os.path.join(REPO, "kaggle", "text_tower_efficiency.json"))
    ap.add_argument("--viso-flops", type=float, default=2.3924e9, help="ViSO (T-SIG) batch-1 FLOPs per question (results.md efficiency row)")
    ap.add_argument("--vision-flops", type=float, default=44405096448, help="SigLIP 2 vision tower FLOPs per image")
    ap.add_argument("--no-routing", action="store_true")
    ap.add_argument("--teacher-dev", default=None, help="decision-API teacher jsonl for the DEV split (default: dev.jsonl next to --teacher)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    items = rd(os.path.join(a.s1_data, "test.jsonl"))
    masks = view_masks(items, os.path.join(a.s1_data, "eval_flags.jsonl"))
    grid = np.linspace(0.0, 1.0, 101)
    s1_sets = []
    for d in a.viso_runs:
        pr = {r["id"]: r for r in rd(os.path.join(d, "calibration", "preds_test_pertype.jsonl"))}
        s1_sets.append([item_scores(it, pr.get(it["id"])) for it in items])
    systems = {}
    if a.vlm_run:
        pr = {r["id"]: r for r in rd(os.path.join(a.vlm_run, "calibration", "preds_test_pertype.jsonl"))}
        systems[a.vlm_name] = {"scores": [item_scores(it, pr.get(it["id"])) for it in items], "refusals": 0}
    if a.teacher:
        pt, ref = teacher_preds(items, a.teacher)
        sc = [item_scores(it, pt.get(it["id"])) for it in items]
        systems["decision-API teacher (reference)"] = {"scores": sc, "refusals": ref, "refused": np.array([it["id"] not in pt for it in items])}
    routing = None
    if not a.no_routing:
        dev_items = rd(os.path.join(a.s1_data, "dev.jsonl"))
        type_dev = np.array([TYPE_NAMES.index(it["questions"]["q"]["type"]) for it in dev_items])
        type_test = np.array([TYPE_NAMES.index(it["questions"]["q"]["type"]) for it in items])
        s1_dev_sets = []
        for d in a.viso_runs:
            pr = {r["id"]: r for r in rd(os.path.join(d, "calibration", "preds_dev_pertype.jsonl"))}
            s1_dev_sets.append([item_scores(it, pr.get(it["id"])) for it in dev_items])
        s2_dev = {}
        if a.vlm_run:
            pr = {r["id"]: r for r in rd(os.path.join(a.vlm_run, "calibration", "preds_dev_pertype.jsonl"))}
            s2_dev[a.vlm_name] = [item_scores(it, pr.get(it["id"])) for it in dev_items]
        if a.teacher:
            tdev = a.teacher_dev or os.path.join(os.path.dirname(a.teacher), "dev.jsonl")
            ptd, _ = teacher_preds(dev_items, tdev)
            s2_dev["decision-API teacher (reference)"] = [item_scores(it, ptd.get(it["id"])) for it in dev_items]
        routing = {"type_dev": type_dev, "type_test": type_test, "s1_dev": s1_dev_sets, "s2_dev": s2_dev, "n_dev": len(dev_items)}
    # cost: relative FLOPs per question (first question on an image: vision tower + text tower + ViSO); the VLM forward is one unit per question
    text_f = 0.0
    if os.path.isfile(a.efficiency_text):
        text_f = float(json.load(open(a.efficiency_text))["siglip2_text_tower"]["flops_batch1"])
    cost_viso = a.vision_flops + text_f + a.viso_flops
    vf = vlm_flops()
    res = {"n_test_items": len(items), "views": {v: int(masks[v].sum()), } if False else {v: int(masks[v].sum()) for v in VIEWS}, "viso_runs": a.viso_runs, "grid": [float(x) for x in grid],
           "cost_model": {"viso_flops_first_question_on_image": cost_viso, "viso_flops_components": {"vision_tower": a.vision_flops, "text_tower_T-SIG": text_f, "viso": a.viso_flops}, "vlm_forward": vf,
                          "note": "relative cost at coverage x = (cost_viso + (1 - x) * cost_vlm) / cost_vlm for the VLM; the API teacher has no local cost (external model)"},
           "viso_alone": {}, "systems": {}}
    # System One alone
    for v in VIEWS:
        aggs = [aggregate([s1[i] for i in np.flatnonzero(masks[v])]) for s1 in s1_sets]
        res["viso_alone"][v] = {m: [g[m] for g in aggs] for m in ("accuracy", "nll", "ece")}
    for name, sysd in systems.items():
        s2 = sysd["scores"]
        entry = {"refusals": sysd["refusals"], "per_seed": []}
        s2_alone = {v: aggregate([s2[i] for i in np.flatnonzero(masks[v])]) for v in VIEWS}
        entry["system_two_alone"] = {v: {m: s2_alone[v][m] for m in ("accuracy", "nll", "ece")} for v in VIEWS}
        for si, s1 in enumerate(s1_sets):
            # items the teacher refused keep System One's answer (also at coverage 0)
            s2_eff = [s1[i] if (sysd.get("refused") is not None and sysd["refused"][i]) else s2[i] for i in range(len(items))]
            conf = np.array([s["conf"] for s in s1])
            ids = np.array([it["id"] for it in items])
            order = np.lexsort((ids, -conf))                           # most confident first, ties by id
            corr = np.array([s["correct"] for s in s1])
            oracle_order = np.lexsort((ids, -conf, -corr))             # correct items first, then by confidence
            cv, ov = curve(s1, s2_eff, order, masks, grid), curve(s1, s2_eff, oracle_order, masks, grid)
            acc_s2 = {v: float(cv[v]["accuracy"][0]) for v in VIEWS}
            kc = int(round(corr.sum()))                                # oracle: System One answers exactly the items it gets right
            rt = None
            if routing is not None:
                s2_dev_eff = [routing["s1_dev"][si][i] if (name.startswith("decision") and s2_dev_missing(routing, name, i)) else routing["s2_dev"][name][i] for i in range(routing["n_dev"])]
                rt = routing_curves(routing["s1_dev"][si], s2_dev_eff, routing["type_dev"], s1, s2_eff, routing["type_test"], masks, list(range(0, 101, 5)))
            entry["per_seed"].append({"routing": rt, "curve": {v: {m: cv[v][m].tolist() for m in cv[v]} for v in VIEWS}, "oracle_curve": {v: {m: ov[v][m].tolist() for m in ov[v]} for v in VIEWS},
                                      "oracle_point": {"viso_coverage": float(corr.sum() / len(items)), **{v: {m: float(ov[v][m][int(round(corr.sum() / len(items) * 100))]) for m in ("accuracy", "nll", "ece")} for v in VIEWS}},
                                      "match": {v: match_points(grid, cv[v]["accuracy"], acc_s2[v]) for v in VIEWS}, "system_two_accuracy_with_fallback": acc_s2})
        res["systems"][name] = entry
    json.dump(res, open(os.path.join(a.out_dir, "cascade.json"), "w"), indent=1)
    write_md(res, a.out_dir)
    try:
        plot(res, a.out_dir)
    except Exception as e:
        print(f"(plot failed: {type(e).__name__}: {e})")
    return res


def write_md(res, out_dir):
    L = ["# Cascade: System One (ViSO A2, per-type calibrated; escalation by calibrated max_prob) -> System Two", "",
         f"Test split, n = {res['n_test_items']} (Core {res['views']['core']}, Image-unseen {res['views']['image_unseen']}); mean ± sd over {len(res['viso_runs'])} System-One seeds. "
         "Coverage x = share of test items answered by System One (its most confident); the rest (1 - x) is the System-Two call rate. Teacher refusals keep System One's answer.", ""]
    L += ["## System One alone (coverage 100 %)", "", "| view | accuracy | NLL | ECE |", "|---|---|---|---|"]
    for v in VIEWS:
        d = res["viso_alone"][v]
        L.append(f"| {v} | {ms(d['accuracy'])} | {ms(d['nll'])} | {ms(d['ece'])} |")
    cm = res["cost_model"]
    for name, e in res["systems"].items():
        L += ["", f"## System Two = {name}" + (f" ({e['refusals']} refusals)" if e["refusals"] else ""), "", "System Two alone (coverage 0 %; refusals fall back to System One):", "", "| view | accuracy | NLL | ECE |", "|---|---|---|---|"]
        for v in VIEWS:
            d = e["system_two_alone"][v]
            L.append(f"| {v} | {d['accuracy']:.3f} | {d['nll']:.3f} | {d['ece']:.3f} |")
        L += ["", "Cascade (All items; Core and Image-unseen accuracy in the last two columns):", "", "| System-One coverage | System-Two call rate | accuracy | NLL | ECE | Core acc | Image-unseen acc | relative compute vs System Two (VLM) |", "|---|---|---|---|---|---|---|---|"]
        vf = cm["vlm_forward"].get("flops")
        for k in range(0, 101, 10):
            row = {v: {m: [sd["curve"][v][m][k] for sd in e["per_seed"]] for m in ("accuracy", "nll", "ece")} for v in VIEWS}
            rel = f"{(cm['viso_flops_first_question_on_image'] + (1 - k / 100) * vf) / vf:.2f}" if (vf and "VLM" in name or vf and "Qwen" in name) else "n/a (external model)"
            L.append(f"| {k} % | {100 - k} % | {ms(row['all']['accuracy'])} | {ms(row['all']['nll'])} | {ms(row['all']['ece'])} | {ms(row['core']['accuracy'])} | {ms(row['image_unseen']['accuracy'])} | {rel} |")
        L += ["", "Largest System-One coverage at which the cascade still reaches a fraction of System Two's own accuracy (mean ± sd; call rate = 1 - coverage):", "", "| view | ≥ 95 % of System Two | ≥ 99 % of System Two |", "|---|---|---|"]
        for v in VIEWS:
            def cell(f):
                xs = [sd["match"][v][f] for sd in e["per_seed"] if sd["match"][v][f] is not None]
                return (f"{100 * np.mean(xs):.0f} % coverage (call rate {100 - 100 * np.mean(xs):.0f} %)" + (f" ± {100 * np.std(xs, ddof=1):.0f}" if len(xs) > 1 else "")) if xs else "not reached"
            L.append(f"| {v} | {cell('0.95')} | {cell('0.99')} |")
        if e["per_seed"][0].get("routing"):
            L += routing_md(e, cm, name)
        op = [sd["oracle_point"] for sd in e["per_seed"]]
        L += ["", f"Oracle escalation (System One answers exactly the items it gets right = {100 * np.mean([o['viso_coverage'] for o in op]):.1f} % coverage; System Two the rest): "
              f"All accuracy {ms([o['all']['accuracy'] for o in op])}, NLL {ms([o['all']['nll'] for o in op])}, Core accuracy {ms([o['core']['accuracy'] for o in op])}, "
              f"Image-unseen accuracy {ms([o['image_unseen']['accuracy'] for o in op])}. Any real escalation signal lies between the confidence curve and this bound."]
    L += ["", "## Cost model", "", f"- System One per question on a new image: SigLIP 2 vision tower {cm['viso_flops_components']['vision_tower']:.3e} + T-SIG text tower {cm['viso_flops_components']['text_tower_T-SIG']:.3e} + ViSO "
          f"{cm['viso_flops_components']['viso']:.3e} = {cm['viso_flops_first_question_on_image']:.3e} FLOPs (later questions on the same image reuse the vision tower).",
          f"- System Two (VLM) unit, estimated: {cm['vlm_forward'].get('flops', float('nan')):.3e} FLOPs ({json.dumps({k: v for k, v in cm['vlm_forward'].items() if k != 'flops'})}).",
          "- Relative compute = (System One cost + call rate x System Two cost) / System Two cost. These are FLOP estimates (matmul terms only), not measured latency.",
          "- The decision-API teacher is an external model, labelled a reference: it has no local compute and is never a baseline."]
    open(os.path.join(out_dir, "cascade.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def interp_global(sd, view, metric, cov):
    g = np.linspace(0, 1, 101)
    return float(np.interp(cov, g, sd["curve"][view][metric]))


def routing_md(e, cm, name):
    """per-type routing tables: thresholds chosen on dev, applied to test; compared with the global-confidence curve at the SAME actual test coverage"""
    vf = cm["vlm_forward"].get("flops")
    L = ["", f"### Per-type routing (thresholds chosen on DEV, applied to TEST; System Two = {name})", "",
         "Policy A = dev-optimal split of the escalation budget between yes/no, score and choice (dynamic programme on the dev cascade accuracy); policy B = yes/no always goes to System Two, one dev-chosen threshold for score + choice. "
         "`global acc` = the global-threshold cascade at the same actual test coverage (interpolated on the 1 % grid); Δ = routing - global. Mean ± sd over seeds.", ""]
    for key, title in (("dev_dp", "Policy A: dev-optimal per-type thresholds"), ("yes_no_to_system_two", "Policy B: yes/no always to System Two")):
        L += [f"**{title}**", "", "| dev budget (System-One coverage) | test coverage | call rate | per-type dev coverage (noul / score / choice) | accuracy | NLL | ECE | Core acc | Image-unseen acc | global acc | Δ acc | relative compute |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in range(0, 21, 2):                                        # grid_pct = 0, 5, ..., 100: every 10 %
            rows = [sd["routing"][key][k] for sd in e["per_seed"]]
            cov = [r["test_coverage"] for r in rows]
            glob = [interp_global(sd, "all", "accuracy", r["test_coverage"]) for sd, r in zip(e["per_seed"], rows)]
            d = [r["all"]["accuracy"] - g for r, g in zip(rows, glob)]
            pt = rows[0]["per_type_dev_coverage"]
            ptxt = " / ".join("-" if x is None else f"{100 * np.mean([r['per_type_dev_coverage'][i] for r in rows]):.0f} %" for i, x in enumerate(pt))
            rel = f"{(cm['viso_flops_first_question_on_image'] + (1 - np.mean(cov)) * vf) / vf:.2f}" if vf and ("Qwen" in name) else "n/a (external model)"
            L.append(f"| {rows[0]['target_dev_coverage']} % | {100 * np.mean(cov):.1f} % | {100 - 100 * np.mean(cov):.1f} % | {ptxt} | {ms([r['all']['accuracy'] for r in rows])} | {ms([r['all']['nll'] for r in rows])} | "
                     f"{ms([r['all']['ece'] for r in rows])} | {ms([r['core']['accuracy'] for r in rows])} | {ms([r['image_unseen']['accuracy'] for r in rows])} | {ms(glob)} | {np.mean(d):+.3f} | {rel} |")
        L.append("")
        L += ["Largest test coverage reaching a fraction of System Two's accuracy (All items):", ""]
        acc_s2 = [sd["system_two_accuracy_with_fallback"]["all"] for sd in e["per_seed"]]
        cells = []
        for f in (0.95, 0.99):
            xs = []
            for sd, a2 in zip(e["per_seed"], acc_s2):
                ok = [r["test_coverage"] for r in sd["routing"][key] if r["all"]["accuracy"] >= f * a2]
                if ok:
                    xs.append(max(ok))
            cells.append(f"≥ {int(f * 100)} %: " + (f"{100 * np.mean(xs):.0f} % coverage (call rate {100 - 100 * np.mean(xs):.0f} %)" if xs else "not reached"))
        L += ["- " + " | ".join(cells), ""]
    return L


def plot(res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.array(res["grid"]) * 100
    fig, axes = plt.subplots(len(res["systems"]), 3, figsize=(15, 4.2 * len(res["systems"])), squeeze=False)
    for r, (name, e) in enumerate(res["systems"].items()):
        for c, (m, title) in enumerate((("accuracy", "accuracy"), ("nll", "NLL"), ("ece", "ECE"))):
            ax = axes[r][c]
            for v, col in (("all", "C0"), ("core", "C1"), ("image_unseen", "C2")):
                cv = np.array([sd["curve"][v][m] for sd in e["per_seed"]])
                ax.plot(grid, cv.mean(0), color=col, label=f"cascade {v}")
                ax.fill_between(grid, cv.mean(0) - cv.std(0), cv.mean(0) + cv.std(0), color=col, alpha=0.2)
            ov = np.array([sd["oracle_curve"]["all"][m] for sd in e["per_seed"]]).mean(0)
            ax.plot(grid, ov, "k--", lw=1, label="oracle escalation (all)")
            if m == "accuracy" and e["per_seed"][0].get("routing"):
                for key, col, lab in (("dev_dp", "C3", "per-type routing A (dev-chosen)"), ("yes_no_to_system_two", "C4", "routing B (yes/no -> System Two)")):
                    xs = np.mean([[r["test_coverage"] for r in sd["routing"][key]] for sd in e["per_seed"]], 0) * 100
                    ys = np.mean([[r["all"]["accuracy"] for r in sd["routing"][key]] for sd in e["per_seed"]], 0)
                    ax.plot(xs, ys, color=col, marker="o", ms=3, lw=1.2, label=lab)
            ax.axhline(e["system_two_alone"]["all"][m], color="r", ls=":", label="System Two alone (all)")
            ax.axhline(np.mean(res["viso_alone"]["all"][m]), color="g", ls=":", label="ViSO alone (all)")
            ax.set_xlabel("System-One coverage (%): ViSO answers its most confident x %")
            ax.set_ylabel(title)
            ax.set_title(f"System Two = {name}" if c == 0 else title)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cascade.png"), dpi=130)


if __name__ == "__main__":
    main()
