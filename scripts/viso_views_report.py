"""Report additions for Phase 2c (plans/viso_phase2c_siglip2_plan.md, addendum 3): the Image-unseen column, the seen-vs-unseen gap per method, accuracy and calibration by
AGREEMENT BIN (top share of the gold distribution: < 0.3 / 0.3-0.6 / >= 0.6) and the efficiency rows (the frozen SigLIP 2 tower, once per image, + ViSO). Used by `viso_kaggle_run.py --finalize`
(appended to results.md) and standalone:
    python scripts/viso_views_report.py --manifest kaggle/experiments_siglip2.json --runs-dir <out>/runs --s1-dir ../vivqa_s1_v2/data [--packed ~/data/siglip2_packed] [--out views.md]
Only generated files are quoted; the metrics are the benchmark's own (s1bench.metrics)."""
import argparse
import json
import os
import statistics
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.viso_s1 import read_jsonl  # noqa: E402
from s1bench.metrics import evaluate  # noqa: E402

BINS = (("< 0.3", 0.0, 0.3), ("0.3-0.6", 0.3, 0.6), (">= 0.6", 0.6, 1.0001))
ORDER = ("A0", "A0T", "A1", "A2", "A3", "A3c", "A4", "A5", "A6")


def top_share(item):
    return max(item["gold"]["q"]["probabilities"].values())


def ms(vals, nd=4):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return "-"
    s = f"{statistics.mean(vals):.{nd}f}"
    return s + (f" ± {statistics.stdev(vals):.{nd}f}" if len(vals) > 1 else "") + (f" (n={len(vals)})" if len(vals) > 1 else "")


def seeds_of(manifest, method):
    fam = manifest["methods"][method]["family"]
    return sorted({j["seed"] for j in manifest["jobs"] if j["method"] == fam})


def split_gold(s1_data, split="test"):
    """-> {view: {id: item}} with All, Core, Image-unseen and Core-seen (Core minus Image-unseen)"""
    import viso_bootstrap
    full = {it["id"]: it for it in read_jsonl(os.path.join(s1_data, f"{split}.jsonl"))}
    out = {"all": full}
    for v in ("core", "image_unseen"):
        out[v], _ = viso_bootstrap.select_view(full, s1_data, v, None)
    out["core_seen"] = {i: it for i, it in out["core"].items() if i not in out["image_unseen"]}
    return out


def metric(items, preds, key="accuracy"):
    if not items:
        return None
    return evaluate(list(items), preds)["primary"]["all"][key]


def bin_items(items):
    return [{i: it for i, it in items.items() if lo <= top_share(it) < hi} for _, lo, hi in BINS]


def views_markdown(manifest, runs_dir, s1_data, split="test", vision_eff=None, model_eff=None, text_eff=None):
    import viso_bootstrap
    views = split_gold(s1_data, split)
    methods = [m for m in ORDER if m in manifest["methods"]] + [m for m in manifest["methods"] if m not in ORDER]       # then every other method of the manifest (A2T, ABL_*, A3fix ...)
    preds = {}
    for m in methods:
        for s in seeds_of(manifest, m):
            p = viso_bootstrap.load_method(manifest, runs_dir, views["all"], m, s, split)
            if p is not None:
                preds[(m, s)] = p
    have = [m for m in methods if any(k[0] == m for k in preds)]
    n = {k: len(v) for k, v in views.items()}
    L = ["", "## Views: All / Core (headline) / Image-unseen (headline for trained-vs-zero-shot comparisons)", "",
         f"Items: All {n['all']}, Core {n['core']}, Image-unseen {n['image_unseen']} (Core-seen {n['core_seen']}). Cells are mean ± std over seeds.", "",
         "| method | acc All | acc Core | acc Image-unseen | NLL Core | NLL Image-unseen | ECE Core | ECE Image-unseen |", "|---|---|---|---|---|---|---|---|"]
    per = {}
    for m in have:
        ss = [k for k in preds if k[0] == m]
        cell = lambda view, key: ms([metric(views[view].values(), preds[k], key) for k in ss])
        per[m] = ss
        L.append(f"| {m} | {cell('all', 'accuracy')} | {cell('core', 'accuracy')} | {cell('image_unseen', 'accuracy')} | {cell('core', 'nll')} | {cell('image_unseen', 'nll')} | {cell('core', 'ece')} | {cell('image_unseen', 'ece')} |")
    L += ["", "## Seen-vs-unseen gap (a memorisation measurement: 35% of test items share an image with train / calib / dev)", "",
          "| method | acc Core-seen | acc Image-unseen | gap (seen − unseen) |", "|---|---|---|---|"]
    for m in have:
        seen = [metric(views["core_seen"].values(), preds[k]) for k in per[m]]
        unseen = [metric(views["image_unseen"].values(), preds[k]) for k in per[m]]
        gap = [a - b for a, b in zip(seen, unseen) if a is not None and b is not None]
        L.append(f"| {m} | {ms(seen)} | {ms(unseen)} | {ms(gap)} |")
    bins = bin_items(views["core"])
    L += ["", "## Agreement bins (Core; top share of the gold soft label: " + ", ".join(f"{b[0]} n = {len(x)}" for b, x in zip(BINS, bins)) + ")", "",
          "| method | acc " + " | acc ".join(b[0] for b in BINS) + " | ECE " + " | ECE ".join(b[0] for b in BINS) + " |", "|---|" + "---|" * (2 * len(BINS))]
    for m in have:
        acc = [ms([metric(b.values(), preds[k]) for k in per[m]]) for b in bins]
        ece = [ms([metric(b.values(), preds[k], "ece") for k in per[m]]) for b in bins]
        L.append(f"| {m} | " + " | ".join(acc) + " | " + " | ".join(ece) + " |")
    if vision_eff or model_eff:
        L += ["", "## Efficiency rows (batch 1): what runs on the device per image / per question", "",
              "The frozen SigLIP 2 VISION tower runs once per image; the frozen TEXT encoder (T-SIG: the SigLIP 2 text tower, or BamiBERT in the ablation) runs once per QUESTION; ViSO runs per question on top of both. "
              "The cached features of training are only a shortcut: at inference all three run.", "",
              "| component | params | FLOPs | trained? | note |", "|---|---|---|---|---|"]
        vp = vf = tp = tf = bp = bf = 0
        if vision_eff:
            vp, vf = vision_eff["params_total"], vision_eff["flops_batch1"]
            L.append(f"| SigLIP 2 B/16 256 px vision tower (once per image) | {vp:,} (encoder {vision_eff['params_encoder']:,} + MAP head {vision_eff['params_map_head']:,}) | {vf:.4e} | no, frozen | "
                     f"CPU batch-1 latency {vision_eff['cpu_latency_ms_median_batch1']} ms ({vision_eff['cpu_threads']} threads) |")
        if text_eff:
            t, b = text_eff["siglip2_text_tower"], text_eff["bamibert"]
            tp, tf, bp, bf = t["params_total"], t["flops_batch1"], b["params_total"], b["flops_batch1"]
            lat = lambda x: f"CPU batch-1 latency {x['cpu_latency_ms_4_threads']['median']} ms (4 threads) / {x['cpu_latency_ms_1_threads']['median']} ms (1 thread), {text_eff['machine']['cpu']}"
            L.append(f"| SigLIP 2 TEXT tower = T-SIG (once per question) | {tp:,} (token-embedding table {t['token_embedding_table_params']:,} + transformer {t['params_without_embedding_table']:,}) | {tf:.4e} | no, frozen | "
                     f"{t['input']}; {t['flops_batch1_20_tokens_unpadded']:.3e} FLOPs for a 20-token question without padding; {lat(t)} |")
            L.append(f"| BamiBERT text encoder (once per question; ablation ABL_bami) | {bp:,} (token-embedding table {b['token_embedding_table_params']:,} + transformer {b['params_without_embedding_table']:,}) | {bf:.4e} | no, frozen | {b['input']}; {lat(b)} |")
        for arch, e in (model_eff or {}).items():
            L.append(f"| {arch} (per question, without the encoders above) | {e['params']['total']:,} | {e['flops_batch1']:.4e} | yes | encoder {e['params']['encoder']:,}, heads {e['params']['heads_and_classifier']:,} |")
        if model_eff and "ViSO" in model_eff:
            vi = model_eff["ViSO"]
            vp_, vf_ = vi["params"]["total"], vi["flops_batch1"]
            if vision_eff and text_eff:
                main_bami = manifest.get("main_encoder") == "bami"
                tsig_row = (f"| {'**' if not main_bami else ''}ON-DEVICE TOTAL, T-SIG model (vision + text tower + ViSO){'**' if not main_bami else ''} | {'**' if not main_bami else ''}{vp + tp + vp_:,}{'**' if not main_bami else ''} | {vf + tf + vf_:.4e} | | first question on a new image; "
                            f"later questions on the same image: text tower + ViSO = {tp + vp_:,} params, {tf + vf_:.4e} FLOPs (the image tokens are cached) |")
                bami_row = (f"| {'**' if main_bami else ''}ON-DEVICE TOTAL, BamiBERT model{' (MAIN)' if main_bami else ' (ablation)'} (vision + BamiBERT + ViSO){'**' if main_bami else ''} | {'**' if main_bami else ''}{vp + bp + vp_:,}{'**' if main_bami else ''} | {vf + bf + vf_:.4e} | | first question on a new image; "
                            f"later questions on the same image: BamiBERT + ViSO = {bp + vp_:,} params, {bf + vf_:.4e} FLOPs; the ViSO row is the same layer sizes for both encoders (the text projection reads 768-d states) |")
                L += ([bami_row, tsig_row] if main_bami else [tsig_row, bami_row])
            elif vision_eff:
                L.append(f"| tower + ViSO (first question on a new image; text encoder NOT included) | {vp + vp_:,} | {vf + vf_:.4e} | | no text-tower figures were supplied |")
            if text_eff:
                L.append(f"| note | | | | the 196.6 M-parameter token-embedding table of the SigLIP 2 text tower is a lookup (no FLOPs) and could be pruned to the ~32 k question vocabulary for deployment; latency numbers are from the local CPU above, not from Kaggle |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO, "kaggle", "experiments_siglip2.json"))
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--s1-dir", default=os.path.join(REPO, "..", "vivqa_s1_v2", "data"))
    ap.add_argument("--packed", default=None, help="SigLIP 2 packed dir holding vision_tower_efficiency.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    manifest = json.load(open(a.manifest, encoding="utf-8"))
    veff = None
    if a.packed and os.path.isfile(os.path.join(os.path.expanduser(a.packed), "vision_tower_efficiency.json")):
        veff = json.load(open(os.path.join(os.path.expanduser(a.packed), "vision_tower_efficiency.json")))
    tpath = os.path.join(REPO, "kaggle", "text_tower_efficiency.json")
    teff = json.load(open(tpath)) if os.path.isfile(tpath) else None
    md = views_markdown(manifest, a.runs_dir, a.s1_dir, a.split, veff, None, teff)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
