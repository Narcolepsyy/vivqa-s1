"""Group-1 sanity baselines on ViVQA-S1 (no image, no trained model): uniform, class prior, question-only prior (+ the VQA question-type prior as an oracle-ish bound),
generic-candidate shortcut, English-word (Latin-script) giveaway. Predictions are written next to the results (NOT into the benchmark directory) and scored with the
benchmark's own evaluate.py code on All / Core / Image-unseen (+ S1-Human), per split.
    python scripts/s1_sanity_baselines.py [--s1-root ../vivqa_s1_v2] [--out ~/data/s1_sanity] [--splits dev test]
Hard-pick baselines (generic shortcut, English giveaway) put 0.9 on the pick: read their accuracy only (their calibration is meaningless).
Uses only the benchmark's frozen train.jsonl to fit priors. No network, no keys."""
import argparse
import importlib.util
import json
import os
import sys
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PICK = 0.9


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-root", default=os.path.join(REPO, "..", "vivqa_s1_v2"))
    ap.add_argument("--out", default="~/data/s1_sanity")
    ap.add_argument("--splits", nargs="*", default=["dev", "test"])
    a = ap.parse_args()
    root = os.path.abspath(a.s1_root)
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "baselines"))
    import prior_baseline as PB                                   # noqa: E402  (the benchmark's own prior code)
    from s1bench.distractors import GENERIC_K, key               # noqa: E402
    from s1bench.fixes import is_foreign_latin, vn_lexicon        # noqa: E402
    EV = load(os.path.join(root, "evaluate.py"), "s1_evaluate")
    data = os.path.join(root, "data")
    train = PB.read(os.path.join(data, "train.jsonl"))
    by_type, by_key = PB.fit(train)
    freq = Counter(it["gold_label"] for it in train if it["questions"]["q"]["type"] == "choice")
    allq = [it["questions"]["q"]["instructions"] for sp in ("train", "calib", "dev", "test") for it in PB.read(os.path.join(data, f"{sp}.jsonl"))]
    lex = vn_lexicon(allq)

    def pick(pr, chosen):
        rest = {c: p for c, p in pr.items() if c != chosen}
        z = sum(rest.values()) or 1.0
        return {c: (PICK if c == chosen else (1 - PICK) * rest.get(c, 0.0) / z) for c in pr}

    def generic(it, by_type, by_key):
        pr = PB.predict(it, "key", by_type, by_key)
        q = it["questions"]["q"]
        gen = sorted((c for c in q.get("criteria", {}) if key(c) in GENERIC_K), key=lambda c: (-freq.get(c, 0), c))
        return pick(pr, gen[0]) if q["type"] == "choice" and gen else pr

    def english(it, by_type, by_key):
        """the candidate written in Latin script that is not Vietnamese (a brand / name / English word) is picked when it is the only one"""
        pr = PB.predict(it, "key", by_type, by_key)
        q = it["questions"]["q"]
        lat = [c for c in q.get("criteria", {}) if is_foreign_latin(c, lex)]
        return pick(pr, lat[0]) if q["type"] == "choice" and len(lat) == 1 else pr

    baselines = {"uniform": lambda it, *_: PB.predict(it, "uniform", by_type, by_key), "class_prior": lambda it, *_: PB.predict(it, "type", by_type, by_key),
                 "question_only_prior": lambda it, *_: PB.predict(it, "key", by_type, by_key), "vqa_question_type_prior": lambda it, *_: PB.predict(it, "vqakey", by_type, by_key),
                 "generic_shortcut": lambda it, *_: generic(it, by_type, by_key), "english_word_giveaway": lambda it, *_: english(it, by_type, by_key)}
    results = {}
    for sp in a.splits:
        gold = PB.read(os.path.join(data, f"{sp}.jsonl"))
        flags = os.path.join(data, "eval_flags.jsonl")
        for name, fn in baselines.items():
            preds = {it["id"]: {"id": it["id"], "probabilities": fn(it)} for it in gold}
            with open(os.path.join(out, "preds", f"{name}_{sp}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                for it in gold:
                    f.write(json.dumps(preds[it["id"]], ensure_ascii=False) + "\n")
            res = {"all": EV.evaluate(gold, preds)}
            if sp == "test" and os.path.exists(flags):
                _, views = EV.view_blocks(gold, preds, flags, f"{sp}.jsonl")
                res.update(views)
            results.setdefault(name, {})[sp] = res
    json.dump(results, open(os.path.join(out, "sanity_results.json"), "w"), indent=1, ensure_ascii=False)
    L = ["# Group-1 sanity baselines (no image, no trained model)", "", f"benchmark: {root}", ""]
    for sp in a.splits:
        L += [f"## {sp}: accuracy / NLL / ECE by view (all types)", "", "| baseline | All acc | All nll | All ece | Core acc | Image-unseen acc | S1-Human soft acc (All) |", "|---|---|---|---|---|---|---|"]
        for name in baselines:
            r = results[name][sp]
            g = lambda v, m: (r.get(v) or {}).get("primary", {}).get("all", {}).get(m)
            f = lambda x: "-" if x is None else f"{x:.4f}"
            sh = r["all"]["s1_human"]["all"]["soft_accuracy"]
            L.append(f"| {name} | {f(g('all', 'accuracy'))} | {f(g('all', 'nll'))} | {f(g('all', 'ece'))} | {f(g('core', 'accuracy'))} | {f(g('image_unseen', 'accuracy'))} | {f(sh)} |")
        L.append("")
    L += ["Per type (test, All view): ", ""]
    for name in baselines:
        if "test" in a.splits:
            p = results[name]["test"]["all"]["primary"]
            L.append(f"- {name}: " + ", ".join(f"{t} {p[t]['accuracy']:.3f}" for t in ("noul", "score", "choice") if t in p))
    open(os.path.join(out, "sanity_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
