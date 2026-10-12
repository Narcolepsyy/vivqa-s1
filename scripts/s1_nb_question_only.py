"""Question-only naive Bayes on the yes/no items (word uni+bigrams, add-one smoothing), Vietnamese question vs the ENGLISH source question, v1 and v2 benchmarks, dev and test.
    python scripts/s1_nb_question_only.py
Shows whether yes/no is answerable from the question text alone (it is not: ~0.53 in both languages) - a benchmark property, not a translation effect. No image, no model weights."""
import collections
import json
import math
import os
import re

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def feats(text):
    w = re.findall(r"\w+", (text or "").lower())
    return set(w) | {a + "_" + b for a, b in zip(w, w[1:])}


def nb(train, test, field):
    cnt = {True: collections.Counter(), False: collections.Counter()}
    n = collections.Counter()
    for it in train:
        y = it["gold_label"] == "true"
        n[y] += 1
        cnt[y].update(feats(field(it)))
    vocab = len(set(cnt[True]) | set(cnt[False]))
    ok = 0
    for it in test:
        s = {}
        for y in (True, False):
            tot = sum(cnt[y].values())
            s[y] = math.log(n[y] / (n[True] + n[False])) + sum(math.log((cnt[y][f] + 1) / (tot + vocab)) for f in feats(field(it)))
        ok += (s[True] > s[False]) == (it["gold_label"] == "true")
    return ok / len(test)


def main():
    print("| benchmark | split | n yes/no | NB on Vietnamese question | NB on English question | 'true' share |")
    print("|---|---|---|---|---|---|")
    for v in ("vivqa_s1", "vivqa_s1_v2"):                              # v1 was archived (~/data/archive_v1x); only the existing versions are evaluated
        if not os.path.isdir(f"{PROJECT}/{v}"):
            continue
        tr = [it for it in rd(f"{PROJECT}/{v}/data/train.jsonl") if it["questions"]["q"]["type"] == "noul"]
        for sp in ("dev", "test"):
            te = [it for it in rd(f"{PROJECT}/{v}/data/{sp}.jsonl") if it["questions"]["q"]["type"] == "noul"]
            vi = nb(tr, te, lambda it: it["questions"]["q"]["instructions"])
            en = nb(tr, te, lambda it: it["meta"].get("en_q"))
            share = sum(it["gold_label"] == "true" for it in te) / len(te)
            print(f"| {v} | {sp} | {len(te)} | {vi:.3f} | {en:.3f} | {share:.3f} |")


if __name__ == "__main__":
    main()
