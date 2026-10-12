"""Deterministic synthetic scoring case shared by the view tests and the golden-file generator (noul items, two seeds)."""
import json
import os

import numpy as np

N = 400


def noul_items(n=N):
    return [{"id": i, "questions": {"q": {"type": "noul", "instructions": "x"}},
             "gold": {"q": {"probabilities": {"true": 1.0}}}, "gold_label": "true", "human_subset": (i % 2 == 0)} for i in range(n)]


def preds_with_accuracy(n, acc, conf=0.9, seed=0):
    rng = np.random.RandomState(seed)
    ok = rng.permutation(n) < int(acc * n)
    return {i: {"probabilities": {"true": conf if ok[i] else 1 - conf, "false": 1 - conf if ok[i] else conf}} for i in range(n)}


def build_case(tmp_path):
    """-> (manifest, runs_dir, s1_data_dir). s1_data_dir/test.jsonl holds the gold items."""
    s1 = tmp_path / "s1" / "data"
    s1.mkdir(parents=True)
    with open(s1 / "test.jsonl", "w") as f:
        for it in noul_items():
            f.write(json.dumps(it) + "\n")
    manifest = {"jobs": [{"id": f"{fam}_s{s}", "seed": s, "method": fam} for fam in ("FA", "FB") for s in (0, 1)],
                "methods": {"A": {"family": "FA", "variant": "uncal"}, "B": {"family": "FB", "variant": "uncal"}},
                "comparisons": [["A", "B"]]}
    runs = tmp_path / "runs"
    for fam, acc in (("FA", 0.8), ("FB", 0.6)):
        for s in (0, 1):
            d = runs / f"{fam}_s{s}"
            d.mkdir(parents=True)
            with open(d / "preds_test.jsonl", "w") as f:
                for i, p in preds_with_accuracy(N, acc, seed=10 * s + (fam == "FB")).items():
                    f.write(json.dumps({"id": i, "probabilities": p["probabilities"]}) + "\n")
    return manifest, str(runs), str(s1)
