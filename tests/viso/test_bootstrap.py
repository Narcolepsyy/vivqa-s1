import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import viso_bootstrap as B  # noqa: E402

N = 400


def noul_items(n=N, human_every=2):
    return [{"id": i, "questions": {"q": {"type": "noul", "instructions": "x"}},
             "gold": {"q": {"probabilities": {"true": 1.0}}}, "gold_label": "true", "human_subset": (i % human_every == 0)}
            for i in range(n)]


def preds_with_accuracy(n, acc, conf=0.9, seed=0):
    rng = np.random.RandomState(seed)
    ok = rng.permutation(n) < int(acc * n)
    return {i: {"probabilities": {"true": conf if ok[i] else 1 - conf, "false": 1 - conf if ok[i] else conf}} for i in range(n)}


def test_known_difference_is_recovered_and_significant():
    items = noul_items()
    sa, sb = B.scores_for(items, preds_with_accuracy(N, 0.8, seed=1)), B.scores_for(items, preds_with_accuracy(N, 0.6, seed=2))
    idx = B.resample_indices(N, 500, 0)
    a, b = B.boot_primary(sa, idx), B.boot_primary(sb, idx)
    pa, pb = B.point_metrics(sa), B.point_metrics(sb)
    d = B.paired(a, b, pa, pb, B.PRIMARY)["accuracy"]
    assert d["diff"] == pytest.approx(0.2, abs=1e-9)
    lo, hi = d["ci95"]
    assert 0.1 < lo < 0.2 < hi < 0.3 and d["share_a_gt_b"] > 0.99
    assert B.paired(a, a, pa, pa, B.PRIMARY)["accuracy"]["ci95"] == [0.0, 0.0]          # a method against itself


def test_identical_methods_have_a_ci_that_contains_zero():
    items = noul_items()
    sa, sb = B.scores_for(items, preds_with_accuracy(N, 0.7, seed=1)), B.scores_for(items, preds_with_accuracy(N, 0.7, seed=2))
    idx = B.resample_indices(N, 500, 0)
    d = B.paired(B.boot_primary(sa, idx), B.boot_primary(sb, idx), B.point_metrics(sa), B.point_metrics(sb), ("accuracy",))["accuracy"]
    assert d["ci95"][0] < 0 < d["ci95"][1]


def test_bootstrap_is_deterministic_and_uses_shared_indices():
    assert np.array_equal(B.resample_indices(50, 20, 0), B.resample_indices(50, 20, 0))
    assert not np.array_equal(B.resample_indices(50, 20, 0), B.resample_indices(50, 20, 1))
    items = noul_items(100)
    sc = B.scores_for(items, preds_with_accuracy(100, 0.7))
    x, y = B.boot_primary(sc, B.resample_indices(100, 30, 0)), B.boot_primary(sc, B.resample_indices(100, 30, 0))
    assert all(np.array_equal(x[k], y[k]) for k in x)


def test_metrics_come_from_s1bench_scores():
    from s1bench import metrics as M
    items = noul_items(100)
    preds = preds_with_accuracy(100, 0.7)
    sc = B.scores_for(items, preds)
    ref = M.evaluate(items, preds)["primary"]["all"]
    pt = B.point_metrics(sc)
    assert pt["accuracy"] == pytest.approx(ref["accuracy"]) and pt["nll"] == pytest.approx(ref["nll"]) and pt["ece"] == pytest.approx(ref["ece"])
    full = B.boot_primary(sc, np.arange(100)[None, :])                    # one "resample" = the data itself
    assert full["accuracy"][0] == pytest.approx(ref["accuracy"]) and full["aurc"][0] == pytest.approx(ref["aurc"])


def test_mean_std_over_seeds():
    r = B.mean_std([1.0, 2.0, 3.0])
    assert r["mean"] == 2.0 and r["std"] == pytest.approx(1.0) and r["n"] == 3
    assert B.mean_std([5.0])["std"] is None and B.mean_std([])["mean"] is None


def test_run_over_fake_runs_dir_with_two_seeds(tmp_path):
    items = noul_items()
    s1 = tmp_path / "s1"
    s1.mkdir()
    with open(s1 / "test.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    manifest = {"jobs": [{"id": f"{fam}_s{s}", "seed": s} for fam in ("FA", "FB") for s in (0, 1)],
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
    res = B.run(manifest, str(runs), str(s1), resamples=100)
    c = res["comparisons"]["A_vs_B"]
    assert set(c["per_seed"]) == {"0", "1"}
    d = c["diff_mean_std_over_seeds"]["accuracy"]
    assert d["mean"] == pytest.approx(0.2, abs=1e-9) and d["n"] == 2
    assert res["methods"]["A"]["mean_std"]["accuracy"]["mean"] == pytest.approx(0.8)
    assert c["per_seed"]["0"]["s1_human"]["soft_accuracy"]["ci95"][0] > 0
    assert B.run(manifest, str(runs), str(s1), resamples=100) == res          # deterministic
