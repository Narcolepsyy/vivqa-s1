import numpy as np
import pytest
import torch

from utils import viso_calibration as C
from utils.viso_s1 import NEG_INF


def synthetic(n=5000, k=5, true_T=2.0, seed=0, scale=2.0):
    """Logits z; labels are drawn from softmax(z / true_T) -> the optimal temperature is true_T."""
    rng = np.random.RandomState(seed)
    z = rng.randn(n, k) * scale
    p = np.exp(z / true_T)
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p])
    return z, np.ones((n, k), bool), y


def test_temperature_recovered_within_10_percent():
    z, m, y = synthetic(true_T=2.0)
    t = C.fit_temperature(z, m, gold_idx=y)
    assert abs(t - 2.0) / 2.0 < 0.10
    z, m, y = synthetic(true_T=0.5, seed=1, scale=1.0)
    assert abs(C.fit_temperature(z, m, gold_idx=y) - 0.5) / 0.5 < 0.10


def test_temperature_is_clamped():
    z, m, y = synthetic(true_T=2.0)
    y_bad = (z.argmin(1))                      # anti-correlated labels push T to the upper bound
    t = C.fit_temperature(z * 50, m, gold_idx=y_bad)
    assert 0.1 <= t <= 10.0 and t == pytest.approx(10.0, rel=1e-3)
    t_lo = C.fit_temperature(z / 50, m, gold_idx=z.argmax(1))
    assert t_lo == pytest.approx(0.1, rel=1e-3)


def test_per_type_fits_are_independent():
    z0, m0, y0 = synthetic(2000, 5, 2.0, 0)
    z1, m1, y1 = synthetic(2000, 5, 0.5, 1, scale=1.0)
    z, m, y = np.vstack([z0, z1]), np.vstack([m0, m1]), np.concatenate([y0, y1])
    types = np.array([0] * 2000 + [1] * 2000)
    T = C.fit_temperatures(z, m, types, gold_idx=y)
    assert abs(T[0] - 2.0) < 0.2 and abs(T[1] - 0.5) < 0.05 and T[2] == 1.0       # type 2 has no items -> 1.0
    alone = C.fit_temperature(z0, m0, gold_idx=y0)
    assert T[0] == pytest.approx(alone, rel=1e-6)                                  # not affected by the other type


def test_soft_target_fit_matches_hard_fit_for_one_hot():
    z, m, y = synthetic()
    target = np.eye(5)[y]
    assert C.fit_temperature(z, m, target=target) == pytest.approx(C.fit_temperature(z, m, gold_idx=y), rel=1e-4)


def test_masked_columns_never_get_probability():
    z = np.random.RandomState(0).randn(4, 16)
    m = np.zeros((4, 16), bool); m[:, :2] = True
    p = C.probs_from_logits(z, m)
    assert np.allclose(p[:, :2].sum(1), 1.0) and float(p[:, 2:].sum()) == 0.0


def test_escalation_detects_errors_driven_by_max_prob():
    rng = np.random.RandomState(0)
    n = 3000
    mp = rng.uniform(0.5, 1.0, n)                             # max_prob of a 2-way distribution
    err = (mp < 0.75).astype(float)                           # error is a function of max_prob
    probs = np.zeros((n, 16)); probs[:, 0] = mp; probs[:, 1] = 1 - mp
    mask = np.zeros((n, 16), bool); mask[:, :2] = True
    types = rng.randint(0, 3, n)
    x = C.escalation_features(probs, mask, types)
    assert x.shape == (n, 9)
    oof = C.oof_predict(x, err)
    assert C.auroc(oof, err) > 0.9
    assert C.auroc(1 - mp, err) > 0.9
    f1, f2 = C.fold_ids(n), C.fold_ids(n)
    assert np.array_equal(f1, f2) and sorted(np.bincount(f1).tolist()) == [600] * 5


def test_auroc_and_selective_metrics_hand_cases():
    assert C.auroc([0.1, 0.9], [0, 1]) == 1.0 and C.auroc([0.9, 0.1], [0, 1]) == 0.0
    assert C.auroc([0.5, 0.5], [0, 1]) == 0.5
    s = C.selective_metrics([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0])
    assert s["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4) and s["sel_acc@0.5"] == 1.0


def test_correct_fraction_ties():
    p = np.array([[0.5, 0.5, 0.0], [0.2, 0.7, 0.1]])
    assert C.correct_fraction(p, np.array([1, 1])).tolist() == [0.5, 1.0]


def test_vector_calibration_corrects_fixed_space_bias():
    rng = np.random.RandomState(42)
    n = 1000
    z = np.zeros((n, 16), dtype=np.float64)
    z[:, :2] = rng.randn(n, 2)
    true_bias = np.array([-0.8, 0.8])
    p_true = np.exp(z[:, :2] + true_bias)
    p_true /= p_true.sum(1, keepdims=True)
    y = np.array([rng.choice(2, p=row) for row in p_true])
    mask = np.zeros((n, 16), dtype=bool)
    mask[:, :2] = True
    types = np.zeros(n, dtype=np.int64)
    temps, biases = C.fit_vector_calibration(z, mask, types, gold_idx=y)
    assert biases[0][1] - biases[0][0] > 0.5
    p_vec = C.probs_from_logits(z, mask, types, temps, biases=biases)
    p_scal = C.probs_from_logits(z, mask, types, temps)
    acc_vec = (p_vec[:, :2].argmax(1) == y).mean()
    acc_scal = (p_scal[:, :2].argmax(1) == y).mean()
    assert acc_vec >= acc_scal


def test_escalation_interactions_and_member_disagreement():
    rng = np.random.RandomState(0)
    n = 100
    probs = np.zeros((n, 16))
    probs[:, 0] = 0.7
    probs[:, 1] = 0.3
    mask = np.zeros((n, 16), bool)
    mask[:, :2] = True
    types = rng.randint(0, 3, n)
    member_probs = np.stack([probs, probs[:, ::-1]], axis=0)
    x_inter = C.escalation_features(probs, mask, types, include_interactions=True)
    assert x_inter.shape == (n, 15)
    x_mem = C.escalation_features(probs, mask, types, include_interactions=True, member_probs=member_probs)
    assert x_mem.shape == (n, 20)


def test_cascade_concave_envelope_and_dp_monotonicity():
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
    import viso_cascade as VC
    rng = np.random.RandomState(7)
    gains = []
    for _ in range(3):
        raw = np.concatenate([[0.0], np.cumsum(rng.choice([-1.0, 0.0, 1.0], size=50))])
        env = VC.concave_envelope(raw)
        inc = np.diff(env)
        assert np.all(np.diff(inc) <= 1e-9)  # non-increasing marginal increments (concave)
        gains.append(env)
    plan = VC.dp_budget(gains, 150)
    for k in range(150):
        c_curr, c_next = plan[k], plan[k + 1]
        assert all(c_next[t] >= c_curr[t] for t in range(3))


def test_vivqa_s1_dataset_hf_dict_of_lists_and_letter_debiasing(tmp_path):
    import json
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from vivqa_s1.dataset import ViVQAS1Dataset
    row = {
        "id": 1, "image": "000001.jpg", "coco_id": 1, "question": "Có mèo không?",
        "question_type": "noul", "choices": ["không", "có"], "answer": "true",
        "answer_distribution": {"key": ["false", "true"], "value": [0.2, 0.8]},
        "gold_source": "human10", "split": "dev"
    }
    (tmp_path / "dev.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    ds = ViVQAS1Dataset(split="dev", data_dir=tmp_path, load_image=False, use_parquet=False)
    it = ds[0]
    assert it["answer"] == "có" and it["answer_idx"] == 1
    assert it["answer_distribution"] == {"không": pytest.approx(0.2), "có": pytest.approx(0.8)}


def test_colab_residual_decision_head_starts_at_exact_zeroshot_parity():
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "kaggle")))
    import colab_heavy_boost as CHB
    rng = np.random.RandomState(0)
    W0 = torch.from_numpy(rng.randn(16, 32).astype(np.float32))
    x = torch.from_numpy(rng.randn(8, 32).astype(np.float32))
    lin_head = CHB.ResidualDecisionHead(W0, arch="linear")
    res_head = CHB.ResidualDecisionHead(W0, arch="res_mlp", bottleneck=16)
    with torch.no_grad():
        assert torch.allclose(lin_head(x, train_mode=False), x @ W0.T, atol=1e-6)
        assert torch.allclose(res_head(x, train_mode=False), x @ W0.T, atol=1e-6)


