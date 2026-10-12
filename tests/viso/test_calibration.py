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
