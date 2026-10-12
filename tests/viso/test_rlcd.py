import math
import sys

import pytest
import torch

from utils import viso_rlcd as R
from utils.viso_s1 import NEG_INF
import utils.viso_s1  # noqa: F401  (puts s1bench on sys.path)
from s1bench.metrics import rps as s1_rps


def onehot(k, n=16, i=0):
    v = torch.zeros(n)
    v[i] = 1.0
    return v


def test_rps_matches_s1bench_and_hand_cases():
    p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    g = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert abs(float(R.rps(p, g)) - s1_rps(p.tolist(), g.tolist())) < 1e-6
    assert float(R.rps(g, g)) == 0.0
    worst = R.rps(torch.tensor([1.0, 0, 0]), torch.tensor([0, 0, 1.0]))
    assert abs(float(worst) - 1.0) < 1e-6                      # worst case = 1 after normalising by K-1
    near = R.rps(torch.tensor([0, 1.0, 0, 0]), torch.tensor([1.0, 0, 0, 0]))
    far = R.rps(torch.tensor([0, 0, 0, 1.0]), torch.tensor([1.0, 0, 0, 0]))
    assert float(near) < float(far)                            # ordinal: nearer is better


def test_spherical_hand_cases():
    g = torch.tensor([0.0, 1.0, 0.0])
    assert abs(float(R.spherical(torch.tensor([0.0, 1.0, 0.0]), g)) - 1.0) < 1e-6      # perfect = best
    assert abs(float(R.spherical(torch.tensor([0.5, 0.5, 0.0]), g)) - 0.5 / math.sqrt(0.5)) < 1e-6
    assert float(R.spherical(torch.tensor([1.0, 0.0, 0.0]), g)) == 0.0


def test_reward_perfect_prediction_is_best_and_rps_only_for_score():
    g = torch.zeros(2, 16); g[:, 3] = 1.0
    perfect, wrong = torch.zeros(2, 1, 16), torch.zeros(2, 1, 16)
    perfect[:, 0, 3] = 1.0
    wrong[:, 0, 0] = 1.0
    types = torch.tensor([0, 1])                                # noul, score
    rp, rw = R.rewards(perfect, g, types), R.rewards(wrong, g, types)
    assert bool((rp > rw).all())
    assert abs(float(rp[0, 0]) - 0.75) < 1e-6                    # noul: 0.75 * spherical only
    assert abs(float(rp[1, 0]) - (0.75 + 1.0)) < 1e-6            # score: + 1.0 * (1 - RPS(=0))


def test_advantages_have_zero_mean_per_group():
    r = torch.randn(7, 4, generator=torch.Generator().manual_seed(1))
    a = R.group_advantage(r)
    assert float(a.mean(1).abs().max()) < 1e-6
    assert float(R.group_advantage(torch.ones(2, 4)).abs().max()) == 0.0   # constant rewards -> zero advantage


def make_batch(b=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, 16, generator=g)
    mask = torch.zeros(b, 16, dtype=torch.bool)
    types = torch.tensor([0, 1, 2, 0, 1, 2])[:b]
    for i, t in enumerate(types.tolist()):
        mask[i, :{0: 2, 1: 11, 2: 16}[t]] = True
    z = z.masked_fill(~mask, NEG_INF)
    target = torch.zeros(b, 16)
    for i in range(b):
        n = int(mask[i].sum())
        target[i, int(torch.randint(n, (1,), generator=g))] = 1.0
    return z, mask, target, types


def test_pg_loss_deterministic_for_a_fixed_seed():
    z, mask, target, types = make_batch()
    a, ra = R.policy_gradient_loss(z, mask, target, types, 0.4, torch.Generator().manual_seed(5))
    b, rb = R.policy_gradient_loss(z, mask, target, types, 0.4, torch.Generator().manual_seed(5))
    c, _ = R.policy_gradient_loss(z, mask, target, types, 0.4, torch.Generator().manual_seed(6))
    assert torch.equal(a, b) and float(ra) == float(rb) and not torch.equal(a, c)


def test_pg_gradient_is_finite_and_nonzero_and_masked():
    z, mask, target, types = make_batch()
    z = z.clone().requires_grad_(True)
    loss, _ = R.policy_gradient_loss(z, mask, target, types, 0.4, torch.Generator().manual_seed(0))
    R.reduce_loss(loss, types).backward()
    assert torch.isfinite(z.grad).all() and float(z.grad.abs().sum()) > 0
    assert float(z.grad[~mask].abs().sum()) == 0.0               # no gradient on masked columns


def test_sigma_schedule_endpoints():
    assert abs(R.sigma_at(0.0) - 0.4) < 1e-12 and abs(R.sigma_at(1.0) - 0.1) < 1e-12
    assert abs(R.sigma_at(0.5) - 0.25) < 1e-12 and R.sigma_at(2.0) == R.sigma_at(1.0)


def test_cross_entropies_and_type_balance():
    z, mask, target, types = make_batch()
    gold = target.argmax(1)
    assert torch.allclose(R.soft_ce(z, mask, target), R.hard_ce(z, mask, gold), atol=1e-6)   # one-hot targets
    per = torch.arange(6, dtype=torch.float)
    assert float(R.reduce_loss(per, types)) == pytest.approx(2.5)
    assert float(R.reduce_loss(per, types, balance=True)) == pytest.approx((1.5 + 2.5 + 3.5) / 3)


def test_perfect_logits_give_near_zero_ce():
    z, mask, target, types = make_batch()
    sharp = torch.full_like(z, NEG_INF)
    sharp = torch.where(mask, torch.full_like(z, -20.0), sharp)
    sharp[torch.arange(6), target.argmax(1)] = 20.0
    assert float(R.soft_ce(sharp, mask, target).max()) < 1e-6
