"""Gradient audit of the RLCD objective on the LOGITS of a finished soft-CE run (no model, CPU, seconds): is the policy-gradient (PG) term's sign right, and how large is it next to the soft-CE gradient?
  python scripts/viso_rl_gradcheck.py --run-dir <runs/A2_s0> --s1-dir ../vivqa_s1_v2/data [--split dev] [--sigmas 0.4 0.25 0.1]
For every item (logits z = the run's dev logits, soft gold g) it computes with autograd, w.r.t. z:
  g_ce   = grad of soft_ce(z, g)                                   (what stage 1 optimised; ||g_ce|| = 0 at a perfect fit)
  g_pg   = grad of utils.viso_rlcd.policy_gradient_loss(z, ...)    (K = NUM_SAMPLES noisy samples, as in training)
  g_true = grad of -E_eps[ reward(softmax(z + sigma*eps)) ] estimated with MANY reparameterised samples (the direction PG should follow, up to noise)
and reports per type: mean ||g_ce||, mean ||g_pg||, the ratio, cos(g_pg, g_true) (> 0 = the sign is right), the same cosine averaged over many independent PG draws, and the share of the summed
gradient norm that comes from PG. The training uses AdamW + global-norm clipping, so a gradient dominated by zero-mean PG noise moves the weights as a random walk of the size of the learning rate."""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from utils import viso_calibration as C  # noqa: E402
from utils import viso_rlcd as R  # noqa: E402
from utils.viso_s1 import MAX_SPACE, label_space, read_jsonl  # noqa: E402

TYPES = ("noul", "score", "choice")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--s1-dir", required=True, help="<benchmark>/data")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--sigmas", type=float, nargs="*", default=[0.4, 0.25, 0.1])
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-items", type=int, default=1500)
    ap.add_argument("--true-samples", type=int, default=4000)
    ap.add_argument("--draws", type=int, default=200, help="independent PG draws averaged for the cosine")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    lg = C.load_logits(os.path.join(a.run_dir, f"logits_{a.split}.npz"))
    by_id = {it["id"]: it for it in read_jsonl(os.path.join(a.s1_dir, f"{a.split}.jsonl"))}
    items = [by_id[int(i)] for i in lg["ids"]][:a.max_items]
    n = len(items)
    z0 = torch.from_numpy(np.asarray(lg["logits"][:n], np.float32))
    mask = torch.from_numpy(np.asarray(lg["space_mask"][:n], bool))
    tid = torch.from_numpy(np.asarray(lg["type_id"][:n], np.int64))
    tgt = torch.zeros(n, MAX_SPACE)
    for i, it in enumerate(items):
        sp = label_space(it["questions"]["q"])
        for j, k in enumerate(sp):
            tgt[i, j] = it["gold"]["q"]["probabilities"].get(k, 0.0)
    z0 = z0.masked_fill(~mask, 0.0)
    rep = {"run_dir": a.run_dir, "split": a.split, "n_items": n, "num_samples": a.num_samples, "sigmas": {}}

    z = z0.clone().requires_grad_(True)
    ce = R.soft_ce(z, mask, tgt)
    g_ce, = torch.autograd.grad(ce.sum(), z)
    rep["soft_ce_gradient_norm_by_type"] = {t: float(g_ce[tid == k].norm(dim=-1).mean()) for k, t in enumerate(TYPES)}
    rep["soft_ce_mean_by_type"] = {t: float(ce.detach()[tid == k].mean()) for k, t in enumerate(TYPES)}
    for sigma in a.sigmas:
        # true reward gradient: reparameterised expectation with many samples (autograd through softmax(z + sigma * eps))
        gen = torch.Generator().manual_seed(0)
        zt = z0.clone().requires_grad_(True)
        eps = torch.randn((n, a.true_samples, MAX_SPACE), generator=gen) * mask[:, None, :]
        p = torch.softmax((zt[:, None, :] + sigma * eps).masked_fill(~mask[:, None, :], -1e9), dim=-1)
        reward = R.rewards(p, tgt, tid).mean(1)
        g_true, = torch.autograd.grad(reward.sum(), zt)                     # ascent direction of the expected reward
        norms, cos_one, cos_avg_num = [], [], torch.zeros_like(z0)
        gen = torch.Generator().manual_seed(1)
        for d in range(a.draws):
            zp = z0.clone().requires_grad_(True)
            pg, _ = R.policy_gradient_loss(zp, mask, tgt, tid, sigma, gen, a.num_samples)
            g_pg, = torch.autograd.grad(pg.sum(), zp)                        # gradient of the LOSS: should point against g_true
            norms.append(g_pg.norm(dim=-1))
            cos_avg_num += g_pg
            if d < 20:
                cos_one.append(torch.nn.functional.cosine_similarity(-g_pg, g_true, dim=-1, eps=1e-12))
        norms = torch.stack(norms).mean(0)
        avg = -cos_avg_num / a.draws
        cos_avg = torch.nn.functional.cosine_similarity(avg, g_true, dim=-1, eps=1e-12)
        cos_one = torch.stack(cos_one).mean(0)
        rows = {}
        for k, t in enumerate(TYPES):
            m = tid == k
            nce, npg = float(g_ce[m].norm(dim=-1).mean()), float(norms[m].mean())
            rows[t] = {"grad_norm_soft_ce": round(nce, 4), "grad_norm_pg": round(npg, 4), "pg_over_ce": round(npg / max(nce, 1e-9), 2),
                       "cos(-g_pg_single_draw, true reward gradient)": round(float(cos_one[m].mean()), 4),
                       f"cos(-mean of {a.draws} PG draws, true reward gradient)": round(float(cos_avg[m].mean()), 4),
                       "true_reward_gradient_norm": round(float(g_true[m].norm(dim=-1).mean()), 4),
                       "cos(soft-CE descent direction, true reward gradient)": round(float(torch.nn.functional.cosine_similarity(-g_ce, g_true, dim=-1, eps=1e-12)[m].mean()), 4)}
        rep["sigmas"][str(sigma)] = rows
    s = json.dumps(rep, indent=1)
    print(s)
    if a.out:
        open(a.out, "w").write(s)
    return rep


if __name__ == "__main__":
    main()
