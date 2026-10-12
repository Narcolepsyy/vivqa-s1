"""Pure functions for the ViSO objectives: masked cross-entropies and RLCD (soft-CE + a GRPO-style policy-gradient
term with proper-scoring-rule rewards). All tensors use the shared 16-wide layout; `mask` is the bool space_mask
(True = valid label). Masked logits carry -1e9 and never receive probability mass."""
import torch

from .viso_s1 import NEG_INF

SCORE_TYPE_ID = 1       # ordinal (RPS) reward applies to `score` items only
N_LEVELS = 11


def masked_log_softmax(logits, mask):
    return torch.log_softmax(logits.masked_fill(~mask, NEG_INF), dim=-1)


def hard_ce(logits, mask, gold_idx):
    """Per-item CE against the one-hot gold_idx."""
    return -masked_log_softmax(logits, mask).gather(1, gold_idx[:, None]).squeeze(1)


def soft_ce(logits, mask, target):
    """Per-item -sum_i target_i log softmax(logits)_i over the masked space."""
    logp = masked_log_softmax(logits, mask).masked_fill(~mask, 0.0)
    return -(target * logp).sum(-1)


def spherical(p, g):
    """Spherical score sum_i g_i p_i / ||p||_2 (last dim = labels); in [0,1], 1 iff p == one-hot on the gold mode."""
    return (g * p).sum(-1) / p.norm(dim=-1).clamp(min=1e-12)


def rps(p, g):
    """Ranked probability score over ordered levels (last dim K), normalised by K-1; same definition as
    s1bench.metrics.rps."""
    k = p.shape[-1]
    return ((p.cumsum(-1) - g.cumsum(-1))[..., :k - 1] ** 2).sum(-1) / (k - 1)


def rewards(p, g, type_id, spherical_weight=0.75, rps_weight=1.0):
    """p: (B,K,16) sampled decision distributions, g: (B,16) soft gold, type_id: (B,) -> (B,K) rewards.
    r = spherical_weight * spherical(p, g); score items add rps_weight * (1 - RPS(p, g)) over the 11 ordered
    levels. noul / choice are unordered, so RPS is not used for them."""
    g_exp = g[:, None, :].expand_as(p)
    r = spherical_weight * spherical(p, g_exp)
    is_score = (type_id == SCORE_TYPE_ID)[:, None]
    ordinal = 1.0 - rps(p[..., :N_LEVELS], g_exp[..., :N_LEVELS])
    return r + rps_weight * ordinal * is_score


def sigma_at(progress, sigma_start=0.4, sigma_end=0.1):
    """Linear annealing 0.4 -> 0.1; progress in [0,1] (fraction of the RL training done)."""
    progress = min(max(float(progress), 0.0), 1.0)
    return sigma_start + (sigma_end - sigma_start) * progress


def group_advantage(r, eps=1e-6):
    """(B,K) -> per-item standardised advantage over the K samples (population std)."""
    return (r - r.mean(1, keepdim=True)) / (r.std(1, unbiased=False, keepdim=True) + eps)


def policy_gradient_loss(logits, mask, target, type_id, sigma, generator, num_samples=4,
                         spherical_weight=0.75, rps_weight=1.0):
    """GRPO-style term. For each item draw K noisy logits z_k = z.detach() + sigma * eps_k (eps ~ N(0, I) on the
    valid columns, from the dedicated `generator`), p_k = softmax(z_k), reward r_k as in `rewards`, group advantage
    A_k, and loss = - mean_k [A_k.detach() * log N(z_k; z, sigma^2 I)] with z_k a constant and
    log N = -||z_k - z||^2 / (2 sigma^2) over the valid columns (z carries the gradient).
    Returns (per-item loss (B,), mean reward)."""
    z = logits.masked_fill(~mask, 0.0)
    eps = torch.randn((z.shape[0], num_samples, z.shape[1]), generator=generator).to(z.device)
    eps = eps * mask[:, None, :]
    z_k = z.detach()[:, None, :] + sigma * eps
    p_k = torch.softmax(z_k.masked_fill(~mask[:, None, :], NEG_INF), dim=-1)
    r = rewards(p_k, target, type_id, spherical_weight, rps_weight)
    adv = group_advantage(r).detach()
    log_n = -(((z_k - z[:, None, :]) * mask[:, None, :]) ** 2).sum(-1) / (2.0 * sigma ** 2)
    return -(adv * log_n).mean(1), r.mean().detach()


def reduce_loss(per_item, type_id, balance=False):
    """Mean over items, or (balance=True) the mean of the per-type means."""
    if not balance:
        return per_item.mean()
    means = [per_item[type_id == t].mean() for t in range(3) if bool((type_id == t).any())]
    return torch.stack(means).mean()
