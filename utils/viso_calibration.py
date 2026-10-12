"""Per-type temperature scaling, ECE helpers, logits/prediction IO and the post-hoc escalation head.
Pure numpy/torch, CPU, deterministic."""
import json
import math

import numpy as np
import torch

from .viso_s1 import MAX_SPACE, NEG_INF, TYPE_IDS, label_space

LOG_T_MIN, LOG_T_MAX = math.log(0.1), math.log(10.0)


# --------------------------------------------------------------------------------------------- gold / IO
def gold_arrays(items):
    """items: S1 jsonl dicts -> dict(ids, type_id, space_mask[N,16], gold_idx[N], target[N,16], human[N])."""
    n = len(items)
    out = dict(ids=np.zeros(n, np.int64), type_id=np.zeros(n, np.int64), space_mask=np.zeros((n, MAX_SPACE), bool),
               gold_idx=np.zeros(n, np.int64), target=np.zeros((n, MAX_SPACE), np.float32), human=np.zeros(n, bool))
    for i, it in enumerate(items):
        q = it["questions"]["q"]
        space = label_space(q)
        out["ids"][i], out["type_id"][i] = it["id"], TYPE_IDS[q["type"]]
        out["space_mask"][i, :len(space)] = True
        out["gold_idx"][i] = space.index(it["gold_label"])
        for k, key in enumerate(space):
            out["target"][i, k] = it["gold"]["q"]["probabilities"].get(key, 0.0)
        out["human"][i] = bool(it["human_subset"])
    return out


def save_logits(path, ids, type_id, logits, space_mask):
    from utils.atomic import atomic_savez
    atomic_savez(path, ids=np.asarray(ids), type_id=np.asarray(type_id), logits=np.asarray(logits, np.float32), space_mask=np.asarray(space_mask))


def load_logits(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def probs_from_logits(logits, space_mask, type_id=None, temps=None):
    """softmax(logits / T[type]) over the masked space. temps: None (T=1), a float, or {type_id: T}."""
    z = torch.as_tensor(logits, dtype=torch.float64)
    m = torch.as_tensor(space_mask)
    if temps is None:
        t = torch.ones(z.shape[0], 1, dtype=torch.float64)
    elif isinstance(temps, dict):
        t = torch.tensor([temps.get(int(k), 1.0) for k in type_id], dtype=torch.float64)[:, None]
    else:
        t = torch.full((z.shape[0], 1), float(temps), dtype=torch.float64)
    p = torch.softmax((z / t).masked_fill(~m, NEG_INF), dim=-1)
    return (p * m).numpy()


def write_preds_jsonl(path, items, probs):
    """{"id", "probabilities"} per line, keyed by each item's own label keys (input of vivqa_s1_v2/evaluate.py)."""
    from utils.atomic import atomic_text
    lines = []
    for it, p in zip(items, probs):
        space = label_space(it["questions"]["q"])
        lines.append(json.dumps({"id": it["id"], "probabilities": {k: float(p[j]) for j, k in enumerate(space)}}, ensure_ascii=False) + "\n")
    atomic_text(path, "".join(lines))


# --------------------------------------------------------------------------------------------- temperature
def fit_temperature(logits, space_mask, gold_idx=None, target=None, max_iter=200):
    """Temperature T in [0.1, 10] minimising the mean NLL vs `gold_idx` (primary), or the cross-entropy vs the soft
    `target` distribution; LBFGS on log T (clamped). Returns 1.0 for an empty set."""
    z = torch.as_tensor(logits, dtype=torch.float64)
    if z.shape[0] == 0:
        return 1.0
    m = torch.as_tensor(space_mask)
    u = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([u], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")
    gi = None if gold_idx is None else torch.as_tensor(gold_idx)
    tg = None if target is None else torch.as_tensor(target, dtype=torch.float64)

    def closure():
        opt.zero_grad()
        t = torch.exp(u.clamp(LOG_T_MIN, LOG_T_MAX))
        logp = torch.log_softmax((z / t).masked_fill(~m, NEG_INF), dim=-1)
        loss = -logp.gather(1, gi[:, None]).mean() if gi is not None else -(tg * logp.masked_fill(~m, 0.0)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(min(max(float(torch.exp(u.detach().clamp(LOG_T_MIN, LOG_T_MAX))), 0.1), 10.0))


def fit_temperatures(logits, space_mask, type_id, gold_idx=None, target=None):
    """One temperature per type (independent fits). -> {type_id: T}."""
    out = {}
    for t in range(3):
        r = np.nonzero(type_id == t)[0]
        out[t] = fit_temperature(logits[r], space_mask[r], None if gold_idx is None else gold_idx[r],
                                 None if target is None else target[r])
    return out


# --------------------------------------------------------------------------------------------- metrics
def correct_fraction(probs, gold_idx, tol=1e-12):
    """1/|argmax set| if the gold is in it (ties credited fractionally, as in s1bench.metrics), else 0."""
    mx = probs.max(1, keepdims=True)
    tied = probs >= mx - tol
    return tied[np.arange(len(gold_idx)), gold_idx] / tied.sum(1)


def ece(conf, correct, bins=15):
    n = len(conf)
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(correct[b == k].sum() - conf[b == k].sum()) for k in range(bins)) / max(n, 1))


# --------------------------------------------------------------------------------------------- escalation
def escalation_features(probs, space_mask, type_id):
    """[max_prob, entropy/log|space|, top1-top2 margin, one-hot type(3), max_prob * one-hot type(3)] -> (N, 9)."""
    p = np.where(space_mask, probs, 0.0)
    size = space_mask.sum(1)
    top2 = -np.sort(-p, axis=1)[:, :2]
    ent = -(np.where(p > 0, p * np.log(np.clip(p, 1e-12, None)), 0.0)).sum(1) / np.log(np.clip(size, 2, None))
    oh = np.eye(3)[type_id]
    mp = top2[:, 0]
    return np.concatenate([mp[:, None], ent[:, None], (top2[:, 0] - top2[:, 1])[:, None], oh, oh * mp[:, None]], 1)


def fit_logreg(x, y, l2=1e-3, max_iter=200):
    """Logistic regression (torch, LBFGS, L2 penalty on the weights; features are standardised with the training
    mean/std first). -> (w, b, mean, std) as numpy."""
    x = np.asarray(x, float)
    mu, sd = x.mean(0), x.std(0) + 1e-8
    xt, yt = torch.as_tensor((x - mu) / sd, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy(), mu, sd


def predict_logreg(wb, x):
    w, b, mu, sd = wb
    return 1.0 / (1.0 + np.exp(-(((np.asarray(x, float) - mu) / sd) @ w + b)))


def fold_ids(n, folds=5, seed=0):
    """Deterministic fold assignment (seeded permutation, balanced)."""
    f = np.empty(n, np.int64)
    f[np.random.RandomState(seed).permutation(n)] = np.arange(n) % folds
    return f


def oof_predict(x, y, folds=5, seed=0, l2=1e-3):
    fid = fold_ids(len(y), folds, seed)
    out = np.zeros(len(y))
    for k in range(folds):
        tr, te = fid != k, fid == k
        out[te] = predict_logreg(fit_logreg(x[tr], y[tr], l2), x[te])
    return out


def auroc(score, label):
    """Rank-based AUROC (average ranks for ties); label 1 = positive (error)."""
    score, label = np.asarray(score, float), np.asarray(label).astype(bool)
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def selective_metrics(confidence, correct, coverages=(0.5, 0.7, 0.9)):
    """Sort by confidence (desc, ties by index); AURC = mean running risk; selective accuracy at the coverages."""
    order = np.argsort(-np.asarray(confidence), kind="mergesort")
    c = np.asarray(correct, float)[order]
    risk = np.cumsum(1.0 - c) / np.arange(1, len(c) + 1)
    out = {"aurc": float(risk.mean())}
    for cov in coverages:
        k = max(1, int(round(cov * len(c))))
        out[f"sel_acc@{cov}"] = float(c[:k].mean())
    return out
