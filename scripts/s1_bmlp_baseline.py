"""B-MLP: the honest simple-model floor of the sanity group. Question word bag (sum of learned word embeddings) x pooled SigLIP 2 image vector (token 0 of the cache) -> small MLP with one
head per type; choice candidates are scored by <fused vector, mean of the candidate's word embeddings>. No pre-training, no attention, no constituents, CPU only.
    python scripts/s1_bmlp_baseline.py [--seeds 0 1 2] [--epochs 25]
The epoch is selected by dev soft-CE (never by test). Train words only for the vocabulary. Predictions: ~/data/s1_sanity/preds/bmlp_s<seed>_<split>.jsonl; metrics through the benchmark's own
evaluate.py on All / Core / Image-unseen (test) -> ~/data/s1_sanity/bmlp_results.{json,md}. Uncalibrated (T = 1), like ViSO's A2."""
import argparse
import importlib.util
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("VIVQA_S1_ROOT", os.path.abspath(os.path.join(REPO, "..", "vivqa_s1_v2")))   # s1bench of the v2 benchmark (utils.viso_s1 reads it)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from data_utils.utils import preprocess_sentence  # noqa: E402
from utils.viso_packed_features import PackedFeatures  # noqa: E402
from utils.viso_s1 import MAX_SPACE, label_space, read_jsonl  # noqa: E402

TYPES = ("noul", "score", "choice")
LQ, LC = 32, 6
LANG = "vi"                                    # --lang en: the question text is the ENGLISH VQA v2 question (meta.en_q); candidates stay as in the benchmark


def qtext(it):
    return (it["meta"].get("en_q") or "") if LANG == "en" else it["questions"]["q"]["instructions"]


def load_eval(root):
    spec = importlib.util.spec_from_file_location("s1_evaluate", os.path.join(root, "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Data:
    def __init__(self, items, words, pk, img_cache):
        n, nw = len(items), len(words)
        self.q = torch.full((n, LQ), nw, dtype=torch.long)
        self.cand = torch.full((n, MAX_SPACE, LC), nw, dtype=torch.long)
        self.ty = torch.tensor([TYPES.index(it["questions"]["q"]["type"]) for it in items])
        self.tgt = torch.zeros(n, MAX_SPACE)
        self.mask = torch.zeros(n, MAX_SPACE, dtype=torch.bool)
        self.img = torch.stack([img_cache(it["coco_id"], pk) for it in items])
        self.gold = torch.zeros(n, dtype=torch.long)
        for i, it in enumerate(items):
            q = it["questions"]["q"]
            toks = [words.get(w, nw - 1) for w in preprocess_sentence(qtext(it), None)][:LQ]
            self.q[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
            sp = label_space(q)
            pr = it["gold"]["q"]["probabilities"]
            for k, key in enumerate(sp):
                self.tgt[i, k] = pr.get(key, 0.0)
                self.mask[i, k] = True
                if q["type"] == "choice":
                    ids = [words.get(w, nw - 1) for w in preprocess_sentence(key, None)][:LC]
                    self.cand[i, k, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            self.gold[i] = sp.index(it["gold_label"])


class Net(nn.Module):
    def __init__(self, nw, d=512):
        super().__init__()
        self.nw = nw
        self.qe = nn.EmbeddingBag(nw + 1, d, mode="sum", padding_idx=nw)
        self.q = nn.ReLU()
        self.i = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.3))
        self.heads = nn.ModuleList([nn.Linear(d, MAX_SPACE) for _ in TYPES])
        self.ce = nn.Embedding(nw + 1, d, padding_idx=nw)

    def forward(self, q, img, ty, cand):
        f = self.mix(self.q(self.qe(q)) * self.i(img))
        out = torch.stack([h(f) for h in self.heads], 1)[torch.arange(len(ty)), ty]
        cm = (cand != self.nw)
        cv = (self.ce(cand) * cm.unsqueeze(-1)).sum(2) / cm.sum(2, keepdim=True).clamp(min=1)
        return torch.where((ty == 2)[:, None], torch.einsum("bd,bkd->bk", f, cv), out)


def probs_of(net, d, bs=1024):
    net.eval()
    out = []
    with torch.no_grad():
        for b in torch.arange(len(d.ty)).split(bs):
            lg = net(d.q[b], d.img[b], d.ty[b], d.cand[b]).masked_fill(~d.mask[b], -1e9)
            out.append(torch.softmax(lg, -1))
    return torch.cat(out)


def soft_ce(p, d):
    return float(-(d.tgt * torch.log(p.clamp(min=1e-9))).sum(-1).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-root", default=os.path.join(REPO, "..", "vivqa_s1_v2"))
    ap.add_argument("--packed", default=os.path.expanduser("~/data/siglip2_packed"))
    ap.add_argument("--out", default="~/data/s1_sanity")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lang", choices=("vi", "en"), default="vi", help="en: use the English source questions (meta.en_q) instead of the Vietnamese ones (benchmark-property check)")
    ap.add_argument("--limit", type=int, default=0, help="first N items of each split (tests)")
    a = ap.parse_args(argv)
    global LANG
    LANG = a.lang
    tag = "bmlp" if a.lang == "vi" else "bmlp_en"
    root, out = os.path.abspath(a.s1_root), os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    sys.path.insert(0, root)
    EV = load_eval(root)
    data_dir = os.path.join(root, "data")
    items = {sp: read_jsonl(os.path.join(data_dir, f"{sp}.jsonl")) for sp in ("train", "dev", "test")}
    if a.limit:
        items = {k: v[:a.limit] for k, v in items.items()}
    words = {}
    for it in items["train"]:
        for w in preprocess_sentence(qtext(it), None):
            words.setdefault(w, len(words))
        if it["questions"]["q"]["type"] == "choice":
            for c in it["questions"]["q"]["criteria"]:
                for w in preprocess_sentence(c, None):
                    words.setdefault(w, len(words))
    words["<unk>"] = len(words)
    nw = len(words)
    pk = PackedFeatures(a.packed)
    cache = {}
    def img(cid, pk_):
        if cid not in cache:
            cache[cid] = torch.from_numpy(pk_.get_array(cid)[0].copy())
        return cache[cid]
    D = {sp: Data(items[sp], words, pk, img) for sp in items}
    flags = os.path.join(data_dir, "eval_flags.jsonl")
    results = {}
    for seed in a.seeds:
        torch.manual_seed(seed)
        net = Net(nw)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.05)
        best, best_state, best_ep = 1e9, None, 0
        for ep in range(1, a.epochs + 1):
            net.train()
            for b in torch.randperm(len(D["train"].ty)).split(128):
                d = D["train"]
                lg = net(d.q[b], d.img[b], d.ty[b], d.cand[b]).masked_fill(~d.mask[b], -1e9)
                loss = -(d.tgt[b] * torch.log_softmax(lg, -1)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            v = soft_ce(probs_of(net, D["dev"]), D["dev"])
            if v < best:
                best, best_ep, best_state = v, ep, {k: t.clone() for k, t in net.state_dict().items()}
        net.load_state_dict(best_state)
        res = {"best_epoch": best_ep, "dev_soft_ce": round(best, 4)}
        for sp in ("dev", "test"):
            pr = probs_of(net, D[sp]).numpy()
            preds = {it["id"]: {"id": it["id"], "probabilities": {k: float(pr[i][j]) for j, k in enumerate(label_space(it["questions"]["q"]))}} for i, it in enumerate(items[sp])}
            with open(os.path.join(out, "preds", f"{tag}_s{seed}_{sp}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                for it in items[sp]:
                    f.write(json.dumps(preds[it["id"]], ensure_ascii=False) + "\n")
            res[sp] = {"all": EV.evaluate(items[sp], preds)}
            if sp == "test" and os.path.exists(flags) and not a.limit:
                _, views = EV.view_blocks(items[sp], preds, flags, f"{sp}.jsonl")
                res[sp].update(views)
        results[f"seed{seed}"] = res
        t = res["test"]["all"]["primary"]
        print(f"seed {seed}: best epoch {best_ep}, dev soft-CE {best:.4f} | test acc per type " + ", ".join(f"{k} {t[k]['accuracy']:.3f}" for k in TYPES if k in t) + f" | all {t['all']['accuracy']:.3f}", flush=True)
    json.dump(results, open(os.path.join(out, f"{tag}_results.json"), "w"), indent=1, ensure_ascii=False)
    L = [f"# B-MLP ({'English' if a.lang == 'en' else 'Vietnamese'} question word bag x pooled SigLIP 2 image vector): the honest simple-model floor", "", f"benchmark: {root} | seeds {a.seeds} | epoch chosen by dev soft-CE | T = 1 (uncalibrated)", "",
         "| split | view | metric | " + " | ".join(f"seed {s}" for s in a.seeds) + " | mean |", "|---|---|---|" + "---|" * (len(a.seeds) + 1)]
    for sp in ("dev", "test"):
        for view, getter in (("All", lambda r: r[sp]["all"]), ("Core", lambda r: r[sp].get("core") or {}), ("Image-unseen", lambda r: r[sp].get("image_unseen") or {})):
            for typ in ("all",) + TYPES:
                vals = []
                for s in a.seeds:
                    p = getter(results[f"seed{s}"]).get("primary", {}).get(typ)
                    vals.append(None if p is None else p["accuracy"])
                if all(v is None for v in vals):
                    continue
                ok = [v for v in vals if v is not None]
                L.append(f"| {sp} | {view} | acc {typ} | " + " | ".join("-" if v is None else f"{v:.4f}" for v in vals) + f" | {np.mean(ok):.4f} |")
    open(os.path.join(out, f"{tag}_results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:14]))
    return results


if __name__ == "__main__":
    main()
