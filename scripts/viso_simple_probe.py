"""Is the data learnable with the cached features? Tiny question-bag-of-words x SigLIP-2-image model on the typed targets (soft CE), no ViSO code involved.
    python scripts/viso_simple_probe.py [--no-image] [--grid]      # per-type dev accuracy after each epoch
The answer for each type is a linear layer over a fused vector; choice items score candidates by (fused . candidate-word-bag embedding)."""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from utils.viso_packed_features import PackedFeatures  # noqa: E402
from utils.viso_s1 import MAX_SPACE, label_space, read_jsonl  # noqa: E402
from data_utils.utils import preprocess_sentence  # noqa: E402

TYPES = ("noul", "score", "choice")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default=os.path.join(REPO, "..", "vivqa_s1_v2", "data"))
    ap.add_argument("--packed", default=os.path.expanduser("~/data/siglip2_packed"))
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--fusion", choices=("product", "sum"), default="product", help="how the question and image vectors are combined")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    data = {sp: read_jsonl(os.path.join(a.s1, f"{sp}.jsonl")) for sp in ("train", "dev")}
    words = {}
    for sp in ("train",):
        for it in data[sp]:
            for w in preprocess_sentence(it["questions"]["q"]["instructions"], None):
                words.setdefault(w, len(words))
            if it["questions"]["q"]["type"] == "choice":
                for c in it["questions"]["q"]["criteria"]:
                    for w in preprocess_sentence(c, None):
                        words.setdefault(w, len(words))
    pk = PackedFeatures(a.packed)
    img = {}
    def image(cid):
        if cid not in img:
            img[cid] = torch.from_numpy(pk.get_array(cid)[0].copy())          # pooled token (768)
        return img[cid]
    NW = len(words) + 1
    def bag(text):
        v = torch.zeros(NW)
        for w in preprocess_sentence(text, None):
            v[words.get(w, NW - 1)] = 1
        return v
    def encode(items):
        q = torch.stack([bag(it["questions"]["q"]["instructions"]) for it in items])
        im = torch.stack([image(it["coco_id"]) for it in items])
        ty = torch.tensor([TYPES.index(it["questions"]["q"]["type"]) for it in items])
        tgt = torch.zeros(len(items), MAX_SPACE); msk = torch.zeros(len(items), MAX_SPACE, dtype=torch.bool)
        cand = torch.full((len(items), MAX_SPACE, 6), NW, dtype=torch.long)          # word ids of each candidate (NW = padding)
        gold = torch.zeros(len(items), dtype=torch.long)
        for i, it in enumerate(items):
            sp = label_space(it["questions"]["q"]); pr = it["gold"]["q"]["probabilities"]
            for k, key in enumerate(sp):
                tgt[i, k] = pr.get(key, 0.0); msk[i, k] = True
                if it["questions"]["q"]["type"] == "choice":
                    ids = [words.get(w, NW - 1) for w in preprocess_sentence(key, None)][:6]
                    cand[i, k, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            gold[i] = sp.index(it["gold_label"])
        return q, im, ty, tgt, msk, cand, gold
    tr, dv = encode(data["train"]), encode(data["dev"])
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.q = nn.Sequential(nn.Linear(NW, 512), nn.ReLU()); s.i = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 512), nn.ReLU())
            s.mix = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3)); s.heads = nn.ModuleList([nn.Linear(512, MAX_SPACE) for _ in TYPES])
            s.ce = nn.Embedding(NW + 1, 512, padding_idx=NW)
        def forward(s, q, im, ty, cand):
            f = s.q(q) if a.no_image else (s.q(q) * s.i(im) if a.fusion == "product" else s.q(q) + s.i(im))
            f = s.mix(f)
            out = torch.stack([h(f) for h in s.heads], 1)[torch.arange(len(ty)), ty]
            ch = torch.einsum("bd,bkd->bk", f, s.ce(cand).sum(2) / (cand != NW).sum(2, keepdim=True).clamp(min=1))
            return torch.where((ty == 2)[:, None], ch, out)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.05)
    n = len(tr[0])
    for ep in range(1, a.epochs + 1):
        net.train(); perm = torch.randperm(n)
        for b in perm.split(128):
            q, im, ty, tgt, msk, cand, gold = (x[b] for x in tr)
            lg = net(q, im, ty, cand).masked_fill(~msk, -1e9)
            loss = -(tgt * torch.log_softmax(lg, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            q, im, ty, tgt, msk, cand, gold = dv
            pred = net(q, im, ty, cand).masked_fill(~msk, -1e9).argmax(-1)
        print(f"epoch {ep:2d} dev acc " + str({t: round(float((pred[ty == k] == gold[ty == k]).float().mean()), 3) for k, t in enumerate(TYPES)}), flush=True)


if __name__ == "__main__":
    main()
