"""Does the frozen fused representation h of a trained model contain ANY yes/no information? Extract h for the yes/no items of train and dev (eval mode), z-score it with the train statistics and fit a
logistic regression (L2, LBFGS) on train; report dev accuracy / NLL for several L2 strengths (chance: acc 0.5, NLL 0.693).
    python scripts/viso_h_probe.py --ckpt <probe checkpoint .pth> [--device cuda] [--packed ... --s1-data ... --text-cache ...] [--limit-train N]
If even this probe is at chance, the representation carries no usable yes/no signal (the problem is upstream of the head)."""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train_viso import load_config  # noqa: E402
from builders.task_builder import build_task  # noqa: E402
from data_utils.datasets.s1_dataset import s1_collate  # noqa: E402


@torch.no_grad()
def features(task, split, limit=0):
    ds = task.datasets[split]
    ds.items = [it for it in ds.items if it["questions"]["q"]["type"] == "noul"]
    ds.items = ds.items[:limit] if limit else ds.items
    ds.index = {it["id"]: i for i, it in enumerate(ds.items)}
    task.model.eval()
    H, Y = [], []
    for batch in DataLoader(ds, batch_size=64, shuffle=False, collate_fn=s1_collate):
        batch.to(task.device)
        H.append(task.model.encoder(batch)[0].float().cpu())
        Y.append((batch.gold_idx == 1).long().cpu())                        # gold 'true' = column 1 of the noul space [false, true]
    return torch.cat(H), torch.cat(Y)


def fit(Xtr, ytr, Xdv, ydv, l2):
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr.float()) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        lg = Xdv @ w + b
        nll = float(torch.nn.functional.binary_cross_entropy_with_logits(lg, ydv.float()))
        tr_nll = float(torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr.float()))
        return round(float(((lg > 0).long() == ydv).float().mean()), 4), round(nll, 4), round(tr_nll, 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--packed", default=None)
    ap.add_argument("--s1-data", default=None)
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the table as json")
    a = ap.parse_args(argv)
    ck = os.path.expanduser(a.ckpt)
    side = json.load(open(ck[:-4] + ".json"))
    cfg_file = side["config_file"] if os.path.isfile(side["config_file"]) else os.path.join(REPO, "configs", os.path.basename(side["config_file"]))
    over = []
    for key, val in (("DATASET.FEATURE_DATASET.PACKED_PATH", a.packed), ("DATASET.VOCAB.S1_DIR", a.s1_data), ("DATASET.FEATURE_DATASET.TEXT_CACHE", a.text_cache)):
        if val:
            over += [key, os.path.expanduser(val)]
    if over:
        over += ["DATASET.FEATURE_DATASET.WORKERS", "0"]
    opts = side["opts"] + over + ["MODEL.NAME", "hprobe", "TRAINING.OUTPUT_PATH", "/tmp/viso_hprobe", "TRAINING.CHECKPOINT_PATH", "/tmp/viso_hprobe", "MODEL.DEVICE", a.device]
    task = build_task(load_config(cfg_file, opts))
    task.model.load_state_dict(torch.load(ck, map_location=a.device, weights_only=True)["state_dict"])
    Htr, ytr = features(task, "train", a.limit_train)
    Hdv, ydv = features(task, "dev")
    mu, sd = Htr.mean(0), Htr.std(0).clamp(min=1e-6)
    Xtr, Xdv = (Htr - mu) / sd, (Hdv - mu) / sd
    print(f"checkpoint {ck}: yes/no items train {len(ytr)} dev {len(ydv)} | dev 'true' share {float(ydv.float().mean()):.3f} (majority-class acc {max(float(ydv.float().mean()), 1 - float(ydv.float().mean())):.3f})")
    table = {}
    for l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        acc, nll, tr = fit(Xtr, ytr, Xdv, ydv, l2)
        table[str(l2)] = {"dev_acc": acc, "dev_nll": nll, "train_nll": tr}
        print(f"   L2 {l2:g}: dev acc {acc:.4f}  dev NLL {nll:.4f}  (train NLL {tr:.4f}; chance NLL 0.6931)", flush=True)
    if a.out:
        json.dump(table, open(a.out, "w"), indent=1)
    return table


if __name__ == "__main__":
    main()
