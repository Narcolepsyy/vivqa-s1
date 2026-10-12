"""Does the model USE the image (and the question)? Evaluates a probe / run checkpoint on dev three ways and prints per-type accuracy and NLL:
    normal | image-shuffled (the SigLIP features of the items of a batch are permuted, questions untouched) | question-shuffled (the text side is permuted, image untouched)
plus the size of the two fusion summands vision_proj(pooled vision) and text_proj(pooled text) that are added before the final LayerNorm of h.
    python scripts/viso_vision_check.py --ckpt ~/data/viso_diag/ckpt_viso_siglip2_<hash>.pth [--device cpu] [--limit 1500]
If accuracy / NLL barely move when the images are shuffled, the model ignores the vision features."""
import argparse
import copy
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train_viso import load_config  # noqa: E402
from builders.task_builder import build_task  # noqa: E402
from data_utils.datasets.s1_dataset import s1_collate  # noqa: E402
from tasks.typed_decision_task import preds_dict  # noqa: E402
from utils import viso_calibration as calib  # noqa: E402
from utils.viso_s1 import label_space  # noqa: E402
from s1bench.metrics import evaluate as s1_evaluate  # noqa: E402

TYPES = ("noul", "score", "choice")


def shuffled(batch, what, gen):
    b = copy.copy(batch)
    perm = torch.randperm(len(batch.id), generator=gen)
    if what == "image":
        for k in ("region_features", "region_mask", "region_boxes"):
            if k in b:
                b[k] = batch[k][perm]
    elif what == "question":
        for k in ("question_tokens", "question_states", "question_state_mask"):
            if k in b:
                b[k] = batch[k][perm]
    return b


@torch.no_grad()
def run(task, what, limit, seed=0):
    ds = task.datasets["dev"]
    items = ds.items[:limit] if limit else ds.items
    ds.items = items
    task.model.eval()
    gen = torch.Generator().manual_seed(seed)
    logits, masks, ids, vnorm, tnorm, vstd, tstd = [], [], [], [], [], [], []
    enc = task.model.encoder
    for batch in DataLoader(ds, batch_size=64, shuffle=False, collate_fn=s1_collate):
        batch.to(task.device)
        b = shuffled(batch, what, gen) if what != "normal" else batch
        logits.append(task.model(b)["logits"].float().cpu().numpy()); masks.append(batch.space_mask.cpu().numpy()); ids.append(batch.id.cpu().numpy())
    lg = np.concatenate(logits)
    probs = calib.probs_from_logits(lg, np.concatenate(masks))
    ev = s1_evaluate(items, preds_dict(items, probs))["primary"]
    return {t: (round(ev[t]["accuracy"], 3), round(ev[t]["nll"], 3)) for t in TYPES if t in ev}


@torch.no_grad()
def per_item(task, what, limit, seed):
    """per dev item: (type id, correct 0/1, -log p(gold label)) of one pass (what = normal | image | question)"""
    ds = task.datasets["dev"]
    items = ds.items[:limit] if limit else ds.items
    ds.items = items
    task.model.eval()
    gen = torch.Generator().manual_seed(seed)
    lg, mk, ty = [], [], []
    for batch in DataLoader(ds, batch_size=64, shuffle=False, collate_fn=s1_collate):
        batch.to(task.device)
        b = shuffled(batch, what, gen) if what != "normal" else batch
        lg.append(task.model(b)["logits"].float().cpu().numpy()); mk.append(batch.space_mask.cpu().numpy()); ty.append(batch.type_id.cpu().numpy())
    probs = calib.probs_from_logits(np.concatenate(lg), np.concatenate(mk))
    gold = np.array([label_space(it["questions"]["q"]).index(it["gold_label"]) for it in items])
    pg = probs[np.arange(len(items)), gold]
    correct = (np.where(np.concatenate(mk), probs, -1).argmax(1) == gold).astype(float)
    return np.concatenate(ty), correct, -np.log(np.clip(pg, 1e-9, None))


def shuffle_drop_ci(task, what, limit, n_shuffles=5, n_boot=2000):
    """paired bootstrap over dev items of (normal - shuffled): accuracy drop and NLL increase, per type. The shuffled value of an item is its mean over n_shuffles different shuffles."""
    ty, c0, l0 = per_item(task, "normal", limit, 0)
    runs = [per_item(task, what, limit, 100 + k) for k in range(n_shuffles)]
    c1, l1 = np.mean([r[1] for r in runs], 0), np.mean([r[2] for r in runs], 0)
    rng = np.random.default_rng(0)
    out = {}
    for t, name in enumerate(TYPES):
        idx = np.where(ty == t)[0]
        if not len(idx):
            continue
        dacc, dnll = c0[idx] - c1[idx], l1[idx] - l0[idx]                      # positive = the model is worse when the input is shuffled
        boots = [rng.integers(0, len(idx), len(idx)) for _ in range(n_boot)]
        ci = lambda d: tuple(round(float(x), 4) for x in np.percentile([d[b].mean() for b in boots], [2.5, 97.5]))
        out[name] = {"n": int(len(idx)), "acc_drop": round(float(dacc.mean()), 4), "acc_drop_ci95": ci(dacc), "nll_increase": round(float(dnll.mean()), 4), "nll_increase_ci95": ci(dnll)}
    return out


from utils.viso_s1 import yesno_form  # noqa: E402,F401  (the shared definition of the three yes/no question forms)


def forms_report(task, limit, n_shuffles=5, n_boot=2000):
    """dev yes/no accuracy / NLL by question form and the image-shuffle drop with a paired bootstrap CI for each form"""
    ds = task.datasets["dev"]
    ty, c0, l0 = per_item(task, "normal", limit, 0)
    runs = [per_item(task, "image", limit, 100 + k) for k in range(n_shuffles)]
    c1, l1 = np.mean([r[1] for r in runs], 0), np.mean([r[2] for r in runs], 0)
    forms = np.array([yesno_form(it) for it in ds.items[:limit] if True] if limit else [yesno_form(it) for it in ds.items])
    rng = np.random.default_rng(0)
    out = {}
    for f in ("existence", "attribute", "other", "all"):
        idx = np.where((ty == 0) & ((forms == f) if f != "all" else True))[0]
        if not len(idx):
            continue
        dacc, dnll = c0[idx] - c1[idx], l1[idx] - l0[idx]
        boots = [rng.integers(0, len(idx), len(idx)) for _ in range(n_boot)]
        ci = lambda d: tuple(round(float(x), 4) for x in np.percentile([d[b].mean() for b in boots], [2.5, 97.5]))
        out[f] = {"n": int(len(idx)), "acc": round(float(c0[idx].mean()), 4), "nll": round(float(l0[idx].mean()), 4), "image_shuffle_acc_drop": round(float(dacc.mean()), 4), "acc_drop_ci95": ci(dacc),
                  "image_shuffle_nll_increase": round(float(dnll.mean()), 4), "nll_increase_ci95": ci(dnll)}
    return out


@torch.no_grad()
def fusion_norms(task, n=512):
    """mean norm and per-dim std over items of the two summands of h (hook on the final LayerNorm input)"""
    enc, store = task.model.encoder, {}
    if not (hasattr(enc, "vision_proj") and hasattr(enc, "text_proj")):                 # B-BUTD has no sum fusion of two projections: nothing to report
        return {"note": f"{type(enc).__name__} has no vision_proj + text_proj fusion"}
    h1 = enc.vision_proj.register_forward_hook(lambda m, i, o: store.__setitem__("v", o.detach()))
    h2 = enc.text_proj.register_forward_hook(lambda m, i, o: store.__setitem__("t", o.detach()))
    ds = task.datasets["dev"]
    batch = s1_collate([ds[i] for i in range(0, len(ds), max(1, len(ds) // n))][:n]).to(task.device)
    task.model.eval()
    task.model(batch)
    h1.remove(); h2.remove()
    v, t = store["v"], store["t"]
    return {"vision_proj: mean norm / std across items": (round(float(v.norm(dim=-1).mean()), 3), round(float(v.std(0).mean()), 4)),
            "text_proj:   mean norm / std across items": (round(float(t.norm(dim=-1).mean()), 3), round(float(t.std(0).mean()), 4))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N dev items")
    ap.add_argument("--packed", default=None, help="override DATASET.FEATURE_DATASET.PACKED_PATH (a checkpoint trained on Kaggle remembers the Kaggle paths)")
    ap.add_argument("--s1-data", default=None, help="override DATASET.VOCAB.S1_DIR (the data/ directory of the benchmark)")
    ap.add_argument("--text-cache", default=None, help="override DATASET.FEATURE_DATASET.TEXT_CACHE")
    ap.add_argument("--no-forms", action="store_true", help="skip the yes/no question-form report")
    ap.add_argument("--dump-peritem", default=None, help="write the per-item dev results (id, type, correct, NLL, yes/no form) of the NORMAL pass to this npz (paired comparisons between runs)")
    ap.add_argument("--only-peritem", action="store_true", help="with --dump-peritem: write the per-item results and stop (no shuffle evaluations)")
    ap.add_argument("--no-ci", action="store_true", help="skip the paired-bootstrap CI of the shuffle drops")
    a = ap.parse_args(argv)
    ck = os.path.expanduser(a.ckpt)
    side = json.load(open(ck[:-4] + ".json")) if os.path.isfile(ck[:-4] + ".json") else None
    cfg_file = side["config_file"] if side else os.path.join(os.path.dirname(ck), "run_config.yaml")
    if not os.path.isfile(cfg_file):                                          # trained on Kaggle: the config lives in this repo under the same relative path
        cfg_file = os.path.join(REPO, "configs", os.path.basename(cfg_file))
    over = []
    for key, val in (("DATASET.FEATURE_DATASET.PACKED_PATH", a.packed), ("DATASET.VOCAB.S1_DIR", a.s1_data), ("DATASET.FEATURE_DATASET.TEXT_CACHE", a.text_cache)):
        if val:
            over += [key, os.path.expanduser(val)]
    if over:
        over += ["DATASET.FEATURE_DATASET.WORKERS", "0"]
    opts = (side["opts"] if side else []) + over + ["MODEL.NAME", "vcheck", "TRAINING.OUTPUT_PATH", "/tmp/viso_vcheck", "TRAINING.CHECKPOINT_PATH", "/tmp/viso_vcheck", "MODEL.DEVICE", a.device]
    task = build_task(load_config(cfg_file, opts))
    task.model.load_state_dict(torch.load(ck, map_location=a.device, weights_only=True)["state_dict"])
    if a.dump_peritem:
        ty, c0, l0 = per_item(task, "normal", a.limit, 0)
        items = task.datasets["dev"].items
        forms = np.array([yesno_form(it) if it["questions"]["q"]["type"] == "noul" else "" for it in items[:len(ty)]])
        np.savez(a.dump_peritem, ids=np.array([it["id"] for it in items[:len(ty)]]), type_id=ty, correct=c0, nll=l0, form=forms)
    if a.dump_peritem and a.only_peritem:
        print("per-item dump written:", a.dump_peritem)
        return
    print("checkpoint:", ck, "| dev items:", a.limit or len(task.datasets["dev"].items))
    print("fusion summands:", fusion_norms(task))
    for what in ("normal", "image", "question"):
        print(f"dev ({what + '-shuffled' if what != 'normal' else 'normal':16s}) (acc, nll) per type:", run(task, what, a.limit), flush=True)
    if not a.no_forms:
        print("yes/no by question form (dev; image-shuffle drop = normal - shuffled, paired bootstrap 95% CI):")
        for f, r in forms_report(task, a.limit).items():
            print(f"   {f:9s} n={r['n']:5d}  acc {r['acc']:.3f}  NLL {r['nll']:.3f} | acc drop {r['image_shuffle_acc_drop']:+.4f} {r['acc_drop_ci95']} | NLL increase {r['image_shuffle_nll_increase']:+.4f} {r['nll_increase_ci95']}", flush=True)
    if not a.no_ci:
        for what in ("image", "question"):
            print(f"{what}-shuffle drop (normal - shuffled; mean of 5 shuffles per item; paired bootstrap 95% CI over dev items):")
            for name, r in shuffle_drop_ci(task, what, a.limit).items():
                print(f"   {name:7s} n={r['n']:5d}  acc drop {r['acc_drop']:+.4f} {r['acc_drop_ci95']}  |  NLL increase {r['nll_increase']:+.4f} {r['nll_increase_ci95']}", flush=True)


if __name__ == "__main__":
    main()
