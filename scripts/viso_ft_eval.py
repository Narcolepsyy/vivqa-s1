"""ViSO-FT analysis on a finished run (best_model.pth): (1) the paired bootstrap of the FT predictions against reference runs (dev and test), (2) the Y4-style image-shuffle check.
  python scripts/viso_ft_eval.py --config-file <run>/config.yaml --refs ABL_bami_s0=<dir with preds_test.jsonl> A2_s0=<dir> --out <run>/ft_report
(1) For every reference: accuracy / NLL / Brier / ECE / AURC differences FT - reference with 95 % paired-bootstrap CIs (items resampled together), on dev and test (the test split is still unfrozen).
(2) The image-shuffle check: the model is evaluated normally and with the images of each batch permuted (questions untouched; the live vision tower runs on the permuted images). Per type and for yes/no by
    question form (existence / attribute / other): accuracy and NLL normal vs shuffled, the drop with a paired-bootstrap CI. If accuracy barely moves, the model does not use the image.
Outputs: ft_report.json / ft_report.md (+ per-item arrays peritem_<split>.npz: correct / nll normal and shuffled)."""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data_utils.datasets.s1_dataset import s1_collate  # noqa: E402
from tasks.typed_decision_task import loader_kwargs, preds_dict  # noqa: E402
from utils import viso_calibration as calib  # noqa: E402
from utils.viso_s1 import read_jsonl, yesno_form  # noqa: E402
from s1bench import metrics as M  # noqa: E402

import viso_bootstrap as B  # noqa: E402

TYPES = ("noul", "score", "choice")


@torch.no_grad()
def logits_for(model, ds, batch_size, device, workers, shuffle_images=False, seed=0):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=s1_collate, **loader_kwargs(workers))
    gen = torch.Generator().manual_seed(seed)
    out = []
    for batch in loader:
        if shuffle_images:
            perm = torch.randperm(len(batch.id), generator=gen)
            batch.image_u8 = batch.image_u8[perm]
        batch.to(device)
        out.append(model(batch)["logits"].float().cpu().numpy())
    return np.concatenate(out)


def scores_of(ds, logits):
    mask = np.stack([_space_mask(it) for it in ds.items])
    probs = calib.probs_from_logits(logits, mask)
    return B.scores_for(ds.items, preds_dict(ds.items, probs))


def _space_mask(it):
    from utils.viso_s1 import MAX_SPACE, label_space
    m = np.zeros(MAX_SPACE, bool)
    m[:len(label_space(it["questions"]["q"]))] = True
    return m


def boot_diff(a, b, idx, key):
    """paired bootstrap of mean(a) - mean(b) for per-item arrays a, b"""
    d = (np.asarray(a, float) - np.asarray(b, float))
    dist = d[idx].mean(1)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)]}


def shuffle_check(task, split, resamples=1000, seed=0):
    ds = task.datasets[split]
    fd = task.config.DATASET.FEATURE_DATASET
    w = int(getattr(fd, "EVAL_WORKERS", 0))
    sc_n = scores_of(ds, logits_for(task.model, ds, task.batch_size, task.device, w))
    sc_s = scores_of(ds, logits_for(task.model, ds, task.batch_size, task.device, w, shuffle_images=True, seed=seed))
    idx_all = B.resample_indices(len(ds.items), resamples, seed)
    groups = {"all": np.arange(len(ds.items))}
    for t in TYPES:
        groups[t] = np.array([i for i, s in enumerate(sc_n) if s["type"] == t])
    for form in ("existence", "attribute", "other"):
        groups[f"yes/no:{form}"] = np.array([i for i, (it, s) in enumerate(zip(ds.items, sc_n)) if s["type"] == "noul" and yesno_form(it) == form])
    res = {"n_items": len(ds.items), "groups": {}}
    for name, ix in groups.items():
        if not len(ix):
            continue
        sub_idx = B.resample_indices(len(ix), resamples, seed)
        cn = np.array([sc_n[i]["correct"] for i in ix]); cs = np.array([sc_s[i]["correct"] for i in ix])
        nn_ = np.array([sc_n[i]["nll"] for i in ix]); ns = np.array([sc_s[i]["nll"] for i in ix])
        res["groups"][name] = {"n": int(len(ix)), "acc_normal": float(cn.mean()), "acc_shuffled": float(cs.mean()), "acc_drop": boot_diff(cn, cs, sub_idx, "accuracy"),
                               "nll_normal": float(nn_.mean()), "nll_shuffled": float(ns.mean()), "nll_increase": boot_diff(ns, nn_, sub_idx, "nll")}
    np.savez(os.path.join(task.output_path, "ft_report", f"peritem_{split}.npz"), ids=np.array([it["id"] for it in ds.items]), correct_normal=np.array([s["correct"] for s in sc_n]),
             correct_shuffled=np.array([s["correct"] for s in sc_s]), nll_normal=np.array([s["nll"] for s in sc_n]), nll_shuffled=np.array([s["nll"] for s in sc_s]))
    return res


def paired_vs_refs(items, ft_preds, refs, split, resamples=1000, seed=0):
    sc_ft = B.scores_for(items, ft_preds)
    idx = B.resample_indices(len(items), resamples, seed)
    bt = B.boot_primary(sc_ft, idx)
    pt = B.point_metrics(sc_ft)
    out = {"ft_point": {k: pt[k] for k in B.PRIMARY}}
    for name, path in refs.items():
        f = os.path.join(os.path.expanduser(path), f"preds_{split}.jsonl")
        if not os.path.isfile(f):
            out[name] = {"error": f"{f} not found"}
            continue
        rp = {r["id"]: r for r in read_jsonl(f)}
        sc_r = B.scores_for(items, rp)
        br, pr = B.boot_primary(sc_r, idx), B.point_metrics(sc_r)
        out[name] = {"ref_point": {k: pr[k] for k in B.PRIMARY}, "paired_ft_minus_ref": B.paired(bt, br, pt, pr, B.PRIMARY)}
    return out


def to_md(rep):
    L = ["# ViSO-FT report", ""]
    for split, d in rep.get("paired", {}).items():
        L += [f"## Paired bootstrap, FT - reference ({split}, n = {rep['n'][split]})", "", f"FT: " + ", ".join(f"{k} {v:.4f}" for k, v in d["ft_point"].items()), "",
              "| reference | metric | FT - ref | 95% CI | verdict |", "|---|---|---|---|---|"]
        for name, r in d.items():
            if name == "ft_point":
                continue
            if "error" in r:
                L.append(f"| {name} | | | | {r['error']} |")
                continue
            for k, v in r["paired_ft_minus_ref"].items():
                lo, hi = v["ci95"]
                better = (v["diff"] < 0) if k in ("nll", "brier", "ece", "aurc") else (v["diff"] > 0)
                verdict = ("FT better" if better else "FT worse") + ", CI excludes 0" if (lo > 0 or hi < 0) else "n.s."
                L.append(f"| {name} | {k} | {v['diff']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {verdict} |")
        L.append("")
    for split, d in rep.get("shuffle", {}).items():
        L += [f"## Image-shuffle check ({split}): images permuted inside each batch, questions untouched", "", "| group | n | acc normal | acc shuffled | drop (CI) | NLL normal | NLL shuffled | increase (CI) |", "|---|---|---|---|---|---|---|---|"]
        for name, g in d["groups"].items():
            a, n = g["acc_drop"], g["nll_increase"]
            L.append(f"| {name} | {g['n']} | {g['acc_normal']:.3f} | {g['acc_shuffled']:.3f} | {a['diff']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {g['nll_normal']:.3f} | {g['nll_shuffled']:.3f} | "
                     f"{n['diff']:+.3f} [{n['ci95'][0]:+.3f}, {n['ci95'][1]:+.3f}] |")
        L.append("")
    return "\n".join(L) + "\n"


def run(task, refs, resamples=1000, splits=("dev", "test"), shuffle_splits=("dev",)):
    out_dir = os.path.join(task.output_path, "ft_report")
    os.makedirs(out_dir, exist_ok=True)
    task.load_checkpoint("best_model.pth")
    rep = {"paired": {}, "shuffle": {}, "n": {}}
    for split in splits:
        ds = task.datasets[split]
        pf = os.path.join(task.output_path, f"preds_{split}.jsonl")
        if not os.path.isfile(pf):
            lg = np.load(os.path.join(task.output_path, f"logits_{split}.npz"))
            calib.write_preds_jsonl(pf, ds.items, calib.probs_from_logits(lg["logits"], lg["space_mask"]))
        ft_preds = {r["id"]: r for r in read_jsonl(pf)}
        rep["paired"][split] = paired_vs_refs(ds.items, ft_preds, refs, split, resamples)
        rep["n"][split] = len(ds.items)
    for split in shuffle_splits:
        rep["shuffle"][split] = shuffle_check(task, split, resamples)
    json.dump(rep, open(os.path.join(out_dir, "ft_report.json"), "w"), indent=1)
    open(os.path.join(out_dir, "ft_report.md"), "w", encoding="utf-8").write(to_md(rep))
    print(to_md(rep))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--refs", nargs="*", default=[], help="NAME=<run dir with preds_dev.jsonl / preds_test.jsonl>")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--shuffle-splits", nargs="*", default=["dev"])
    a = ap.parse_args(argv)
    import train_viso  # noqa: F401  (registers datasets / models / task)
    from builders.task_builder import build_task
    from train_viso import load_config
    task = build_task(load_config(a.config_file, a.opts))
    return run(task, dict(r.split("=", 1) for r in a.refs), a.resamples, shuffle_splits=tuple(a.shuffle_splits))


if __name__ == "__main__":
    main()
