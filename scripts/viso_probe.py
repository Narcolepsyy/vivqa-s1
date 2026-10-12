"""Learning probe: train a config for N steps on the real train split and print per-type train loss / dev accuracy, so "is the model learning?" is answered in minutes.
    python scripts/viso_probe.py --steps 800 --opts MODEL.VISION_EMBEDDING.FEATURE_NORM True TRAINING.LR_ENCODER 0.0005
Writes nothing but a throw-away run dir under --work. Per-type loss/accuracy are what the full training log should also show (train_log.jsonl has per_type_* fields)."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_viso import load_config  # noqa: E402
from builders.task_builder import build_task  # noqa: E402
from tasks.typed_decision_task import collect_logits  # noqa: E402
from utils import viso_rlcd as rlcd  # noqa: E402
from utils.viso_s1 import label_space  # noqa: E402

TYPES = ("noul", "score", "choice")


def dev_report(task):
    """per-type dev accuracy and NLL (temperature-free) from s1bench"""
    ev = task.evaluate_split("dev")["primary"]
    return {t: (round(ev[t]["accuracy"], 3), round(ev[t]["nll"], 3)) for t in TYPES if t in ev} | {"all": (round(ev["all"]["accuracy"], 3), round(ev["all"]["nll"], 3))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default=os.path.join(REPO, "configs", "viso_siglip2.yaml"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--work", default="/tmp/viso_probe")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--only-type", choices=TYPES, default=None, help="train only on items of this type (e.g. the yes/no-only overfit check)")
    ap.add_argument("--n-items", type=int, default=0, help="with --only-type: use only the first N train items of that type")
    ap.add_argument("--early-stop-step", type=int, default=0, help="stop if the yes/no train loss is still >= --early-stop-loss at this step")
    ap.add_argument("--early-stop-loss", type=float, default=0.685)
    ap.add_argument("--grad-at", type=int, nargs="*", default=[100, 500], help="steps at which per-module gradient norms are printed")
    ap.add_argument("--ckpt-name", default=None, help="file name stem of the saved model (default: <config>_<hash of opts>)")
    ap.add_argument("--ckpt-dir", default=os.path.expanduser("~/data/viso_diag"), help="the final model is saved here as ckpt_<config>_<hash of opts>.pth (+ .json with the arguments)")
    a = ap.parse_args(argv)
    shutil.rmtree(os.path.join(a.work, "probe"), ignore_errors=True)
    cfg = load_config(a.config_file, ["MODEL.NAME", "probe", "TRAINING.OUTPUT_PATH", a.work, "TRAINING.CHECKPOINT_PATH", a.work] + a.opts)
    task = build_task(cfg)
    if a.only_type:
        keep = [it for it in task.datasets["train"].items if it["questions"]["q"]["type"] == a.only_type]
        keep = keep[:a.n_items] if a.n_items else keep
        task.datasets["train"].items = keep
        task.datasets["train"].index = {it["id"]: i for i, it in enumerate(keep)}
        from torch.utils.data import DataLoader
        from data_utils.datasets.s1_dataset import s1_collate
        task.train_loader = DataLoader(task.datasets["train"], batch_size=task.batch_size, shuffle=True, collate_fn=s1_collate, generator=task.train_gen)
        print(f"training on {len(keep)} {a.only_type} items only", flush=True)
    sched = task._make_scheduler(a.steps)
    tag = hashlib.sha1(json.dumps([a.config_file, a.opts, a.steps]).encode()).hexdigest()[:8]
    os.makedirs(a.ckpt_dir, exist_ok=True)
    stem = f"ckpt_{a.ckpt_name}" if a.ckpt_name else f"ckpt_{os.path.basename(a.config_file)[:-5]}_{tag}"
    side = {"config_file": a.config_file, "opts": a.opts, "steps": a.steps, "only_type": a.only_type}
    best, history = {}, []                                             # best dev NLL per key ("all" = overall, noul, score, choice): checkpoint + metrics AT that step

    def evaluate(step_now, train_losses):
        rep = dev_report(task)
        print("   dev (acc, nll):", rep, flush=True)
        history.append({"step": step_now, "train_loss": train_losses, "dev": {k: list(v) for k, v in rep.items()}})
        for key, (acc, nll) in rep.items():
            if key not in best or nll < best[key]["nll"]:
                best[key] = {"step": step_now, "nll": nll, "acc": acc, "dev": {k: list(v) for k, v in rep.items()}}
                path_b = os.path.join(a.ckpt_dir, f"{stem}_best_{key}.pth")
                torch.save({"state_dict": task.model.state_dict()}, path_b)
                json.dump(dict(side, selected="best_dev_nll_" + key, step=step_now), open(path_b[:-4] + ".json", "w"))
                print(f"   BEST dev NLL ({key}) at step {step_now}: nll {nll:.4f} acc {acc:.3f} | all types {rep}", flush=True)
        return rep

    hist = {n: [] for n in TYPES}
    step, t0, last = 0, time.time(), {}
    while step < a.steps:
        for batch in task.train_loader:
            task.model.train()
            batch.to(task.device)
            per = task.item_loss(task.model(batch), batch, "soft_ce")
            loss = rlcd.reduce_loss(task.weighted(per, batch.type_id), batch.type_id, bool(cfg.LOSS.TYPE_BALANCE))
            task.optim.zero_grad()
            loss.backward()
            if step + 1 in a.grad_at:
                g = {}
                for n, p in task.model.named_parameters():
                    if p.grad is not None:
                        g.setdefault(".".join(n.split(".")[:2]), []).append(float(p.grad.norm()) ** 2)
                print(f"grad norms at step {step + 1}:", {k: round(float(np.sqrt(sum(v))), 4) for k, v in g.items()}, flush=True)
            torch.nn.utils.clip_grad_norm_(task.model.parameters(), cfg.TRAINING.GRAD_CLIP)
            task.optim.step()
            sched.step()
            step += 1
            for t, n in enumerate(TYPES):
                k = batch.type_id == t
                if k.any():
                    hist[n].append(float(per[k].mean()))
            if step % 100 == 0:
                last = {n: round(float(np.mean(v[-60:])), 3) for n, v in hist.items()}
                print(f"step {step} per-type train loss {last}  {time.time() - t0:.0f}s", flush=True)
            if a.early_stop_step and step == a.early_stop_step and hist["noul"] and float(np.mean(hist["noul"][-60:])) >= a.early_stop_loss:
                print(f"EARLY STOP at step {step}: yes/no train loss {np.mean(hist['noul'][-60:]):.4f} >= {a.early_stop_loss}", flush=True)
                evaluate(step, last)
                step = a.steps
                break
            if step % a.eval_every == 0 or step == a.steps:
                evaluate(step, {n: round(float(np.mean(v[-60:])), 4) for n, v in hist.items() if v})
            if step >= a.steps:
                break
    path = os.path.join(a.ckpt_dir, stem + ".pth")
    torch.save({"state_dict": task.model.state_dict()}, path)
    json.dump(side, open(path[:-4] + ".json", "w"))
    json.dump({"history": history, "best": best, "final": history[-1] if history else None}, open(os.path.join(a.ckpt_dir, stem.replace("ckpt_", "", 1) + "_history.json"), "w"), indent=1)
    print("saved", path, flush=True)
    return last


if __name__ == "__main__":
    main()
