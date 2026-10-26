"""System-One Multi-VLM Ensemble: combines calibrated letter-option log-probabilities from multiple fine-tuned System-One VLMs
(e.g., vllm-sr/d3-nano + Qwen/Qwen3.5-2B + openbmb/MiniCPM-V-4.7-1B + Qwen/Qwen3.5-0.8B).
Per-type ensemble weights are fitted strictly on the `calib` split (minimizing soft cross-entropy NLL on `calib.jsonl`),
then evaluated via `s1_vlm_eval.finish_run` on `dev` and `test` (`All`, `Core`, `Image-Unseen`)."""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S1ROOT = os.path.abspath(os.path.join(REPO, "..", "vivqa_s1_v2"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, S1ROOT)

import s1_vlm_eval as V  # noqa: E402
from s1bench.schema import label_space  # noqa: E402
from utils import viso_calibration as C  # noqa: E402


def load_run_split(run_dir, split, canonical_items, temps_override=None):
    lg = C.load_logits(os.path.join(run_dir, f"logits_{split}.npz"))
    if temps_override is not None:
        temps = temps_override
    else:
        temps_file = os.path.join(run_dir, "calibration", "temperatures.json")
        if os.path.isfile(temps_file):
            temps = {int(k): float(v) for k, v in json.load(open(temps_file))["per_type_nll"].items()}
        else:
            temps = None
    raw_probs = C.probs_from_logits(lg["logits"], lg["space_mask"], lg["type_id"], temps)
    id_to_idx = {int(iid): i for i, iid in enumerate(lg["ids"])}

    preds_by_id = {}
    preds_jsonl = os.path.join(run_dir, "calibration", f"preds_{split}_pertype.jsonl")
    if os.path.isfile(preds_jsonl):
        with open(preds_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    preds_by_id[int(r["id"])] = r.get("probabilities", {})

    N = len(canonical_items)
    K = lg["logits"].shape[1]
    ids = np.array([int(it["id"]) for it in canonical_items], dtype=np.int64)
    type_id = np.zeros(N, dtype=np.int64)
    space_mask = np.zeros((N, K), dtype=bool)
    probs = np.zeros((N, K), dtype=np.float32)

    type_map = {"noul": 0, "score": 1, "choice": 2}
    for i, it in enumerate(canonical_items):
        iid = int(it["id"])
        q = it["questions"]["q"]
        t_id = type_map[q["type"]]
        type_id[i] = t_id
        sp = label_space(q)
        L = len(sp)
        space_mask[i, :L] = True

        j = id_to_idx[iid]
        if int(lg["space_mask"][j].sum()) == L:
            p_vec = raw_probs[j, :L].astype(np.float32)
        elif iid in preds_by_id:
            pmap = preds_by_id[iid]
            p_vec = np.array([float(pmap.get(str(lbl), 1e-6)) for lbl in sp], dtype=np.float32)
        else:
            p_vec = np.ones(L, dtype=np.float32) / L
        p_sum = float(p_vec.sum())
        probs[i, :L] = p_vec / p_sum if p_sum > 0 else (1.0 / L)

    log_probs = np.where(space_mask, np.log(np.clip(probs, 1e-12, 1.0)), -1e9).astype(np.float32)
    return ids, type_id, space_mask, probs, log_probs


def fit_per_type_weights(calib_items, ids, type_id, space_mask, model_probs_list, model_log_probs_list):
    """Fit non-negative simplex weights w_t over models per question type t in {0, 1, 2} on `calib` to minimize soft-CE NLL."""
    M = len(model_probs_list)
    targets, _ = V.soft_targets(calib_items)

    weights_log = np.zeros((3, M), dtype=np.float32)
    weights_lin = np.zeros((3, M), dtype=np.float32)
    best_mode_per_type = {}

    for t_id in (0, 1, 2):
        idx = np.flatnonzero(type_id == t_id)
        tgt = torch.tensor(targets[idx], dtype=torch.float32)
        msk = torch.tensor(space_mask[idx], dtype=torch.bool)
        lp_stack = torch.tensor(np.stack([lp[idx] for lp in model_log_probs_list], axis=0), dtype=torch.float32)  # [M, n_t, K]
        p_stack = torch.tensor(np.stack([p[idx] for p in model_probs_list], axis=0), dtype=torch.float32)        # [M, n_t, K]

        # 1) Optimize log-linear pool (geometric mean in probability space)
        u_log = torch.zeros(M, requires_grad=True)
        opt = torch.optim.Adam([u_log], lr=0.05)
        for _ in range(600):
            opt.zero_grad()
            w = torch.softmax(u_log, dim=0).view(M, 1, 1)
            comb = (w * lp_stack).sum(dim=0).masked_fill(~msk, -1e9)
            log_p = torch.log_softmax(comb, dim=-1)
            loss = -(tgt * log_p).sum(dim=-1).mean()
            loss.backward()
            opt.step()
        loss_log = float(loss.detach())
        w_log = torch.softmax(u_log.detach(), dim=0).numpy()

        # 2) Optimize linear probability mixture (arithmetic mean in probability space)
        u_lin = torch.zeros(M, requires_grad=True)
        opt2 = torch.optim.Adam([u_lin], lr=0.05)
        for _ in range(600):
            opt2.zero_grad()
            w = torch.softmax(u_lin, dim=0).view(M, 1, 1)
            comb_p = (w * p_stack).sum(dim=0)
            comb_p = torch.where(msk, torch.clamp(comb_p, min=1e-12), torch.ones_like(comb_p))
            loss2 = -(tgt * torch.log(comb_p) * msk.float()).sum(dim=-1).mean()
            loss2.backward()
            opt2.step()
        loss_lin = float(loss2.detach())
        w_lin = torch.softmax(u_lin.detach(), dim=0).numpy()

        weights_log[t_id] = w_log
        weights_lin[t_id] = w_lin
        best_mode_per_type[t_id] = ("lin" if loss_lin < loss_log else "log", min(loss_lin, loss_log), loss_log, loss_lin)

    return weights_log, weights_lin, best_mode_per_type


def combine_split(type_id, space_mask, probs_list, log_probs_list, weights_log, weights_lin, mode="auto", best_mode_per_type=None):
    N, K = space_mask.shape
    out_logits = np.full((N, K), -1e9, dtype=np.float32)
    for t_id in (0, 1, 2):
        idx = np.flatnonzero(type_id == t_id)
        use_mode = best_mode_per_type[t_id][0] if mode == "auto" else mode
        if use_mode == "lin":
            w = weights_lin[t_id][:, None, None]
            p_stack = np.stack([p[idx] for p in probs_list], axis=0)
            comb_p = (w * p_stack).sum(axis=0)
            lg = np.where(space_mask[idx], np.log(np.clip(comb_p, 1e-12, 1.0)), -1e9)
        else:
            w = weights_log[t_id][:, None, None]
            lp_stack = np.stack([lp[idx] for lp in log_probs_list], axis=0)
            lg = np.where(space_mask[idx], (w * lp_stack).sum(axis=0), -1e9)
        out_logits[idx] = lg.astype(np.float32)
    return out_logits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir pairs")
    ap.add_argument("--s1-data", default=os.path.join(S1ROOT, "data"))
    ap.add_argument("--root", default=S1ROOT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-recalibrate-constituents", action="store_true", help="Use existing temperatures.json instead of fitting on canonical calib.jsonl")
    a = ap.parse_args(argv)

    names, run_dirs = [], []
    for item in a.runs:
        n, d = item.split("=", 1)
        names.append(n)
        run_dirs.append(d)

    os.makedirs(a.out_dir, exist_ok=True)
    splits_items = {sp: V.load_items(a.s1_data, sp, 0) for sp in ("calib", "dev", "test")}
    calib_gold = C.gold_arrays(splits_items["calib"])

    constituent_temps = {}
    if not a.no_recalibrate_constituents:
        for d in run_dirs:
            lg_c = C.load_logits(os.path.join(d, "logits_calib.npz"))
            if len(lg_c["ids"]) == len(splits_items["calib"]) and np.array_equal(lg_c["ids"], calib_gold["ids"]):
                constituent_temps[d] = C.fit_temperatures(lg_c["logits"], lg_c["space_mask"], lg_c["type_id"], gold_idx=calib_gold["gold_idx"])
            else:
                id_map = {int(iid): i for i, iid in enumerate(lg_c["ids"])}
                idx = [id_map[int(it["id"])] for it in splits_items["calib"]]
                constituent_temps[d] = C.fit_temperatures(lg_c["logits"][idx], lg_c["space_mask"][idx], lg_c["type_id"][idx], gold_idx=calib_gold["gold_idx"])

    splits_data = {}
    for sp in ("calib", "dev", "test"):
        probs_list, log_probs_list = [], []
        ref_ids = ref_ty = ref_mask = None
        for d in run_dirs:
            ids, ty, mask, probs, log_probs = load_run_split(d, sp, splits_items[sp], temps_override=constituent_temps.get(d))
            if ref_ids is None:
                ref_ids, ref_ty, ref_mask = ids, ty, mask
            probs_list.append(probs)
            log_probs_list.append(log_probs)
        splits_data[sp] = (ref_ids, ref_ty, ref_mask, probs_list, log_probs_list)

    c_ids, c_ty, c_mask, c_p, c_lp = splits_data["calib"]
    w_log, w_lin, best_modes = fit_per_type_weights(splits_items["calib"], c_ids, c_ty, c_mask, c_p, c_lp)

    type_names = {0: "noul", 1: "score", 2: "choice"}
    print("=== Fitted Per-Type Ensemble Weights on Calib ===")
    for t_id in (0, 1, 2):
        m_str, best_l, l_log, l_lin = best_modes[t_id]
        w = w_lin[t_id] if m_str == "lin" else w_log[t_id]
        w_fmt = ", ".join(f"{n}={v:.3f}" for n, v in zip(names, w))
        print(f"  {type_names[t_id]:6s} (mode={m_str}, calib NLL={best_l:.4f} [log={l_log:.4f}, lin={l_lin:.4f}]): {w_fmt}")

    import viso_calibrate as VC
    for variant, mode_arg in [("ensemble_calib_opt", "auto"), ("ensemble_log_pool", "log"), ("ensemble_lin_pool", "lin")]:
        v_dir = os.path.join(a.out_dir, variant)
        os.makedirs(v_dir, exist_ok=True)
        for sp in ("calib", "dev", "test"):
            ids, ty, mask, p_list, lp_list = splits_data[sp]
            lg = combine_split(ty, mask, p_list, lp_list, w_log, w_lin, mode=mode_arg, best_mode_per_type=best_modes)
            C.save_logits(os.path.join(v_dir, f"logits_{sp}.npz"), ids, ty, lg, mask)
        VC.main(["--run-dir", v_dir, "--s1-dir", a.s1_data, "--member-runs", *run_dirs])
        res = V.evaluate_views(v_dir, a.s1_data, a.root)
        print(f"\n=== {variant} Results ===")
        for sp in ("dev", "test"):
            for vw in ("all", "core", "image_unseen"):
                if vw not in res[sp]["pertype"]:
                    continue
                p = res[sp]["pertype"][vw]["primary"]
                print(
                    f"  {sp:4s} | {vw:12s} | Acc={p['all']['accuracy']*100:.2f}% | NLL={p['all']['nll']:.4f} | "
                    f"Brier={p['all']['brier']:.4f} | ECE={p['all']['ece']:.4f} | "
                    f"noul={p['noul']['accuracy']*100:.2f}% | score={p['score']['accuracy']*100:.2f}% | choice={p['choice']['accuracy']*100:.2f}%"
                )

    meta = {
        "models": dict(zip(names, run_dirs)),
        "weights_log_pool": {type_names[t]: {n: float(w_log[t, i]) for i, n in enumerate(names)} for t in (0, 1, 2)},
        "weights_lin_pool": {type_names[t]: {n: float(w_lin[t, i]) for i, n in enumerate(names)} for t in (0, 1, 2)},
        "best_mode_per_type_on_calib": {type_names[t]: best_modes[t][0] for t in (0, 1, 2)},
    }
    with open(os.path.join(a.out_dir, "ensemble_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
