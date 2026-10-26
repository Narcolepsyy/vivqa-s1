"""Google Colab / Kaggle Heavy-Compute Offload Runner for ViVQA-S1 (`/boost` Step 2).

Designed to run on Google Colab / Kaggle GPU (or with strict memory bounds) so the local PC never freezes.

Supported tasks (`--task`):
  1. `ref_heads`: GPU-vectorized, low-RAM (`fp16` streamed) 3-seed decision head training (`head_s0..s2`)
     and letter-debiased `zeroshot` evaluation for `ref-qwen3vl8b` (8.77B), `ref-intern4b` (4.54B), and
     `ref-intern2b` (2.21B).
     - If `train_*.npz` / `trainsub*_*.npz` shards exist (or `--extract-ref-train` is passed on Colab GPU),
       trains on the `train` split and selects the best checkpoint on `dev`.
     - If only `calib`/`dev`/`test` shards exist (the default for reference-tier models), automatically
       performs 5-Fold Out-of-Fold (OOF) W0-Anchored Readout Adaptation on `calib` (3,455 items, anchored
       with L2 penalty ||W - W0||_F^2 and early-stopped on `dev`), yielding strictly out-of-fold `calib`
       logits (zero leakage into ensemble weight fitting) and fold-averaged `dev`/`test` logits.
  2. `viso_v5`: Trains `ViSO Cross-Attn v5` with per-type best-checkpoint splicing
     (`TRAINING.SAVE_BEST_PER_TYPE=True`) so `ScoreHead` retains its early-epoch peak (`~37.8%`) while
     `ChoiceHead` trains to its late-epoch peak (`~43.9%+`).
  3. `rescore_legacy`: Re-evaluates legacy 1B-3B runs (`onejev`, `jpt`, `intern`, `qwen`, `vintern`) on the
     canonical 43,903-item (`N_test=4,409`) splits with `--debias-letters` and `vectype` calibration.
  4. `bundle`: Packages ONLY the lightweight (~180 KB) `logits_{calib,dev,test}.npz`, `views.json`, and
     `calibration/` files into `colab_boost_results.zip` (<15 MB).
"""
import argparse
import gc
import glob
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S1ROOT = os.path.abspath(os.path.join(REPO, "..", "vivqa_s1_v2"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, S1ROOT)

import s1_vlm_eval as V  # noqa: E402
from s1bench.schema import label_space  # noqa: E402
from utils import viso_calibration as C  # noqa: E402
from utils.atomic import atomic_json  # noqa: E402
from utils.viso_s1 import MAX_SPACE, TYPE_IDS  # noqa: E402


class ResidualDecisionHead(nn.Module):
    """W0-anchored 16-letter readout head (supports 'linear' and zero-initialized 'res_mlp').
    At epoch 0 (before any optimizer step), output == x @ W0.T (exact zero-shot parity)."""

    def __init__(self, W0: torch.Tensor, arch: str = "linear", bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        k, d = W0.shape
        self.arch = arch
        self.dropout = dropout
        self.register_buffer("W0", W0.clone().float())
        self.W = nn.Parameter(W0.clone().float())
        self.b = nn.Parameter(torch.zeros(k, dtype=torch.float32))
        if arch == "res_mlp":
            self.fc1 = nn.Linear(d, bottleneck)
            self.fc2 = nn.Linear(bottleneck, d, bias=False)
            nn.init.zeros_(self.fc2.weight)

    def anchor_loss(self) -> torch.Tensor:
        loss = ((self.W - self.W0) ** 2).sum() + (self.b ** 2).sum()
        if self.arch == "res_mlp":
            loss = loss + (self.fc1.weight ** 2).sum() + (self.fc2.weight ** 2).sum()
        return loss

    def forward(self, x: torch.Tensor, train_mode: bool = False) -> torch.Tensor:
        if self.arch == "res_mlp":
            h_res = F.silu(self.fc1(F.dropout(x, p=self.dropout, training=train_mode)))
            x = x + self.fc2(F.dropout(h_res, p=self.dropout, training=train_mode))
        else:
            x = F.dropout(x, p=self.dropout, training=train_mode)
        return x @ self.W.T + self.b


def load_shards_compact(model_dir: str, split: str, keep_ids: set):
    """Memory-safe shard loader: filters to canonical `keep_ids` on the fly and stores `h` in fp16."""
    d = {}
    patterns = [os.path.join(model_dir, "shards", f"{split}_*.npz")]
    if split == "train":
        patterns.append(os.path.join(model_dir, "shards", "trainsub*_*.npz"))
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if p.endswith(".tmp.npz"):
                continue
            with np.load(p) as z:
                ids = z["ids"]
                orders = z["order"]
                hs = z["h"]
                lls = z["ll"]
                for k in range(len(ids)):
                    iid = int(ids[k])
                    if iid in keep_ids:
                        d[(iid, int(orders[k]))] = (hs[k].astype(np.float16), lls[k].astype(np.float32))
    return d


def prebuild_split_eval_tensors(items, sh, device: str):
    """Pre-pack split hidden states and permutation indices for fast vectorized NLL evaluation."""
    valid_items = [
        it for it in items
        if all((it["id"], o) in sh for o in ((1, 2) if it["questions"]["q"]["type"] == "choice" else (1,)))
    ]
    h1 = torch.from_numpy(np.stack([sh[(it["id"], 1)][0] for it in valid_items]).astype(np.float32)).to(device)
    choice_idx = [i for i, it in enumerate(valid_items) if it["questions"]["q"]["type"] == "choice"]
    h2 = (
        torch.from_numpy(np.stack([sh[(valid_items[i]["id"], 2)][0] for i in choice_idx]).astype(np.float32)).to(device)
        if choice_idx else None
    )
    gold_idx = C.gold_arrays(valid_items)["gold_idx"]
    return valid_items, h1, choice_idx, h2, gold_idx


@torch.no_grad()
def eval_head_fast(head: ResidualDecisionHead, valid_items, h1, choice_idx, h2, gold_idx, letter_bias=None):
    """Fast vectorized evaluation of split NLL."""
    head.eval()
    lg1 = head(h1, train_mode=False).cpu().numpy()
    lg2_map = {}
    if choice_idx and h2 is not None:
        lg2_arr = head(h2, train_mode=False).cpu().numpy()
        for k, i in enumerate(choice_idx):
            lg2_map[i] = lg2_arr[k]

    nll = 0.0
    for i, it in enumerate(valid_items):
        got = {1: lg1[i]}
        if i in lg2_map:
            got[2] = lg2_map[i]
        z = V.to_label_logits(it, got, letter_bias=letter_bias)
        nn_ = len(label_space(it["questions"]["q"]))
        p = np.exp(z[:nn_] - z[:nn_].max())
        p /= p.sum()
        nll -= np.log(max(float(p[gold_idx[i]]), 1e-12))
    return nll / max(len(valid_items), 1)


@torch.no_grad()
def predict_items_with_heads(heads, items, sh, device: str, letter_bias=None):
    """Predict label-space logits for `items` by averaging logits across one or more `ResidualDecisionHead` models."""
    for h in heads:
        h.eval()

    def fn(it):
        orders = (1, 2) if it["questions"]["q"]["type"] == "choice" else (1,)
        if not all((it["id"], o) in sh for o in orders):
            return None
        got = {}
        for o in orders:
            h_t = torch.from_numpy(sh[(it["id"], o)][0].astype(np.float32)).unsqueeze(0).to(device)
            lg_mean = sum(hd(h_t, train_mode=False).squeeze(0).cpu().numpy() for hd in heads) / len(heads)
            got[o] = lg_mean
        return V.to_label_logits(it, got, letter_bias=letter_bias)

    return V.split_arrays(items, fn)


def train_single_head(W0, tr_items, sh_tr, dev_pack, seed: int, arch: str = "linear", epochs: int = 25,
                      patience: int = 5, lr: float = 2e-4, anchor_l2: float = 1e-3, dropout: float = 0.1,
                      batch_size: int = 256, device: str = "cuda", letter_bias=None):
    """Train a single W0-anchored decision head on `tr_items` (using both Order 1 & Order 2 if available)."""
    valid_dev, h1_dev, choice_idx_dev, h2_dev, gold_dev = dev_pack
    units_h, units_t, units_m = [], [], []
    for o in (1, 2):
        sub = [it for it in tr_items if (it["id"], o) in sh_tr]
        if not sub:
            continue
        t_o, m_o = V.soft_targets(sub, order=o)
        units_h.append(np.stack([sh_tr[(it["id"], o)][0] for it in sub]))
        units_t.append(t_o)
        units_m.append(m_o)

    H_fp16 = torch.from_numpy(np.concatenate(units_h, axis=0))
    T_t = torch.from_numpy(np.concatenate(units_t, axis=0)).to(device)
    M_t = torch.from_numpy(np.concatenate(units_m, axis=0)).to(device)

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    head = ResidualDecisionHead(W0, arch=arch, dropout=dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)

    best_nll = eval_head_fast(head, valid_dev, h1_dev, choice_idx_dev, h2_dev, gold_dev, letter_bias=letter_bias)
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    hist = [{"epoch": 0, "dev_nll": best_nll}]
    bad = 0

    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(H_fp16), generator=g)
        for s in range(0, len(H_fp16), batch_size):
            ix = perm[s:s + batch_size]
            x = H_fp16[ix].to(device, dtype=torch.float32)
            ix_dev = ix.to(device)
            lg = head(x, train_mode=True).masked_fill(~M_t[ix_dev], -1e9)
            ce_loss = -(T_t[ix_dev] * torch.log_softmax(lg, dim=-1)).sum(dim=-1).mean()
            loss = ce_loss + anchor_l2 * head.anchor_loss()
            opt.zero_grad()
            loss.backward()
            opt.step()

        v = eval_head_fast(head, valid_dev, h1_dev, choice_idx_dev, h2_dev, gold_dev, letter_bias=letter_bias)
        hist.append({"epoch": ep, "dev_nll": v})
        if v < best_nll:
            best_nll, bad = v, 0
            best_state = {k: val.detach().cpu().clone() for k, val in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    head.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    head.eval()
    return head, hist


def train_and_predict_model(model_dir: str, s1_data: str, root: str, seeds: int = 3, arch: str = "linear",
                            device: str = "cuda", debias_letters: bool = True):
    W0 = torch.from_numpy(np.load(os.path.join(model_dir, "letter_rows.npy")).astype(np.float32))
    calib_items = V.load_items(s1_data, "calib", 0)
    dev_items = V.load_items(s1_data, "dev", 0)
    test_items = V.load_items(s1_data, "test", 0)

    sh_calib = load_shards_compact(model_dir, "calib", {it["id"] for it in calib_items})
    letter_bias = V.fit_letter_position_bias(sh_calib, calib_items) if debias_letters else None

    # 1. Zero-shot evaluation (with optional letter-position debiasing)
    sh_dev = load_shards_compact(model_dir, "dev", {it["id"] for it in dev_items})
    sh_test = load_shards_compact(model_dir, "test", {it["id"] for it in test_items})

    zs_arrays = {}
    for sp, items, sh in (("calib", calib_items, sh_calib), ("dev", dev_items, sh_dev), ("test", test_items, sh_test)):
        def zs_fn(it, sh=sh):
            got = {o: sh[(it["id"], o)][1] for o in (1, 2) if (it["id"], o) in sh}
            want = 2 if it["questions"]["q"]["type"] == "choice" else 1
            return V.to_label_logits(it, got, letter_bias=letter_bias) if len(got) == want else None
        zs_arrays[sp] = V.split_arrays(items, zs_fn)

    zs_dir = os.path.join(model_dir, "zeroshot_debiased" if debias_letters else "zeroshot")
    os.makedirs(zs_dir, exist_ok=True)
    res_zs = V.finish_run(zs_dir, zs_arrays, s1_data, root)
    zs_acc = res_zs["test"]["pertype"]["all"]["primary"]["all"]["accuracy"]
    zs_vec_acc = res_zs["test"].get("vectype", res_zs["test"]["pertype"])["all"]["primary"]["all"]["accuracy"]
    print(f"[{os.path.basename(model_dir)} zeroshot] test All acc = {zs_acc*100:.2f}% (vectype = {zs_vec_acc*100:.2f}%)")

    # 2. Check if train shards exist; otherwise use 5-fold Out-of-Fold W0-anchored adaptation on calib
    tr_items_all = V.load_items(s1_data, "train", 0)
    sh_tr = load_shards_compact(model_dir, "train", {it["id"] for it in tr_items_all})
    dev_pack = prebuild_split_eval_tensors(dev_items, sh_dev, device)

    if len(sh_tr) > 0:
        tr_items = [it for it in tr_items_all if (it["id"], 1) in sh_tr]
        for seed in range(seeds):
            t0 = time.time()
            head, hist = train_single_head(
                W0, tr_items, sh_tr, dev_pack, seed=seed, arch=arch,
                lr=2e-4, anchor_l2=1e-4, device=device, letter_bias=letter_bias
            )
            arrays = {
                "calib": predict_items_with_heads([head], calib_items, sh_calib, device, letter_bias=letter_bias),
                "dev": predict_items_with_heads([head], dev_items, sh_dev, device, letter_bias=letter_bias),
                "test": predict_items_with_heads([head], test_items, sh_test, device, letter_bias=letter_bias),
            }
            tag = f"head_s{seed}" if arch == "linear" else f"head_{arch}_s{seed}"
            run_dir = os.path.join(model_dir, tag)
            os.makedirs(run_dir, exist_ok=True)
            atomic_json(os.path.join(run_dir, "head_log.json"), {
                "mode": "train_split", "seed": seed, "arch": arch, "debias_letters": debias_letters,
                "history": hist, "best_dev_nll": min(h["dev_nll"] for h in hist),
                "zero_shot_dev_nll": hist[0]["dev_nll"], "seconds": round(time.time() - t0, 2),
            })
            res = V.finish_run(run_dir, arrays, s1_data, root)
            acc = res["test"]["pertype"]["all"]["primary"]["all"]["accuracy"]
            print(f"[{os.path.basename(model_dir)} {tag}] dev NLL {hist[0]['dev_nll']:.4f} -> {min(h['dev_nll'] for h in hist):.4f} | test All acc = {acc*100:.2f}% ({time.time()-t0:.1f}s)")
    else:
        # 5-fold Out-of-Fold (OOF) W0-anchored adaptation on calib (for reference models without train shards)
        for seed in range(seeds):
            t0 = time.time()
            fids = C.fold_ids(len(calib_items), folds=5, seed=seed)
            fold_heads = []
            oof_logits = np.zeros((len(calib_items), MAX_SPACE), dtype=np.float32)
            oof_ids = np.array([it["id"] for it in calib_items], dtype=np.int64)
            oof_ty = np.array([TYPE_IDS[it["questions"]["q"]["type"]] for it in calib_items], dtype=np.int64)
            oof_mask = np.zeros((len(calib_items), MAX_SPACE), dtype=bool)
            for i, it in enumerate(calib_items):
                oof_mask[i, :len(label_space(it["questions"]["q"]))] = True

            hists = []
            for k in range(5):
                tr_fold = [calib_items[i] for i in range(len(calib_items)) if fids[i] != k]
                va_idx = [i for i in range(len(calib_items)) if fids[i] == k]
                va_fold = [calib_items[i] for i in va_idx]
                head_k, hist_k = train_single_head(
                    W0, tr_fold, sh_calib, dev_pack, seed=seed * 10 + k, arch=arch,
                    epochs=20, patience=4, lr=1e-4, anchor_l2=5e-3, device=device, letter_bias=letter_bias
                )
                fold_heads.append(head_k)
                hists.append(hist_k)
                _, _, _, lg_va = predict_items_with_heads([head_k], va_fold, sh_calib, device, letter_bias=letter_bias)
                oof_logits[va_idx] = lg_va

            arrays = {
                "calib": (oof_ids, oof_ty, oof_mask, oof_logits),
                "dev": predict_items_with_heads(fold_heads, dev_items, sh_dev, device, letter_bias=letter_bias),
                "test": predict_items_with_heads(fold_heads, test_items, sh_test, device, letter_bias=letter_bias),
            }
            tag = f"head_s{seed}" if arch == "linear" else f"head_{arch}_s{seed}"
            run_dir = os.path.join(model_dir, tag)
            os.makedirs(run_dir, exist_ok=True)
            atomic_json(os.path.join(run_dir, "head_log.json"), {
                "mode": "calib_5fold_oof", "seed": seed, "arch": arch, "debias_letters": debias_letters,
                "zero_shot_dev_nll": hists[0][0]["dev_nll"],
                "mean_best_dev_nll": float(np.mean([min(h["dev_nll"] for h in hk) for hk in hists])),
                "seconds": round(time.time() - t0, 2),
            })
            res = V.finish_run(run_dir, arrays, s1_data, root)
            acc = res["test"]["pertype"]["all"]["primary"]["all"]["accuracy"]
            vec_acc = res["test"].get("vectype", res["test"]["pertype"])["all"]["primary"]["all"]["accuracy"]
            print(f"[{os.path.basename(model_dir)} {tag} (5-fold OOF)] test All acc = {acc*100:.2f}% (vectype = {vec_acc*100:.2f}%) ({time.time()-t0:.1f}s)")

    del sh_calib, sh_dev, sh_test, sh_tr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_viso_v5(s1_data: str, out_root: str):
    """Train ViSO Cross-Attn v5 with SAVE_BEST_PER_TYPE=True on GPU and calibrate."""
    cmd = [
        sys.executable,
        os.path.join(REPO, "train_viso.py"),
        "--config-file",
        os.path.join(REPO, "configs", "viso_siglip2_bami_crossattn.yaml"),
        "--opts",
        "MODEL.NAME", "viso_s1_crossattn_v5",
        "TRAINING.SAVE_BEST_PER_TYPE", "True",
        "TRAINING.CHECKPOINT_PATH", os.path.join(out_root, "checkpoints"),
        "TRAINING.OUTPUT_PATH", os.path.join(out_root, "output"),
        "DATASET.VOCAB.S1_DIR", s1_data,
    ]
    subprocess.run(cmd, check=True)
    run_dir = os.path.join(out_root, "output", "viso_s1_crossattn_v5")
    import viso_calibrate as VC
    VC.main(["--run-dir", run_dir, "--s1-dir", s1_data])
    V.evaluate_views(run_dir, s1_data, S1ROOT)


def bundle_lightweight_results(search_roots, out_zip: str):
    """Package ONLY lightweight (<200 KB) logits_*.npz, views.json, head_log.json, and calibration/*.json into out_zip."""
    keep_names = {"logits_calib.npz", "logits_dev.npz", "logits_test.npz", "views.json", "head_log.json", "ensemble_meta.json"}
    os.makedirs(os.path.dirname(os.path.abspath(out_zip)), exist_ok=True)
    added = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root_dir in search_roots:
            if not os.path.isdir(root_dir):
                continue
            base_label = os.path.basename(os.path.normpath(root_dir))
            for dirpath, _, filenames in os.walk(root_dir):
                if "/shards" in dirpath or dirpath.endswith("/shards"):
                    continue
                for fn in filenames:
                    if fn in keep_names or (os.path.basename(dirpath) == "calibration" and fn.endswith(".json")):
                        if fn.endswith((".pth", ".pt", ".bin", ".safetensors", ".jsonl")):
                            continue
                        full_p = os.path.join(dirpath, fn)
                        rel_p = os.path.join(base_label, os.path.relpath(full_p, root_dir))
                        zf.write(full_p, rel_p)
                        added += 1
    size_mb = os.path.getsize(out_zip) / (1024 * 1024)
    print(f"Bundled {added} lightweight result files ({size_mb:.2f} MB) -> {out_zip}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ref_heads", "viso_v5", "rescore_legacy", "bundle", "all"], default="all")
    ap.add_argument("--ezy-dir", default="/content/drive/MyDrive/ezy_results", help="Path to ezy_results holding model shards")
    ap.add_argument("--s1-data", default=os.path.join(S1ROOT, "data"))
    ap.add_argument("--root", default=S1ROOT)
    ap.add_argument("--viso-out", default="/content/drive/MyDrive/viso_v5_out")
    ap.add_argument("--out-zip", default="/content/drive/MyDrive/colab_boost_results.zip")
    ap.add_argument("--head-arch", choices=["linear", "res_mlp"], default="linear")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-debias-letters", action="store_true")
    a = ap.parse_args(argv)

    debias = not a.no_debias_letters
    if a.task in ("ref_heads", "all"):
        for m in ("ref-qwen3vl8b", "ref-intern4b", "ref-intern2b"):
            m_dir = os.path.join(a.ezy_dir, m)
            if os.path.isdir(os.path.join(m_dir, "shards")):
                train_and_predict_model(m_dir, a.s1_data, a.root, seeds=a.seeds, arch=a.head_arch, device=a.device, debias_letters=debias)
    if a.task in ("rescore_legacy", "all"):
        for m in ("onejev", "jpt", "intern", "qwen", "vintern"):
            m_dir = os.path.join(a.ezy_dir, m)
            if os.path.isdir(os.path.join(m_dir, "shards")):
                train_and_predict_model(m_dir, a.s1_data, a.root, seeds=a.seeds, arch="linear", device=a.device, debias_letters=debias)
    if a.task in ("viso_v5", "all"):
        run_viso_v5(a.s1_data, a.viso_out)
    if a.task in ("bundle", "all"):
        bundle_lightweight_results([a.ezy_dir, a.viso_out], a.out_zip)


if __name__ == "__main__":
    main()
