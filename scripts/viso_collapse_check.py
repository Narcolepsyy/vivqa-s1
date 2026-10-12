"""Does the fused vector h of ViSO depend on the input? Prints, for 512 dev items: mean pairwise cosine of h, std of h per dim (relative to its norm), and how much of the
variance of h is explained by the question alone / the image alone (permuting images across items keeps the question, permuting questions keeps the image).
    python scripts/viso_collapse_check.py [--run-dir ~/data/viso_runs_v2/runs/A2_s1]"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
import torch  # noqa: E402

from train_viso import load_config  # noqa: E402
from builders.task_builder import build_task  # noqa: E402
from data_utils.datasets.s1_dataset import s1_collate  # noqa: E402


def stats(h, name):
    hn = torch.nn.functional.normalize(h, dim=-1)
    cos = (hn @ hn.T)[~torch.eye(len(h), dtype=torch.bool)].mean().item()
    print(f"{name:34s} mean pairwise cos {cos:.3f} | per-dim std / mean-norm {h.std(0).mean().item() / h.norm(dim=-1).mean().item():.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args(argv)
    work = "/tmp/viso_collapse"
    cfg_file = os.path.join(a.run_dir, "run_config.yaml") if a.run_dir else os.path.join(REPO, "configs", "viso_siglip2.yaml")
    opts = ["MODEL.NAME", "collapse", "TRAINING.OUTPUT_PATH", work, "TRAINING.CHECKPOINT_PATH", work, "MODEL.DEVICE", "cuda"] + a.opts
    task = build_task(load_config(cfg_file, opts))
    ds = task.datasets["dev"]
    batch = s1_collate([ds[i] for i in range(0, len(ds), max(1, len(ds) // 512))][:512]).to(task.device)
    for tag in ("init", "trained"):
        if tag == "trained":
            if not a.run_dir or not os.path.isfile(os.path.join(a.run_dir, "best_model.pth")):
                break
            task.model.load_state_dict(torch.load(os.path.join(a.run_dir, "best_model.pth"), map_location=task.device, weights_only=True)["state_dict"])
        task.model.eval()
        with torch.no_grad():
            h, _ = task.model.encoder(batch)
            perm = torch.randperm(len(h), generator=torch.Generator().manual_seed(0)).to(task.device)
            import copy
            b_img = copy.copy(batch); b_img["region_features"] = batch.region_features[perm]; b_img["region_mask"] = batch.region_mask[perm]
            if "region_boxes" in batch:
                b_img["region_boxes"] = batch.region_boxes[perm]
            b_q = copy.copy(batch); b_q["question_tokens"] = batch.question_tokens[perm]
            h_img, _ = task.model.encoder(b_img); h_q, _ = task.model.encoder(b_q)
        stats(h, f"[{tag}] h"); 
        print(f"[{tag}] relative change of h when only the IMAGE is swapped: {((h - h_img).norm(dim=-1) / h.norm(dim=-1)).mean().item():.3f}; when only the QUESTION is swapped: {((h - h_q).norm(dim=-1) / h.norm(dim=-1)).mean().item():.3f}")


if __name__ == "__main__":
    main()
