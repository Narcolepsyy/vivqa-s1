"""Train ViSO-FT (live SigLIP 2 vision + BamiBERT towers, top blocks fine-tuned) and produce everything the comparison needs, resumable end to end.
  python scripts/train_viso_ft.py --out-dir ~/data/ft_run --name viso-ft-bami-s0 --seed 0 --s1-dir <vivqa_s1_v2/data> --image-roots <dir with the COCO images> --sha-list kaggle/s1_images_sha256.tsv \\
      --refs ABL_bami_s0=<run dir> A2_s0=<run dir> [--enc-lora 16] [--resume] [--limit 64 --epochs 2 --batch 8 --workers 0 --amp none --device cpu]   (smoke test)
Steps: (1) uint8 image cache (utils/image_u8_cache.py; skipped for --limit runs, which decode on the fly), (2) the datasets / model / task of configs/viso_ft_bami.yaml, (3) the train-set statistics that
initialise the learnable normalisation (frozen towers, once; skipped when a checkpoint exists), (4) training with per-epoch resumable state (TypedDecisionTask: last.pth every epoch, best-dev-NLL best_model.pth,
--resume continues), (5) logits of calib / dev / test, preds at T = 1, per-type temperature + escalation head on calib (scripts/viso_calibrate.py), (6) scripts/viso_ft_eval.py: paired bootstrap vs --refs
(dev + test) and the image-shuffle check. Printed and stored in ft_run.json: parameters trained vs total, seconds per epoch, peak GPU memory."""
import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default=os.path.join(REPO, "configs", "viso_ft_bami.yaml"))
    ap.add_argument("--out-dir", required=True, help="checkpoints + outputs go to <out-dir>/<name>")
    ap.add_argument("--name", default="viso-ft-bami-s0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s1-dir", default=os.path.join(REPO, "..", "vivqa_s1_v2", "data"))
    ap.add_argument("--image-roots", nargs="*", default=[os.path.join(REPO, "..", "vivqav2_dataset", "images")])
    ap.add_argument("--sha-list", default=os.path.join(REPO, "kaggle", "s1_images_sha256.tsv"))
    ap.add_argument("--cache-dir", default=None, help="uint8 image cache (default <out-dir>/image_cache)")
    ap.add_argument("--refs", nargs="*", default=[], help="NAME=<run dir with preds_dev.jsonl / preds_test.jsonl> for the paired bootstrap")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", default=None, help="bf16 | none (default: the config's FT.AMP = bf16)")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="first N items of every split (smoke test; images decoded on the fly)")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=-1)
    ap.add_argument("--enc-lora", type=int, default=0, help="LoRA rank in both towers instead of unfreezing blocks (optional variant, not the default)")
    ap.add_argument("--norm", default=None, help="stats | layernorm (FT.NORM)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[], help="extra KEY VALUE overrides")
    return ap.parse_args(argv)


def overrides(a):
    cache = a.cache_dir or os.path.join(a.out_dir, "image_cache")
    o = ["TRAINING.SEED", a.seed, "MODEL.NAME", a.name, "MODEL.DEVICE", a.device, "TRAINING.CHECKPOINT_PATH", a.out_dir, "TRAINING.OUTPUT_PATH", a.out_dir, "TRAINING.RESUME", bool(a.resume),
         "DATASET.VOCAB.S1_DIR", os.path.abspath(a.s1_dir), "DATASET.FEATURE_DATASET.LIVE.IMAGE_ROOTS", [os.path.abspath(os.path.expanduser(r)) for r in a.image_roots],
         "DATASET.FEATURE_DATASET.LIVE.SHA_LIST", os.path.abspath(os.path.expanduser(a.sha_list)), "DATASET.FEATURE_DATASET.LIVE.IMAGE_CACHE_DIR", cache]
    if a.amp is not None:
        o += ["MODEL.FT.AMP", a.amp]
    elif a.device == "cpu":
        o += ["MODEL.FT.AMP", "none"]
    if a.batch:
        o += ["DATASET.FEATURE_DATASET.BATCH_SIZE", a.batch]
    if a.workers >= 0:
        o += ["DATASET.FEATURE_DATASET.WORKERS", a.workers, "DATASET.FEATURE_DATASET.EVAL_WORKERS", min(a.workers, 4)]
    if a.limit:
        o += ["DATASET.FEATURE_DATASET.LIMIT", a.limit]
    if a.epochs:
        o += ["TRAINING.EPOCHS_SUP", a.epochs]
    if a.patience >= 0:
        o += ["TRAINING.PATIENCE", a.patience]
    if a.enc_lora:
        o += ["MODEL.FT.LORA_R", a.enc_lora]
    if a.norm:
        o += ["MODEL.FT.NORM", a.norm]
    return o + list(a.opts)


def fit_norm_stats(task, bs=256, log=print):
    """train-set statistics of the frozen towers -> the initial values of the learnable normalisation (once)"""
    from utils.viso_s1 import label_space
    train = task.datasets["train"]
    tok = train.tok
    qtexts = sorted({it["questions"]["q"]["instructions"] for it in train.items})
    ctexts = sorted({t for it in train.items if it["questions"]["q"]["type"] == "choice" for t in label_space(it["questions"]["q"])})

    def batches(texts):
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=48, return_tensors="pt")
            yield enc["input_ids"], enc["attention_mask"]
    t0 = time.time()
    stats = task.model.live.fit_stats(batches(qtexts), batches(ctexts), task.device)
    log(f"normalisation statistics from {len(qtexts)} train questions and {len(ctexts)} candidate strings in {time.time() - t0:.0f}s (mean |q_mu| {stats[0].abs().mean():.3f}, mean q_sd {stats[1].mean():.3f})")
    return stats


def main(argv=None):
    a = parse(argv)
    a.out_dir = os.path.abspath(os.path.expanduser(a.out_dir))
    os.makedirs(a.out_dir, exist_ok=True)
    import train_viso  # noqa: F401  (registers the vocab / dataset / model / task classes)
    from builders.task_builder import build_task
    from train_viso import load_config
    cfg = load_config(a.config_file, overrides(a))
    fd = cfg.DATASET.FEATURE_DATASET
    t_all = time.time()
    if not a.limit:                                                      # the full run: decode every image once into the uint8 memmap
        from data_utils.datasets.s1_live_dataset import all_coco_ids
        from utils.coco_images import ImageStore
        from utils.image_u8_cache import build_or_load
        store = ImageStore(fd.LIVE.SHA_LIST, roots=list(fd.LIVE.IMAGE_ROOTS))
        build_or_load(all_coco_ids(cfg.DATASET.VOCAB.S1_DIR), store, os.path.expanduser(fd.LIVE.IMAGE_CACHE_DIR), workers=max(1, min(int(fd.WORKERS) or 8, 16)))
    task = build_task(cfg)
    ckdir = task.checkpoint_path
    fresh = not any(os.path.isfile(os.path.join(ckdir, f)) for f in ("last.pth", "best_model.pth", "best_model_stage1.pth", "DONE.json"))
    if fresh:
        fit_norm_stats(task)
    rep = task.model.parameter_report()
    print("PARAMETERS: " + json.dumps(rep), flush=True)
    print(f"PARAMETERS trained {rep['params_trained'] / 1e6:.1f} M of {rep['params_total_incl_frozen_towers'] / 1e6:.1f} M total (towers: {rep['towers_trainable'] / 1e6:.1f} M of {rep['towers_total'] / 1e6:.1f} M trainable; "
          f"ViSO layers + normalisation: {rep['viso_layers_trainable'] / 1e6:.1f} M)", flush=True)
    task.start()
    task.get_predictions()
    # post-processing as the Kaggle runner does it: preds at T = 1 for dev / test, then per-type temperature + escalation on calib
    from utils import viso_calibration as C
    from utils.viso_s1 import read_jsonl
    for split in ("dev", "test"):
        lg = C.load_logits(os.path.join(task.output_path, f"logits_{split}.npz"))
        by_id = {it["id"]: it for it in read_jsonl(os.path.join(cfg.DATASET.VOCAB.S1_DIR, f"{split}.jsonl"))}
        C.write_preds_jsonl(os.path.join(task.output_path, f"preds_{split}.jsonl"), [by_id[int(i)] for i in lg["ids"]], C.probs_from_logits(lg["logits"], lg["space_mask"]))
    os.environ.setdefault("VIVQA_S1_ROOT", os.path.abspath(os.path.join(cfg.DATASET.VOCAB.S1_DIR, "..")))
    import viso_calibrate
    viso_calibrate.main(["--run-dir", task.output_path, "--s1-dir", cfg.DATASET.VOCAB.S1_DIR])
    log_rows = [json.loads(l) for l in open(task.log_file) if l.strip()]
    ep = [r for r in log_rows if "epoch" in r and "dev_nll" in r]
    summary = {"name": a.name, "params": rep, "epochs_logged": len(ep), "seconds_per_epoch_mean": round(float(np.mean([r["seconds"] for r in ep if "seconds" in r])), 1) if any("seconds" in r for r in ep) else None,
               "peak_gpu_gb": max([r["peak_gpu_gb"] for r in ep if r.get("peak_gpu_gb") is not None], default=None), "best_dev_nll": min(r["dev_nll"] for r in ep) if ep else None,
               "best_epoch": min(ep, key=lambda r: r["dev_nll"])["epoch"] if ep else None, "wall_seconds_this_session": round(time.time() - t_all), "config": os.path.join(task.output_path, "config.yaml")}
    json.dump(summary, open(os.path.join(task.output_path, "ft_run.json"), "w"), indent=1)
    print("FT RUN SUMMARY: " + json.dumps(summary), flush=True)
    if not a.no_eval:
        import viso_ft_eval
        viso_ft_eval.run(task, dict(r.split("=", 1) for r in a.refs))
    return summary


if __name__ == "__main__":
    main()
