"""Dump raw (temperature-free) logits and predictions of a trained ViSO / FlatS1 run.
    python scripts/viso_predict.py --config-file configs/viso_vivqav2.yaml --opts MODEL.NAME my_run ... [--splits calib dev test]
Writes, to output/viso/<MODEL.NAME>/: logits_<split>.npz (ids, type_id, logits[N,16], space_mask) so calibration never
needs another forward pass, and preds_<split>.jsonl ({"id", "probabilities"} keyed by each item's label keys; the input
format of vivqa_s1_v2/evaluate.py) at T = 1."""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from train_viso import load_config  # noqa: E402  (also registers the new modules)
from builders.task_builder import build_task  # noqa: E402
from tasks.typed_decision_task import collect_logits  # noqa: E402
from utils import viso_calibration as calib  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--splits", nargs="*", default=["calib", "dev", "test"])
    ap.add_argument("--checkpoint", default="best_model.pth")
    args = ap.parse_args(argv)
    task = build_task(load_config(args.config_file, args.opts))
    if task.load_checkpoint(args.checkpoint) is None:
        raise FileNotFoundError(os.path.join(task.checkpoint_path, args.checkpoint))
    for split in args.splits:
        ds = task.datasets[split]
        lg = collect_logits(task.model, ds, task.batch_size, task.device, int(getattr(task.config.DATASET.FEATURE_DATASET, "EVAL_WORKERS", 0)))
        calib.save_logits(os.path.join(task.output_path, f"logits_{split}.npz"), **lg)
        probs = calib.probs_from_logits(lg["logits"], lg["space_mask"])
        calib.write_preds_jsonl(os.path.join(task.output_path, f"preds_{split}.jsonl"), ds.items, probs)
        print(f"{split}: {len(ds)} items -> {task.output_path}")
    return task.output_path


if __name__ == "__main__":
    main()
