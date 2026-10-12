"""Temperature scaling + escalation head from dumped logits (no forward pass).
    python scripts/viso_calibrate.py --run-dir output/viso/<run> [--s1-dir ../vivqa_s1_v2/data]
Reads logits_{calib,dev,test}.npz from the run dir; writes calibration/ with
  temperatures.json                      per-type T (NLL vs gold_label, primary), one global T, per-type T on the soft gold
  preds_<split>_{uncal,pertype,global,softtype}.jsonl   (pertype = the calibrated predictions used for reporting)
  calibration_metrics.json               s1bench primary / S1-Human metrics of every variant on dev and test
  escalation.json                        out-of-fold AUROC / AURC / selective accuracy on calib, then dev and test,
                                         for the learned head vs the baseline score max_prob
Temperatures and the escalation head are fitted on CALIB only."""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from utils import viso_calibration as C  # noqa: E402
from utils.atomic import atomic_json  # noqa: E402
from utils.viso_s1 import read_jsonl  # noqa: E402
from s1bench.metrics import evaluate  # noqa: E402

VARIANTS = ("uncal", "pertype", "global", "softtype")


def load_split(run_dir, s1_dir, split):
    lg = C.load_logits(os.path.join(run_dir, f"logits_{split}.npz"))
    by_id = {it["id"]: it for it in read_jsonl(os.path.join(s1_dir, f"{split}.jsonl"))}
    items = [by_id[int(i)] for i in lg["ids"]]
    return lg, items, C.gold_arrays(items)


def preds_of(items, probs):
    from tasks.typed_decision_task import preds_dict
    return preds_dict(items, probs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--s1-dir", default=os.path.join(REPO, "..", "vivqa_s1_v2", "data"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    out = os.path.join(args.run_dir, "calibration")
    os.makedirs(out, exist_ok=True)

    lg_c, items_c, gold_c = load_split(args.run_dir, args.s1_dir, "calib")
    t_type = C.fit_temperatures(lg_c["logits"], lg_c["space_mask"], lg_c["type_id"], gold_idx=gold_c["gold_idx"])
    t_global = C.fit_temperature(lg_c["logits"], lg_c["space_mask"], gold_idx=gold_c["gold_idx"])
    t_soft = C.fit_temperatures(lg_c["logits"], lg_c["space_mask"], lg_c["type_id"], target=gold_c["target"])
    temps = {"per_type_nll": {str(k): v for k, v in t_type.items()}, "global_nll": t_global,
             "per_type_soft": {str(k): v for k, v in t_soft.items()}, "type_ids": {"noul": 0, "score": 1, "choice": 2},
             "n_calib": int(len(items_c)), "fitted_on": "calib"}
    atomic_json(os.path.join(out, "temperatures.json"), temps)

    settings = {"uncal": None, "pertype": t_type, "global": t_global, "softtype": t_soft}
    metrics, probs_by = {}, {}
    for split in ("calib", "dev", "test"):
        lg, items, gold = (lg_c, items_c, gold_c) if split == "calib" else load_split(args.run_dir, args.s1_dir, split)
        for v, t in settings.items():
            p = C.probs_from_logits(lg["logits"], lg["space_mask"], lg["type_id"], t)
            probs_by[(split, v)] = p
            if split != "calib":
                C.write_preds_jsonl(os.path.join(out, f"preds_{split}_{v}.jsonl"), items, p)
            metrics.setdefault(split, {})[v] = evaluate(items, preds_of(items, p))
    atomic_json(os.path.join(out, "calibration_metrics.json"), metrics)

    # ---- escalation head (fitted on calibrated calib probabilities; per-type T = the reporting variant)
    esc = {"features": ["max_prob", "entropy/log|space|", "top1-top2 margin", "type one-hot(3)", "max_prob*type one-hot(3)"],
           "folds": args.folds, "label": "error = (gold_label not in the argmax set, fractional ties < 0.5)"}
    golds = {"calib": gold_c}
    for split in ("dev", "test"):
        golds[split] = load_split(args.run_dir, args.s1_dir, split)[2]

    def feats_and_error(split, lg):
        p = probs_by[(split, "pertype")]
        correct = C.correct_fraction(p, golds[split]["gold_idx"])
        return C.escalation_features(p, lg["space_mask"], lg["type_id"]), (correct < 0.5).astype(float), p, correct

    x_c, e_c, p_c, corr_c = feats_and_error("calib", lg_c)
    oof = C.oof_predict(x_c, e_c, args.folds, seed=0)
    esc["calib_oof"] = {"auroc_head": C.auroc(oof, e_c), "auroc_max_prob": C.auroc(1.0 - p_c.max(1), e_c),
                        "error_rate": float(e_c.mean()),
                        "selective_head": C.selective_metrics(1.0 - oof, corr_c),
                        "selective_max_prob": C.selective_metrics(p_c.max(1), corr_c)}
    wb = C.fit_logreg(x_c, e_c)
    for split in ("dev", "test"):
        lg = C.load_logits(os.path.join(args.run_dir, f"logits_{split}.npz"))
        x, e, p, corr = feats_and_error(split, lg)
        score = C.predict_logreg(wb, x)
        esc[split] = {"auroc_head": C.auroc(score, e), "auroc_max_prob": C.auroc(1.0 - p.max(1), e),
                      "error_rate": float(e.mean()), "selective_head": C.selective_metrics(1.0 - score, corr),
                      "selective_max_prob": C.selective_metrics(p.max(1), corr)}
    atomic_json(os.path.join(out, "escalation.json"), esc)
    print(json.dumps({"temperatures": temps, "dev_ece": {v: metrics["dev"][v]["primary"]["all"]["ece"] for v in VARIANTS}}))
    return out


if __name__ == "__main__":
    main()
