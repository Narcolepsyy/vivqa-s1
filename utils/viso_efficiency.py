"""Efficiency proxies for ViSO vs FlatS1: parameters, checkpoint size, FLOPs (batch 1) and CPU/GPU latency.
    python -m utils.viso_efficiency --config-file configs/viso_vivqav2.yaml --opts MODEL.DEVICE cpu --out output/viso/eff.json
NOTE: the VinVL region extractor is NOT included (not on-device feasible): this is the cost of the ViSO network on
precomputed region features, not of the features. Inputs are seeded random tensors: 44 regions, 20 question tokens,
16 candidates (choice item, the heaviest type)."""
import argparse
import json
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

N_REGIONS, N_QTOKENS, N_CAND, CAND_LEN = 44, 20, 16, 6


def make_batch(vocab, type_id=2, seed=0, device="cpu", n_regions=N_REGIONS, d_feature=2048, boxes=False, cached_text=False):
    """n_regions / d_feature / boxes default to the VinVL setting (44 regions of 2048); SigLIP 2 configs use 65 tokens of 768 plus the patch-cell boxes."""
    from data_utils.datasets.s1_dataset import S1Batch
    g = torch.Generator().manual_seed(seed)
    n_words = len(vocab)
    q = torch.full((1, vocab.max_question_length), vocab.padding_idx, dtype=torch.long)
    q[0, :N_QTOKENS] = torch.randint(4, n_words, (N_QTOKENS,), generator=g)
    space = {0: 2, 1: 11, 2: N_CAND}[type_id]
    space_mask = torch.zeros(1, 16, dtype=torch.bool)
    space_mask[0, :space] = True
    cand = torch.randint(4, n_words, (1, 16, CAND_LEN), generator=g)
    flat_index = torch.arange(16)[None].clamp(max=vocab.total_answers - 1)
    return S1Batch(
        id=torch.zeros(1, dtype=torch.long), region_features=torch.randn(1, n_regions, d_feature, generator=g),
        region_mask=torch.ones(1, n_regions, dtype=torch.bool), question_tokens=q,
        type_id=torch.tensor([type_id]), space_mask=space_mask, target=space_mask.float() / space,
        gold_idx=torch.zeros(1, dtype=torch.long), human_flag=torch.zeros(1, dtype=torch.bool),
        cand_tokens=cand, cand_mask=space_mask & (type_id == 2), flat_index=flat_index,
        **({"region_boxes": torch.rand(1, n_regions, 4, generator=g)} if boxes else {}),
        # cached-text-encoder configs (T-SIG / BamiBERT): frozen question token states (L x 768, as a 20-token question) and pooled candidate vectors
        **({"question_states": torch.randn(1, N_QTOKENS, 768, generator=g), "question_state_mask": torch.ones(1, N_QTOKENS, dtype=torch.bool),
            "cand_vecs": torch.randn(1, 16, 768, generator=g)} if cached_text else {})).to(device)


def param_counts(model):
    enc = sum(p.numel() for n, p in model.encoder.named_parameters() if not n.startswith("classify."))
    total = sum(p.numel() for p in model.parameters())          # shared tensors are counted once
    out = {"encoder": enc, "total": total, "heads_and_classifier": total - enc}
    for name in ("noul_head", "score_head", "choice_head"):
        if hasattr(model, name):
            out[name] = sum(p.numel() for p in getattr(model, name).parameters())
    if hasattr(model.encoder, "classify"):
        out["flat_classifier"] = sum(p.numel() for p in model.encoder.classify.parameters())
    out["checkpoint_mb_fp32"] = round(total * 4 / 2 ** 20, 2)
    out["checkpoint_mb_fp16"] = round(total * 2 / 2 ** 20, 2)
    return out


def forward_flops(model, batch):
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(batch)
    return int(fc.get_total_flops())


def latency_ms(model, batch, threads, warmup=20, runs=200):
    model.eval()
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        for _ in range(runs):
            t = time.perf_counter()
            model(batch)
            times.append((time.perf_counter() - t) * 1e3)
    torch.set_num_threads(prev)
    times.sort()
    return {"median_ms": round(statistics.median(times), 3), "p95_ms": round(times[int(0.95 * (len(times) - 1))], 3)}


def input_shape(cfg_model):
    """(n_regions, d_feature, boxes) of a model config: VinVL = 44 x 2048, SigLIP 2 (D_FEATURE 768) = 65 x 768 + boxes"""
    ve = cfg_model.VISION_EMBEDDING
    d = int(ve.D_FEATURE)
    return (65 if d == 768 else N_REGIONS), d, bool(getattr(ve, "USE_BOXES", False))


def profile(model, vocab, type_id=2, threads=(1, 4), warmup=20, runs=200, shape=None):
    n_reg, d_feat, use_boxes = shape or (N_REGIONS, 2048, False)
    batch = make_batch(vocab, type_id, n_regions=n_reg, d_feature=d_feat, boxes=use_boxes)
    res = {"params": param_counts(model), "flops_batch1": forward_flops(model, batch), "type_id": type_id,
           "input": {"regions": n_reg, "feature_dim": d_feat, "boxes": use_boxes, "question_tokens": N_QTOKENS, "candidates": N_CAND},
           "cpu_latency": {f"{t}_threads": latency_ms(model, batch, t, warmup, runs) for t in threads},
           "note": "the vision extractor (VinVL, or the frozen SigLIP 2 tower) is NOT included: its cost is reported separately, once per image"}
    if torch.cuda.is_available():
        m, b = model.to("cuda"), make_batch(vocab, type_id, device="cuda", n_regions=n_reg, d_feature=d_feat, boxes=use_boxes)
        for _ in range(warmup):
            m(b)
        torch.cuda.synchronize()
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            m(b)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
        ts.sort()
        res["gpu_latency"] = {"median_ms": round(statistics.median(ts), 3), "p95_ms": round(ts[int(0.95 * (len(ts) - 1))], 3)}
        model.to("cpu")
    return res


def main(argv=None):
    import data_utils.vocabs.s1_vocab  # noqa: F401
    import models.viso  # noqa: F401
    from builders.model_builder import build_model
    from builders.vocab_builder import build_vocab
    from train_viso import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", nargs="+", required=True, help="one or more configs (e.g. viso and flat)")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args(argv)
    results = {}
    for cf in args.config_file:
        cfg = load_config(cf, args.opts + ["MODEL.DEVICE", "cpu"])
        torch.manual_seed(0)
        vocab = build_vocab(cfg.DATASET.VOCAB)
        model = build_model(cfg.MODEL, vocab)
        results[cfg.MODEL.ARCHITECTURE] = profile(model, vocab, warmup=args.warmup, runs=args.runs, shape=input_shape(cfg.MODEL))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()
