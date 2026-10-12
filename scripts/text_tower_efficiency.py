"""Parameters, batch-1 FLOPs and CPU latency of the FROZEN TEXT encoders that run once per QUESTION on the device (the efficiency tables must count them: the cached features are only a training
shortcut). T-SIG = the text tower of google/siglip2-base-patch16-256 (282.3 M parameters, 196.6 M of them the 256k x 768 token-embedding table); BamiBERT (ablation) = Qualcomm-AI-Research/BamiBERT.
  python scripts/text_tower_efficiency.py [--out kaggle/text_tower_efficiency.json] [--runs 100]
Input: one question, random token ids. SigLIP 2's text tower is padded to 64 tokens as it was trained and as scripts/bamibert_extract.py runs it (FLOPs also given for a 20-token question when the
padding is dropped, which that tower does not allow in the published recipe); BamiBERT runs unpadded (20 tokens = utils/viso_efficiency.N_QTOKENS). Latency: THIS machine's CPU, 1 and 4 threads, fp32."""
import argparse
import json
import os
import platform
import statistics
import subprocess
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

SIGLIP = ("google/siglip2-base-patch16-256", "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab")
BAMI = ("Qualcomm-AI-Research/BamiBERT", "57bc1340debbe4e348ec549047a763caebe4a977")
N_Q = 20


def measure(fn, ids, runs, warmup=10):
    fn(ids)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn(ids)
    out = {"flops_batch1": int(fc.get_total_flops())}
    for threads in (1, 4):
        prev = torch.get_num_threads()
        torch.set_num_threads(threads)
        with torch.no_grad():
            for _ in range(warmup):
                fn(ids)
            ts = []
            for _ in range(runs):
                t = time.perf_counter()
                fn(ids)
                ts.append((time.perf_counter() - t) * 1e3)
        torch.set_num_threads(prev)
        ts.sort()
        out[f"cpu_latency_ms_{threads}_threads"] = {"median": round(statistics.median(ts), 2), "p95": round(ts[int(0.95 * (len(ts) - 1))], 2)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "kaggle", "text_tower_efficiency.json"))
    ap.add_argument("--runs", type=int, default=100)
    a = ap.parse_args(argv)
    from transformers import AutoModel
    torch.manual_seed(0)
    res = {}
    m = AutoModel.from_pretrained(SIGLIP[0], revision=SIGLIP[1], dtype=torch.float32)
    tower = m.text_model.eval()
    emb = sum(p.numel() for p in tower.embeddings.token_embedding.parameters())
    total = sum(p.numel() for p in tower.parameters())
    ids64 = torch.randint(3, 250000, (1, 64))
    ids20 = torch.randint(3, 250000, (1, N_Q))
    run = lambda x: tower(input_ids=x)
    res["siglip2_text_tower"] = {"model": SIGLIP[0], "revision": SIGLIP[1], "params_total": total, "token_embedding_table_params": emb, "params_without_embedding_table": total - emb,
                                 "input": "1 x 64 tokens (padded to 64, as trained and as bamibert_extract.py runs it)", **measure(run, ids64, a.runs),
                                 "flops_batch1_20_tokens_unpadded": measure(run, ids20, 3, warmup=1)["flops_batch1"], "frozen": True, "once_per_question": True}
    del m, tower
    b = AutoModel.from_pretrained(BAMI[0], revision=BAMI[1], dtype=torch.float32, add_pooling_layer=False).eval()
    emb_b = sum(p.numel() for p in b.embeddings.word_embeddings.parameters())
    total_b = sum(p.numel() for p in b.parameters())
    idsb = torch.randint(3, b.config.vocab_size, (1, N_Q))
    res["bamibert"] = {"model": BAMI[0], "revision": BAMI[1], "params_total": total_b, "token_embedding_table_params": emb_b, "params_without_embedding_table": total_b - emb_b,
                       "input": f"1 x {N_Q} tokens (unpadded)", **measure(lambda x: b(input_ids=x), idsb, a.runs), "frozen": True, "once_per_question": True}
    try:
        cpu = [ln.split(":", 1)[1].strip() for ln in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines() if ln.startswith("Model name")][0]
    except Exception:
        cpu = platform.processor()
    res["machine"] = {"cpu": cpu, "torch": torch.__version__, "python": platform.python_version(), "note": "latency measured on this local CPU (fp32, batch 1); same machine class as the vision-tower latency in vision_tower_efficiency.json"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
