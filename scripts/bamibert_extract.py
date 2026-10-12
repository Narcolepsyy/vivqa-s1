"""Cache frozen BamiBERT features of every question and every choice-candidate string of ViVQA-S1 (Phase 2c text side).
    python scripts/bamibert_extract.py [--device cpu] [--limit N] [--verify]
Model: Qualcomm-AI-Research/BamiBERT (RoBERTa-style, 12 layers, hidden 768, ~103 M parameters incl. the MLM head, no word segmentation needed),
pinned revision below. Licence: BSD-3-Clause-Clear + Qualcomm Responsible AI licence, research / educational use; cite arXiv:2607.02259. No custom code (standard
`roberta` architecture, safetensors): trust_remote_code is NOT used.
Output (default ~/data/bamibert_cache), everything fp16, keyed by the sha1 of the exact text so the cache can be extended when a split's text changes (A2 / S1-Verified):
  q_states.bin    float16 [total_tokens, 768]   last-layer token states of every unique question (<s> ... </s>, at most MAX_TOK tokens)
  q_index.npz     key (sha1 hex, sorted), offset, length
  c_vecs.npy      float16 [n_candidates, 2, 768]  per unique candidate string: [0] mean over its tokens, [1] the <s> (CLS) state
  c_index.npz     key (sha1 hex, sorted)
  meta.json, verify.json (re-extraction of a random sample: max abs diff)
Re-running only adds the texts that are not cached yet."""
import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL_ID = "Qualcomm-AI-Research/BamiBERT"          # defaults = BamiBERT; --model / --revision / --licence select any other standard (no remote code) encoder
REVISION = "57bc1340debbe4e348ec549047a763caebe4a977"
LICENCE = "BSD-3-Clause-Clear + Qualcomm Responsible AI licence (research / education)"
DIM, MAX_TOK = 768, 48
BACKEND = "hf"                                   # "hf": any standard encoder (BamiBERT ...); "siglip2-text": the frozen text tower of google/siglip2-base-patch16-256 (T-SIG)
LAST_POOLED = None                               # siglip2-text: the pooled (image-aligned) text embeddings of the last encode() call
QBIN, QIDX, CVEC, CIDX, META, VERIFY = "q_states.bin", "q_index.npz", "c_vecs.npy", "c_index.npz", "meta.json", "verify.json"


def key_of(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def s1_texts(s1_dir, extra=()):
    """-> (set of question texts, set of candidate strings) over train / calib / dev / test"""
    from utils.viso_s1 import label_space, read_jsonl
    qs, cs = set(), set()
    files = [os.path.join(s1_dir, f"{sp}.jsonl") for sp in ("train", "calib", "dev", "test")] + [os.path.expanduser(e) for e in extra]          # extra: translated extra-train files (X-YN)
    for path in files:
        for it in read_jsonl(path):
            q = it["questions"]["q"]
            qs.add(q["instructions"])
            if q["type"] == "choice":
                cs.update(label_space(q))
    return qs, cs


def load_model(device, model_id=None, revision=None, tokenizer_json=False):
    if BACKEND == "siglip2-text":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(model_id, revision=revision, dtype=torch.float32)
        tower = model.text_model.eval().to(device)                          # no remote code: standard transformers architecture
        for p in tower.parameters():
            p.requires_grad_(False)
        tower.total_params = sum(p.numel() for p in model.text_model.parameters())
        tower.embedding_params = sum(p.numel() for p in model.text_model.embeddings.token_embedding.parameters())
        del model
        return tower, AutoTokenizer.from_pretrained(model_id, revision=revision)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, PreTrainedTokenizerFast
    # tokenizer_config.json names XLMRobertaTokenizer, which transformers >= 5.6 cannot build from this repo (the card says transformers <= 5.5.0): load the shipped
    # tokenizer.json (byte-level BPE; <s>=0 <pad>=1 </s>=2) directly, with the special tokens of special_tokens_map.json
    model_id, revision = model_id or MODEL_ID, revision or REVISION
    if tokenizer_json:
        tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(model_id, "tokenizer.json", revision=revision), bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                      sep_token="</s>", pad_token="<pad>", cls_token="<s>", mask_token="<mask>")
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision, dtype=torch.float32, add_pooling_layer=False).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@torch.no_grad()
def encode(model, tok, texts, device):
    """-> list of float32 (len_i, 768) numpy arrays (one per text, <s> ... </s>, no padding)"""
    global LAST_POOLED
    if BACKEND == "siglip2-text":                                           # padded to 64 tokens with lower-cased text, input_ids only: how the tower was trained
        b = tok([t.lower() for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
        o = model(input_ids=b["input_ids"])
        h = o.last_hidden_state.float().cpu().numpy()
        LAST_POOLED = o.pooler_output.float().cpu().numpy()
        n = (b["input_ids"] != tok.pad_token_id).sum(1).clamp(min=1).cpu().numpy()
        return [h[i, :n[i]] for i in range(len(texts))]
    b = tok(texts, padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt").to(device)
    h = model(**b).last_hidden_state.float().cpu().numpy()
    n = b["attention_mask"].sum(1).cpu().numpy()
    return [h[i, :n[i]] for i in range(len(texts))]


def read_cache(out):
    """-> dict with the cached keys (empty if none)"""
    if not os.path.isfile(os.path.join(out, QIDX)):
        return {"q_key": np.array([], dtype="U40"), "q_off": np.zeros(0, np.int64), "q_len": np.zeros(0, np.int32), "c_key": np.array([], dtype="U40"),
                "c_vec": np.zeros((0, 2, DIM), np.float16), "q_data": np.zeros((0, DIM), np.float16)}
    qi, ci = np.load(os.path.join(out, QIDX)), np.load(os.path.join(out, CIDX))
    n_tok = int((qi["offset"] + qi["length"]).max()) if len(qi["key"]) else 0
    return {"q_key": qi["key"], "q_off": qi["offset"], "q_len": qi["length"], "c_key": ci["key"], "c_vec": np.load(os.path.join(out, CVEC)),
            "q_data": np.fromfile(os.path.join(out, QBIN), dtype="<f2").reshape(-1, DIM)[:n_tok]}


def main(argv=None):
    global MAX_TOK, BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/bamibert_cache")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--licence", default=LICENCE)
    ap.add_argument("--tokenizer-json", action="store_true", default=None, help="load tokenizer.json directly (BamiBERT: its tokenizer_config names a class that transformers >= 5.6 cannot build)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOK)
    ap.add_argument("--extra-jsonl", nargs="*", default=[], help="extra train jsonl files (e.g. vivqa_s1_v2/extra/xyn_partial_train.jsonl): their question / candidate texts are added to the cache")
    ap.add_argument("--backend", choices=("hf", "siglip2-text"), default="hf")
    ap.add_argument("--s1-dir", default=os.path.join(REPO, "..", "vivqa_s1_v2", "data"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N texts of each kind (tests)")
    ap.add_argument("--verify", action="store_true", help="re-extract 200 random cached questions and compare")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out, exist_ok=True)
    qs, cs = s1_texts(a.s1_dir, a.extra_jsonl)
    qs, cs = sorted(qs), sorted(cs)
    if a.limit:
        qs, cs = qs[:a.limit], cs[:a.limit]
    cache = read_cache(out)
    have_q, have_c = set(cache["q_key"].tolist()), set(cache["c_key"].tolist())
    new_q = [t for t in qs if key_of(t) not in have_q]
    new_c = [t for t in cs if key_of(t) not in have_c]
    print(f"questions: {len(qs)} ({len(new_q)} new) | candidate strings: {len(cs)} ({len(new_c)} new)", flush=True)
    MAX_TOK = a.max_tokens
    BACKEND = a.backend
    tj = a.tokenizer_json if a.tokenizer_json is not None else (a.model == MODEL_ID)
    model, tok = load_model(a.device, a.model, a.revision, tj)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    # --- questions (sorted by length: little padding)
    order = sorted(new_q, key=len)
    keys, lens, chunks = [], [], []
    for i in range(0, len(order), a.batch):
        batch = order[i:i + a.batch]
        for t, h in zip(batch, encode(model, tok, batch, a.device)):
            keys.append(key_of(t)); lens.append(len(h)); chunks.append(h.astype(np.float16))
        if (i // a.batch) % 50 == 0:
            print(f"  questions {i + len(batch)}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    base = len(cache["q_data"])
    q_key = np.concatenate([cache["q_key"], np.array(keys, dtype="U40")]) if keys else cache["q_key"]
    q_len = np.concatenate([cache["q_len"], np.array(lens, np.int32)]) if keys else cache["q_len"]
    q_off = np.concatenate([cache["q_off"], base + np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)]) if keys else cache["q_off"]
    q_data = np.concatenate([cache["q_data"]] + chunks) if chunks else cache["q_data"]
    sel = np.argsort(q_key)
    # --- candidates: mean over the real tokens (without <s> </s>) and the <s> state
    c_new_keys, c_new_vec = [], []
    for i in range(0, len(new_c), a.batch):
        batch = new_c[i:i + a.batch]
        for j, (t, h) in enumerate(zip(batch, encode(model, tok, batch, a.device))):
            if BACKEND == "siglip2-text":                                    # [0] mean of the token states, [1] the pooled text embedding (aligned with the image features)
                vec = np.stack([h.mean(0), LAST_POOLED[j]])
            else:
                body = h[1:-1] if len(h) > 2 else h
                vec = np.stack([body.mean(0), h[0]])
            c_new_keys.append(key_of(t)); c_new_vec.append(vec.astype(np.float16))
    c_key = np.concatenate([cache["c_key"], np.array(c_new_keys, dtype="U40")]) if c_new_keys else cache["c_key"]
    c_vec = np.concatenate([cache["c_vec"], np.stack(c_new_vec)]) if c_new_vec else cache["c_vec"]
    csel = np.argsort(c_key)
    assert np.isfinite(q_data.astype(np.float32)).all() and np.isfinite(c_vec.astype(np.float32)).all(), "non-finite values in the cache"
    q_data.astype("<f2").tofile(os.path.join(out, QBIN + ".tmp"))
    os.replace(os.path.join(out, QBIN + ".tmp"), os.path.join(out, QBIN))
    np.savez(os.path.join(out, QIDX), key=q_key[sel], offset=q_off[sel], length=q_len[sel])
    np.save(os.path.join(out, CVEC), c_vec[csel])
    np.savez(os.path.join(out, CIDX), key=c_key[csel])
    meta = {"model": a.model, "revision": a.revision, "dim": DIM, "max_tokens": MAX_TOK, "n_params_loaded_encoder": n_params, "backend": a.backend, **({"text_tower_params": model.total_params, "token_embedding_table_params": model.embedding_params, "pooled_index": 1, "note": "c_vecs[:,1] = pooled text embedding (cls pool in the config), [:,0] = mean of token states"} if a.backend == "siglip2-text" else {}),
            "n_questions": int(len(q_key)), "n_candidates": int(len(c_key)), "total_tokens": int(len(q_data)), "licence": a.licence,
            "compute_dtype": "float32", "store_dtype": "float16", "seconds": round(time.time() - t0, 1),
            "sha256_q_states": hashlib.sha256(open(os.path.join(out, QBIN), "rb").read()).hexdigest()}
    json.dump(meta, open(os.path.join(out, META), "w"), indent=1)
    print("wrote", out, {k: meta[k] for k in ("n_questions", "n_candidates", "total_tokens", "seconds")}, flush=True)
    if a.verify:
        rng = np.random.default_rng(0)
        pick = [qs[i] for i in rng.choice(len(qs), size=min(200, len(qs)), replace=False)]
        cache = read_cache(out)
        pos = {k: i for i, k in enumerate(cache["q_key"].tolist())}
        diffs = []
        for i in range(0, len(pick), a.batch):
            batch = pick[i:i + a.batch]
            for t, h in zip(batch, encode(model, tok, batch, a.device)):
                j = pos[key_of(t)]
                diffs.append(float(np.abs(cache["q_data"][cache["q_off"][j]:cache["q_off"][j] + cache["q_len"][j]].astype(np.float32) - h).max()))
        rep = {"n_checked": len(pick), "max_abs_diff_vs_fp16_store": max(diffs), "finite": True}
        json.dump(rep, open(os.path.join(out, VERIFY), "w"), indent=1)
        print("verify", rep, flush=True)
    return out


if __name__ == "__main__":
    main()
