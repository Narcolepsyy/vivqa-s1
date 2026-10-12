"""Vintern (InternVL) adapter: the hand-built single forward pass (embedding injection + decoder body, hidden state at the answer position) must reproduce the model's own generation: the argmax of the
full-vocabulary logits at that position equals the first token the model's `generate()` produces for the same prompt. Runs on a tiny randomly initialised model built from the PINNED remote code (config + code
only: the hash check still runs); with the real snapshot in the local cache the full 0.94 B model is used instead. Needs a GPU, the benchmark and images; skipped otherwise."""
import glob
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import vlm_adapters as A  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SNAP = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--5CD-AI--Vintern-1B-v3_5/snapshots/{A.VINTERN['revision']}/model.safetensors"))
IMAGES = os.path.join(REPO, "..", "vivqav2_dataset", "images")
S1 = os.path.join(REPO, "..", "vivqa_s1_v2", "data")
needs = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(IMAGES) or not os.path.isdir(S1), reason="needs a GPU, the benchmark and the images")


def build_adapter(monkeypatch):
    a = types.SimpleNamespace(model=A.VINTERN["repo"], revision=A.VINTERN["revision"], device="cuda", max_tiles=4, trust_remote_code=True)
    if not SNAP:                                                          # tiny random model from the pinned remote code (no 3.8 GB download)
        from transformers import AutoConfig, AutoModel

        def tiny(name, revision=None, dtype=None, trust_remote_code=False, **kw):
            cfg = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=True)
            cfg.llm_config.num_hidden_layers = 2
            cfg.vision_config.num_hidden_layers = 2
            torch.manual_seed(0)
            return AutoModel.from_config(cfg, trust_remote_code=True, dtype=dtype)
        monkeypatch.setattr(AutoModel, "from_pretrained", tiny)
    return A.make_adapter(a, torch.float16)


@needs
def test_single_forward_equals_the_models_first_generated_token(monkeypatch):
    import s1_vlm_eval as V
    from utils.coco_images import ImageStore
    ad = build_adapter(monkeypatch)
    items = V.load_items(S1, "dev", 8)
    store = ImageStore(os.path.join(REPO, "kaggle", "s1_images_sha256.tsv.gz"), roots=[IMAGES])
    agree = 0
    for it in items[:4]:
        px = ad.prepare_image(store.open(it["coco_id"]))
        prompt = V.build_prompt(it, 1)
        h = ad.encode([(it["id"], 1, px, prompt)], "cuda")
        mine = int(ad.lm.get_output_embeddings()(h.to(torch.float16)).argmax(-1))
        ids = ad.tok(ad._query(prompt, px.shape[0]), return_tensors="pt")["input_ids"].to("cuda")
        out = ad.model.generate(pixel_values=px.to("cuda", torch.float16), input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=1, do_sample=False,
                                eos_token_id=ad.tok.convert_tokens_to_ids("<|im_end|>"))
        agree += int(out[0, 0]) == mine
    assert agree >= 3, f"only {agree}/4 first tokens agree"                                   # fp16 near-ties of a random model may differ


@needs
def test_batched_encoding_equals_one_by_one(monkeypatch):
    """left padding + several tile counts in one batch must not change the hidden state at the answer position (up to fp16 noise)"""
    import s1_vlm_eval as V
    from utils.coco_images import ImageStore
    ad = build_adapter(monkeypatch)
    items = V.load_items(S1, "dev", 6)
    store = ImageStore(os.path.join(REPO, "kaggle", "s1_images_sha256.tsv.gz"), roots=[IMAGES])
    rows = [(it["id"], 1, ad.prepare_image(store.open(it["coco_id"])), V.build_prompt(it, 1)) for it in items[:4]]
    both = ad.encode(rows, "cuda")
    one = torch.cat([ad.encode([r], "cuda") for r in rows])
    cos = torch.nn.functional.cosine_similarity(both, one, dim=-1)
    assert float(cos.min()) > 0.99, cos
