"""ViSO-FT (live encoders): parity with the cached pipeline at initialisation, checkpoint / optimizer structure, LoRA variant construction. Needs the pinned Hugging Face models in the local cache and the local
caches (~/data/siglip2_packed, ~/data/bamibert_cache_x7k); skipped otherwise."""
import os

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKED = next((p for p in (os.path.expanduser("~/data/siglip2_packed"), os.path.expanduser("~/data/kaggle_upload_v2/vivqa-siglip2-packed")) if os.path.isfile(os.path.join(p, "regions_fp16.bin"))), None)
BAMI = os.path.expanduser("~/data/bamibert_cache_x7k")
IMAGES = os.path.join(REPO, "..", "vivqav2_dataset", "images")
S1 = os.path.join(REPO, "..", "vivqa_s1_v2", "data")
needs = pytest.mark.skipif(PACKED is None or not os.path.isdir(BAMI) or not os.path.isdir(IMAGES) or not os.path.isdir(S1), reason="needs the local caches, images and benchmark")


def ft_cfg(**over):
    from configs.utils import get_config
    cfg = get_config(os.path.join(REPO, "configs", "viso_ft_bami.yaml"))
    cfg.MODEL.FT.AMP = "none"
    for k, v in over.items():
        setattr(cfg.MODEL.FT, k, v)
    return cfg


@needs
def test_live_features_equal_the_cache_at_initialisation():
    from data_utils.datasets.s1_live_dataset import load_tokenizer
    from models.viso.viso_ft import LiveEncoders
    from utils.coco_images import ImageStore
    from utils.image_u8_cache import MODEL_ID, REVISION  # noqa: F401
    from utils.viso_packed_features import PackedFeatures
    from utils.viso_s1 import read_jsonl
    from utils.viso_text_cache import TextCache
    cfg = ft_cfg()
    live = LiveEncoders(cfg.MODEL.FT).eval()
    items = read_jsonl(os.path.join(S1, "dev.jsonl"))[:4]
    # vision: the packed cache rows (fp16) vs the live 65 x 768 tokens from the uint8 pixels
    from data_utils.datasets.s1_live_dataset import _OnTheFly
    src = _OnTheFly(ImageStore(os.path.join(REPO, "kaggle", "s1_images_sha256.tsv.gz"), roots=[IMAGES]))
    packed = PackedFeatures(PACKED)
    with torch.no_grad():
        tok = live.image_tokens(torch.stack([src.get(it["coco_id"]) for it in items]))
    for k, it in enumerate(items):
        ref = torch.from_numpy(np.asarray(packed.get_array(it["coco_id"]), np.float32))
        assert tok[k].shape == ref.shape == (65, 768)
        cos = torch.nn.functional.cosine_similarity(tok[k].flatten(), ref.flatten(), dim=0)
        d = (tok[k] - ref).abs()                                                    # fp16 cache vs fp32 live (CPU vs the GPU extraction): mean error ~0.003 on tokens of scale ~0.9; a few outlier dims (|x| ~ 30) reach 0.1
        assert cos > 0.9999 and d.mean() < 0.01 and d[0].max() < 0.02, (float(cos), float(d.mean()), float(d[0].max()))
    # text: raw token states of the question and the mean-pooled candidate vector vs the BamiBERT cache (fp16)
    tc = TextCache(BAMI, "mean", False)
    tkz = load_tokenizer(cfg.MODEL.FT.TEXT_MODEL, cfg.MODEL.FT.TEXT_REVISION)
    q = items[0]["questions"]["q"]["instructions"]
    enc = tkz([q], truncation=True, max_length=48, return_tensors="pt")
    with torch.no_grad():
        h = live.token_states(enc["input_ids"], enc["attention_mask"])[0]
    ref_q = torch.from_numpy(tc.question(q))
    assert h.shape == ref_q.shape and (h - ref_q).abs().max() < 0.05
    ch = next(it for it in read_jsonl(os.path.join(S1, "dev.jsonl")) if it["questions"]["q"]["type"] == "choice")
    from utils.viso_s1 import label_space
    cand = label_space(ch["questions"]["q"])[0]
    enc = tkz([cand], truncation=True, max_length=48, return_tensors="pt")
    with torch.no_grad():
        hc = live.token_states(enc["input_ids"], enc["attention_mask"])[0]
    from models.viso.viso_ft import pool_candidates
    assert (pool_candidates(hc[None], torch.ones(1, hc.shape[0], dtype=torch.bool))[0] - torch.from_numpy(tc.candidate(cand))).abs().max() < 0.05


@needs
def test_state_dict_has_no_frozen_tower_weights_and_the_optimizer_has_three_groups():
    from models.viso.viso_ft import ViSOFT
    from builders.vocab_builder import build_vocab
    import data_utils.vocabs.s1_vocab  # noqa: F401
    from tasks.typed_decision_task import TypedDecisionTask
    cfg = ft_cfg()
    cfg.DATASET.VOCAB.S1_DIR = S1
    cfg.MODEL.DEVICE = "cpu"
    model = ViSOFT(cfg.MODEL, build_vocab(cfg.DATASET.VOCAB))
    sd = model.state_dict()
    n_tower = sum(v.numel() for k, v in sd.items() if k.startswith("live."))
    rep = model.parameter_report()
    assert n_tower == rep["towers_trainable"] + 2 * 2 * 768, "only the unfrozen tower weights and the two normalisation affines are checkpointed"
    assert 4.0e7 < rep["towers_trainable"] < 7.0e7 and rep["towers_total"] > 1.9e8
    top = [n for n, p in model.live.named_parameters() if p.requires_grad and n.startswith(("vision.", "text."))]
    assert all(("layers.8" in n or "layers.9" in n or "layers.10" in n or "layers.11" in n or "layer.8" in n or "layer.9" in n or "layer.10" in n or "layer.11" in n or "post_layernorm" in n or "head." in n) for n in top)
    task = TypedDecisionTask.__new__(TypedDecisionTask)
    task.config, task.model = cfg, model
    opt = task._make_optimizer()
    assert [g["lr"] for g in opt.param_groups] == [cfg.TRAINING.LR_ENCODER, cfg.TRAINING.LR_HEADS, cfg.TRAINING.LR_FT]
    assert sum(len(g["params"]) for g in opt.param_groups) == sum(1 for p in model.parameters() if p.requires_grad)
    missing = model.load_state_dict(sd, strict=True)                       # round trip without the frozen weights
    assert not [k for k in missing.missing_keys if k not in model._frozen_keys]


@needs
def test_lora_variant_builds_and_freezes_the_base():
    from models.viso.viso_ft import LiveEncoders
    live = LiveEncoders(ft_cfg(LORA_R=16).MODEL.FT)
    tr = [n for n, p in live.named_parameters() if p.requires_grad and n.startswith(("vision.", "text."))]
    assert tr and all("lora_" in n for n in tr)


@needs
def test_uint8_image_cache_equals_the_on_the_fly_decode_and_reuses_its_file(tmp_path):
    from data_utils.datasets.s1_live_dataset import _OnTheFly
    from utils.coco_images import ImageStore
    from utils.image_u8_cache import build_or_load, to_pixel_values
    from utils.viso_s1 import read_jsonl
    ids = [it["coco_id"] for it in read_jsonl(os.path.join(S1, "dev.jsonl"))[:5]]
    store = ImageStore(os.path.join(REPO, "kaggle", "s1_images_sha256.tsv.gz"), roots=[IMAGES])
    cache = build_or_load(ids, store, str(tmp_path), workers=0, log=lambda *_: None)
    again = build_or_load(ids, store, str(tmp_path), workers=0, log=lambda *_: None)                  # second call: the finished file is reused
    assert again.base == cache.base and len(cache) == len(set(ids))
    fly = _OnTheFly(store)
    for c in ids:
        a = cache.get(c)
        assert a.dtype == torch.uint8 and a.shape == (3, 256, 256) and torch.equal(a, fly.get(c))
    px = to_pixel_values(cache.get(ids[0])[None])
    assert px.min() >= -1.0 and px.max() <= 1.0
