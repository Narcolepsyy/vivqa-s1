import math
import pytest
import torch
import data_utils.datasets.s1_dataset as ds
import models.viso  # noqa: F401
from builders.model_builder import META_ARCHITECTURE
from configs.utils import get_config
from tests.viso.test_bamibert import caches  # noqa: F401


def cfg_for(caches, cross_attn=False):
    cfg = get_config("configs/viso_siglip2_bami.yaml")
    cfg.DATASET.FEATURE_DATASET.PACKED_PATH = caches[0]
    cfg.DATASET.FEATURE_DATASET.TEXT_CACHE = caches[1]
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.CHOICE_CROSS_ATTENTION = cross_attn
    return cfg


def batch_of(cfg, vocab, subset32):
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    by_type = {}
    for i in range(len(dset)):
        by_type.setdefault(dset[i].type_id, i)
    # choice items at index 2
    return ds.s1_collate([dset[by_type[2]], dset[by_type[2]]])


def test_choice_cross_attention_disabled_by_default(caches, vocab, subset32):
    cfg = cfg_for(caches, cross_attn=False)
    batch = batch_of(cfg, vocab, subset32)
    model = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab).eval()
    assert getattr(model.choice_head, "cross_attention_enabled", False) is False
    out = model(batch)
    assert out["logits"].shape == (2, 16)


def test_choice_cross_attention_enabled_forward_and_backward(caches, vocab, subset32):
    cfg = cfg_for(caches, cross_attn=True)
    batch = batch_of(cfg, vocab, subset32)
    model = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab).train()
    assert model.choice_head.cross_attention_enabled is True
    assert hasattr(model.choice_head, "cross_attn")
    assert hasattr(model.choice_head, "cross_gate")
    
    out = model(batch)
    logits = out["logits"]
    assert logits.shape == (2, 16)
    
    # Compute loss on choice items
    loss = logits.sum()
    loss.backward()
    assert model.choice_head.cross_attn.in_proj_weight.grad is not None
    assert model.choice_head.cross_q_ctx.weight.grad is not None
    assert model.choice_head.cross_gate.grad is not None


def test_score_count_pool_enabled_forward_and_backward(caches, vocab, subset32):
    cfg = cfg_for(caches, cross_attn=False)
    cfg.MODEL.SCORE_COUNT_POOL = True
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    by_type = {}
    for i in range(len(dset)):
        by_type.setdefault(dset[i].type_id, i)
    batch = ds.s1_collate([dset[by_type[1]], dset[by_type[1]]])
    model = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab).train()
    assert model.score_head.count_pool_enabled is True
    out = model(batch)
    logits = out["logits"][:, :11]
    assert logits.shape == (2, 11)
    logits.sum().backward()
    assert model.score_head.count_gate.weight.grad is not None
    assert model.score_head.count_scale.grad is not None

