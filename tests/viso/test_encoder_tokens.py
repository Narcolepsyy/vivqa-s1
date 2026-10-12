import pytest
import torch
import data_utils.datasets.s1_dataset as ds
import models.viso  # noqa: F401
from builders.model_builder import META_ARCHITECTURE
from configs.utils import get_config
from tests.viso.test_bamibert import caches  # noqa: F401


def cfg_for(caches, path="configs/viso_siglip2_bami.yaml", **over):
    cfg = get_config(path)
    cfg.DATASET.FEATURE_DATASET.PACKED_PATH = caches[0]
    cfg.DATASET.FEATURE_DATASET.TEXT_CACHE = caches[1]
    cfg.MODEL.DEVICE = "cpu"
    for k, v in over.items():
        setattr(cfg.MODEL, k, v)
    return cfg


def batch_of(cfg, vocab, subset32):
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    by_type = {}
    for i in range(len(dset)):
        by_type.setdefault(dset[i].type_id, i)
    return ds.s1_collate([dset[by_type[0]], dset[by_type[1]], dset[by_type[2]], dset[by_type[2]]])


def test_encoder_returns_unpooled_visual_tokens(caches, vocab, subset32):
    cfg = cfg_for(caches)
    batch = batch_of(cfg, vocab, subset32)
    model = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab).eval()
    
    out = model.encoder(batch)
    # Check 2-tuple unpacking backward-compatibility
    h, bp = out
    assert h.shape == (4, cfg.MODEL.D_MODEL)
    assert bp is None or isinstance(bp, torch.Tensor)
    
    # Check unpooled visual features attribute
    assert hasattr(out, "vision_features"), "Encoder output must expose unpooled vision_features"
    assert hasattr(out, "vision_mask"), "Encoder output must expose vision_mask"
    assert out.vision_features.shape[0] == 4
    assert out.vision_features.shape[2] == cfg.MODEL.D_MODEL
    assert out.vision_features.shape[1] == batch.region_features.shape[1]
    assert out.vision_mask.shape == (4, batch.region_features.shape[1])
