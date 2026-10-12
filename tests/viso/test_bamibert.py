"""Phase 2c text side: frozen BamiBERT features (scripts/bamibert_extract.py, utils/viso_text_cache.py, CachedBertEmbedding). A tiny SYNTHETIC cache in the real layout
(sha1-keyed token states per question, one pooled vector per candidate string) plus the synthetic SigLIP 2 cache of test_siglip2.py: no model weights are needed."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import bamibert_extract as BX  # noqa: E402
import siglip2_extract as SX  # noqa: E402

import data_utils.datasets.s1_dataset as ds  # noqa: E402
import models.viso  # noqa: F401,E402
from builders.model_builder import META_ARCHITECTURE  # noqa: E402
from configs.utils import get_config  # noqa: E402
from utils.viso_s1 import label_space  # noqa: E402
from utils.viso_text_cache import TextCache, key_of  # noqa: E402


@pytest.fixture(scope="module")
def caches(subset32, tmp_path_factory):
    """(siglip2 dir, bamibert dir, items) for the 32-item subset"""
    items = [json.loads(l) for l in open(subset32[0], encoding="utf-8")]
    rng = np.random.default_rng(0)
    sd = str(tmp_path_factory.mktemp("sig"))
    ids = sorted({it["coco_id"] for it in items})
    rng.standard_normal((len(ids) * SX.N_TOK, SX.DIM)).astype("<f2").tofile(os.path.join(sd, SX.BIN))
    np.savez(os.path.join(sd, SX.INDEX), coco_id=np.asarray(ids, dtype=np.int64), offset=np.arange(len(ids), dtype=np.int64) * SX.N_TOK, n_regions=np.full(len(ids), SX.N_TOK, dtype=np.int32))
    np.save(os.path.join(sd, SX.BOXES), SX.grid_boxes())
    json.dump({"dim": SX.DIM, "tokens_per_image": SX.N_TOK, "n_images": len(ids)}, open(os.path.join(sd, SX.META), "w"))
    bd = str(tmp_path_factory.mktemp("bami"))
    qs = sorted({it["questions"]["q"]["instructions"] for it in items})
    cs = sorted({c for it in items if it["questions"]["q"]["type"] == "choice" for c in label_space(it["questions"]["q"])})
    lens = [int(rng.integers(5, 12)) for _ in qs]
    data = rng.standard_normal((sum(lens), BX.DIM)).astype("<f2")
    data.tofile(os.path.join(bd, BX.QBIN))
    keys = [key_of(t) for t in qs]
    order = np.argsort(keys)
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    np.savez(os.path.join(bd, BX.QIDX), key=np.array(keys)[order], offset=off[order], length=np.array(lens, np.int32)[order])
    ckeys = [key_of(t) for t in cs]
    corder = np.argsort(ckeys)
    np.save(os.path.join(bd, BX.CVEC), rng.standard_normal((len(cs), 2, BX.DIM)).astype("<f2")[corder])
    np.savez(os.path.join(bd, BX.CIDX), key=np.array(ckeys)[corder])
    json.dump({"dim": BX.DIM}, open(os.path.join(bd, BX.META), "w"))
    return sd, bd, items, data, off, qs


def cfg_for(caches, arch="ViSO"):
    cfg = get_config("configs/viso_siglip2_bami.yaml" if arch == "ViSO" else "configs/viso_flat_siglip2_bami.yaml")
    cfg.DATASET.FEATURE_DATASET.PACKED_PATH = caches[0]
    cfg.DATASET.FEATURE_DATASET.TEXT_CACHE = caches[1]
    cfg.MODEL.DEVICE = "cpu"
    return cfg


def test_key_is_the_sha1_of_the_exact_text_and_the_reader_returns_what_was_stored(caches):
    sd, bd, items, data, off, qs = caches
    tc = TextCache(bd, "mean")
    for i in (0, 3, len(qs) - 1):
        got = tc.question(qs[i])
        assert got.shape[1] == 768 and np.allclose(got, data[off[i]:off[i] + got.shape[0]].astype(np.float32))
    with pytest.raises(KeyError, match="not in the BamiBERT cache"):
        tc.question("câu hỏi không có trong bộ nhớ đệm")
    assert not np.allclose(TextCache(bd, "mean").candidate(next(c for it in items if it["questions"]["q"]["type"] == "choice" for c in label_space(it["questions"]["q"]))),
                           TextCache(bd, "cls").candidate(next(c for it in items if it["questions"]["q"]["type"] == "choice" for c in label_space(it["questions"]["q"]))))


def test_dataset_adds_states_and_candidate_vectors_and_collate_pads(caches, vocab, subset32):
    cfg = cfg_for(caches)
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    batch = ds.s1_collate([dset[i] for i in range(len(dset))])
    assert batch.question_states.shape[2] == 768 and batch.question_states.shape[:2] == batch.question_state_mask.shape
    lens = batch.question_state_mask.sum(1)
    assert (batch.question_states[~batch.question_state_mask] == 0).all() and int(lens.min()) >= 5
    ch = batch.type_id == 2
    assert batch.cand_vecs.shape[1:] == (16, 768)
    assert (batch.cand_vecs[ch].abs().sum(-1)[batch.cand_mask[ch]] > 0).all() and (batch.cand_vecs[~ch] == 0).all()
    assert (batch.cand_vecs[ch].abs().sum(-1)[~batch.cand_mask[ch]] == 0).all()


def test_default_configs_have_no_text_cache(vocab, subset32, caches):
    cfg = get_config("configs/viso_siglip2.yaml")
    cfg.DATASET.FEATURE_DATASET.PACKED_PATH = caches[0]
    b = ds.s1_collate([ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)[0]])
    assert "question_states" not in b and "cand_vecs" not in b


@pytest.mark.parametrize("arch", ["ViSO", "FlatS1"])
def test_forward_backward_is_finite_and_the_cached_states_get_gradient(caches, vocab, subset32, arch):
    cfg = cfg_for(caches, arch)
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    by_type = {}
    for i in range(len(dset)):
        by_type.setdefault(dset[i].type_id, i)
    batch = ds.s1_collate([dset[by_type[0]], dset[by_type[1]], dset[by_type[2]], dset[by_type[2]]])
    torch.manual_seed(0)
    model = META_ARCHITECTURE.get(arch)(cfg.MODEL, vocab)
    assert model.encoder.cached_text and model.encoder.text_embedding.proj.in_features == 768
    out = model(batch)
    loss = out["logits"].masked_fill(~batch.space_mask, 0).pow(2).mean()
    loss.backward()
    assert torch.isfinite(out["h"]).all() and torch.isfinite(loss)
    assert float(model.encoder.text_embedding.proj.weight.grad.abs().sum()) > 0
    if arch == "ViSO":
        assert float(model.choice_head.proj.weight.grad.abs().sum()) > 0 and model.choice_head.embedding is None


def test_padding_positions_of_the_question_do_not_change_the_output_with_strict_masks(caches, vocab, subset32):
    cfg = cfg_for(caches)
    cfg.MODEL.STRICT_TEXT_MASK = True
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    model = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab).eval()
    b = ds.s1_collate([dset[0], dset[1]])
    with torch.no_grad():
        h1 = model(b)["h"]
        b2 = ds.s1_collate([dset[0], dset[1]])
        b2.question_states = torch.cat([b2.question_states, torch.randn(2, 4, 768)], 1)        # extra padded positions with garbage content
        b2.question_state_mask = torch.cat([b2.question_state_mask, torch.zeros(2, 4, dtype=torch.bool)], 1)
        h2 = model(b2)["h"]
    emb, (m, _) = model.encoder.text_embedding(b2.question_states, b2.question_state_mask)
    assert (emb[~b2.question_state_mask] == 0).all() and (m[:, 0, 0, :][~b2.question_state_mask] < 0).all()
    assert torch.isfinite(h1).all() and torch.allclose(h1, h2, atol=1e-4), float((h1 - h2).abs().max())


def test_default_constituent_masks_keep_the_published_behaviour_and_strict_masks_fix_them():
    """published code: the additive float mask is read as a bool (inverted) and masked_fill is never assigned -> padding content leaks in; strict_mask=True fixes both"""
    from models.constituent_mcan.constituent_module import QuestionConstituentEncoder
    torch.manual_seed(0)
    x = torch.randn(2, 6, 64)
    real = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    x2 = x.clone()
    x2[~real] = torch.randn(int((~real).sum()), 64)                                      # different content in the padded positions
    for strict in (False, True):
        enc = QuestionConstituentEncoder(head=8, d_model=64, d_kv=8, d_ff=128, num_layers=2, dropout=0.0, strict_mask=strict).eval()
        m = real if strict else (~real).float() * -10e4
        with torch.no_grad():
            y, _ = enc(x, m)
            y2, _ = enc(x2, m)
        diff = float((y - y2)[real].abs().max())
        if strict:
            assert diff < 1e-5, diff
        else:
            assert diff > 1e-3, diff          # documents the leak of the published behaviour (so the default stays exactly as published)


def test_standardize_removes_the_common_component_and_is_off_by_default(caches):
    sd, bd, items, data, off, qs = caches
    raw, std = TextCache(bd, "mean"), TextCache(bd, "mean", standardize=True)
    cands = sorted({c for it in items if it["questions"]["q"]["type"] == "choice" for c in label_space(it["questions"]["q"])})
    R = np.stack([raw.candidate(c) for c in cands])
    Z = np.stack([std.candidate(c) for c in cands])
    assert not np.allclose(R, Z) and abs(float(Z.mean(0).max())) < 1e-3 and abs(float(Z.mean(0).min())) < 1e-3
    assert 0.9 < float(Z.std(0).mean()) < 1.1
    q = std.question(qs[0])
    assert q.shape[1] == 768 and np.isfinite(q).all()
    cfg = get_config("configs/viso_siglip2_bami.yaml")
    assert cfg.DATASET.FEATURE_DATASET.TEXT_CACHE_STANDARDIZE is True
