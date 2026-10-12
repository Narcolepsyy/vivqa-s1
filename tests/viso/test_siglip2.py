"""Phase 2c: ViSO on frozen SigLIP 2 cached features (configs/viso_siglip2.yaml, scripts/siglip2_extract.py). Uses a tiny SYNTHETIC packed cache in the SigLIP 2 layout
(65 tokens x 768 per image, shared boxes.npy), so no model weights and no real feature files are needed."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import siglip2_extract as SX  # noqa: E402

import data_utils.datasets.s1_dataset as ds  # noqa: E402
import models.viso  # noqa: F401,E402  (registers ViSO / FlatS1)
from builders.model_builder import META_ARCHITECTURE  # noqa: E402
from configs.utils import get_config  # noqa: E402
from utils.viso_packed_features import PackedFeatures  # noqa: E402


@pytest.fixture(scope="module")
def cache_dir(subset32, tmp_path_factory):
    items = [json.loads(l) for l in open(subset32[0], encoding="utf-8")]
    ids = sorted({it["coco_id"] for it in items})
    d = str(tmp_path_factory.mktemp("siglip2"))
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(ids) * SX.N_TOK, SX.DIM)).astype("<f2")
    data.tofile(os.path.join(d, SX.BIN))
    np.savez(os.path.join(d, SX.INDEX), coco_id=np.asarray(ids, dtype=np.int64), offset=np.arange(len(ids), dtype=np.int64) * SX.N_TOK, n_regions=np.full(len(ids), SX.N_TOK, dtype=np.int32))
    np.save(os.path.join(d, SX.BOXES), SX.grid_boxes())
    json.dump({"dim": SX.DIM, "tokens_per_image": SX.N_TOK, "n_images": len(ids)}, open(os.path.join(d, SX.META), "w"))
    return d, ids, data


def cfg_for(path, arch="ViSO", **model_over):
    cfg = get_config("configs/viso_siglip2.yaml" if arch == "ViSO" else "configs/viso_flat_siglip2.yaml")
    cfg.DATASET.FEATURE_DATASET.PACKED_PATH = path
    cfg.MODEL.DEVICE = "cpu"
    for k, v in model_over.items():
        setattr(cfg.MODEL, k, v)
    return cfg


def make_dataset(cfg, vocab, subset32):
    return ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)


def test_packed_reader_uses_the_dim_of_meta_and_the_shared_boxes(cache_dir):
    d, ids, data = cache_dir
    pf = PackedFeatures(d)
    assert pf.dim == 768 and pf.boxes.shape == (65, 4) and len(pf) == len(ids)
    arr = pf.get_array(ids[1])
    assert arr.shape == (65, 768) and np.array_equal(arr, data[65:130].astype(np.float32))
    b = pf.get_boxes(65)
    assert np.allclose(b[0], [0, 0, 1, 1]) and np.allclose(b[1], [0, 0, 1 / 8, 1 / 8]) and np.allclose(b[64], [7 / 8, 7 / 8, 1, 1])


def test_siglip2_configs_keep_the_vinvl_defaults_untouched():
    old, new = get_config("configs/viso_vivqav2.yaml"), get_config("configs/viso_siglip2.yaml")
    assert old.MODEL.VISION_EMBEDDING.D_FEATURE == 2048 and old.DATASET.FEATURE_DATASET.FEATURE_FORMAT == "npy" and "USE_BOXES" not in old.MODEL.VISION_EMBEDDING
    assert new.MODEL.VISION_EMBEDDING.D_FEATURE == 768 and new.DATASET.FEATURE_DATASET.FEATURE_FORMAT == "packed" and new.DATASET.VOCAB.S1_DIR == "../vivqa_s1_v2/data"
    assert new.DATASET.FEATURE_DATASET.PACKED_PATH == "~/data/siglip2_packed" and new.MODEL.VISION_EMBEDDING.USE_BOXES is True


def test_dataset_round_trip_of_features_and_boxes(cache_dir, vocab, subset32):
    d, ids, data = cache_dir
    dset = make_dataset(cfg_for(d), vocab, subset32)
    s = dset[0]
    assert s.region_features.shape == (65, 768) and s.region_boxes.shape == (65, 4) and s.region_features.dtype == torch.float32
    row = ids.index(s.image_id)
    assert torch.equal(s.region_features, torch.from_numpy(data[row * 65:(row + 1) * 65].astype(np.float32)))
    batch = ds.s1_collate([dset[0], dset[1]])
    assert batch.region_features.shape[1:] == (65, 768) and batch.region_boxes.shape[1:] == (65, 4) and bool(batch.region_mask.all())


def test_use_boxes_needs_a_packed_cache_with_boxes(cache_dir, vocab, subset32, tmp_path):
    cfg = cfg_for(cache_dir[0])
    cfg.DATASET.FEATURE_DATASET.FEATURE_FORMAT = "npy"
    with pytest.raises(ValueError, match="USE_BOXES"):
        make_dataset(cfg, vocab, subset32)


@pytest.mark.parametrize("arch", ["ViSO", "FlatS1"])
def test_forward_backward_is_finite_with_siglip2_features(cache_dir, vocab, subset32, arch):
    d = cache_dir[0]
    cfg = cfg_for(d, arch)
    dset = make_dataset(cfg, vocab, subset32)
    by_type = {}
    for i in range(len(dset)):
        by_type.setdefault(dset[i].type_id, i)
    batch = ds.s1_collate([dset[by_type[0]], dset[by_type[1]], dset[by_type[2]], dset[by_type[2]]])
    torch.manual_seed(0)
    model = META_ARCHITECTURE.get(arch)(cfg.MODEL, vocab)
    assert model.encoder.use_boxes and model.encoder.vision_embedding.proj.in_features == 768
    out = model(batch)
    loss = out["logits"].masked_fill(~batch.space_mask, 0).pow(2).mean()
    loss.backward()
    assert torch.isfinite(out["h"]).all() and torch.isfinite(loss)
    assert model.encoder.box_proj.weight.grad is not None and float(model.encoder.vision_embedding.proj.weight.grad.abs().sum()) > 0


def test_a5_zero_vision_ignores_features_and_boxes(cache_dir, vocab, subset32):
    cfg = cfg_for(cache_dir[0], ZERO_VISION=True)
    dset = make_dataset(cfg, vocab, subset32)
    b = ds.s1_collate([dset[0], dset[1]])
    torch.manual_seed(0)
    model = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab).eval()
    with torch.no_grad():
        h0, _ = model.encoder(b)
        b2 = ds.S1Batch(**{k: v for k, v in b.items()})
        b2.region_features = torch.randn_like(b.region_features)
        b2.region_boxes = torch.rand_like(b.region_boxes)
        h1, _ = model.encoder(b2)
    assert float((h0 - h1).abs().max()) < 1e-6


def test_forward_is_deterministic(cache_dir, vocab, subset32):
    cfg = cfg_for(cache_dir[0])
    dset = make_dataset(cfg, vocab, subset32)
    b = ds.s1_collate([dset[i] for i in range(4)])
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        m = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab).eval()
        with torch.no_grad():
            outs.append(m(b)["logits"])
    assert torch.equal(outs[0], outs[1])


def test_vinvl_default_model_has_no_box_embedding(vocab):
    cfg = get_config("configs/viso_vivqav2.yaml")
    cfg.MODEL.DEVICE = "cpu"
    m = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab)
    assert not m.encoder.use_boxes and not hasattr(m.encoder, "box_proj")


def test_kaggle_prepare_builds_the_siglip2_bundle(cache_dir, tmp_path):
    import kaggle_prepare as K
    d = cache_dir[0]
    meta = json.load(open(os.path.join(d, SX.META)))
    meta.update(model_id=SX.MODEL_ID, revision=SX.REVISION, license=SX.LICENSE)
    json.dump(meta, open(os.path.join(d, SX.META), "w"))
    json.dump({"ok": True}, open(os.path.join(d, SX.VERIFY), "w"))
    json.dump({"params_total": 1}, open(os.path.join(d, "vision_tower_efficiency.json"), "w"))
    assert K.packed_kind(d) == "siglip2"
    out = str(tmp_path / "upload")
    res = K.build(out, d, s1_root="../vivqa_s1_v2", log=lambda *_: None)
    assert set(res) == {"vivqa-siglip2-packed", "vivqa-s1", "viso-code"} and all(r["sha256_ok"] for r in res.values())
    assert sorted(os.listdir(os.path.join(out, "vivqa-siglip2-packed"))) == sorted(K.SIGLIP2_FILES + ("SHA256SUMS.txt", "dataset-metadata.json"))
    assert os.path.isfile(os.path.join(out, "vivqa-s1", "data", "eval_flags.jsonl")) and os.path.isfile(os.path.join(out, "vivqa-s1", "data", "test.jsonl"))
    code = os.path.join(out, "viso-code", "ViVQAv2-main")
    for f in ("kaggle/experiments_siglip2.json", "configs/viso_siglip2.yaml", "configs/viso_flat_siglip2.yaml", "scripts/viso_views_report.py", "scripts/siglip2_extract.py"):
        assert os.path.isfile(os.path.join(code, f)), f
    assert "SigLIP 2" in open(os.path.join(out, "README_KAGGLE.md"), encoding="utf-8").read()


def test_siglip2_manifest_matches_the_v2_benchmark_and_has_no_vinvl_rows():
    import hashlib
    m = json.load(open("kaggle/experiments_siglip2.json"))
    for f, h in m["s1_sha256"].items():
        assert hashlib.sha256(open(os.path.join("../vivqa_s1_v2/data", f), "rb").read()).hexdigest() == h, f
    assert {j["base_config"] for j in m["jobs"]} == {"configs/viso_siglip2.yaml", "configs/viso_flat_siglip2.yaml"}
    assert sorted({j["method"] for j in m["jobs"]}) == ["A0", "A1", "A2", "A3", "A3c", "A5", "A6"] and len(m["jobs"]) == 15
    assert all(j["est_minutes"] for j in m["jobs"]) and set(m["methods"]) >= {"A0T", "A4"}
