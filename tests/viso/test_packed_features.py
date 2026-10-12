import json
import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import viso_pack_features as P  # noqa: E402

import data_utils.datasets.s1_dataset as ds  # noqa: E402
from tests.viso.conftest import FEATURES, S1_DIR, feature_cfg  # noqa: E402
from utils.viso_packed_features import DIM, PackedFeatures  # noqa: E402

N = 20


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("pack"))
    P.pack(FEATURES, S1_DIR, out, limit=N, log=lambda *_: None)
    return out


def test_reader_matches_originals_within_fp16_tolerance(pack_dir):
    pf = PackedFeatures(pack_dir)
    assert len(pf) == N
    for cid in pf.coco_id.tolist():
        orig = np.asarray(P.load_source(FEATURES, cid)["region_features"], dtype=np.float32)
        t, n = pf.get(cid)
        assert isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.shape == (n, DIM) == orig.shape
        assert float(np.abs(t.numpy() - orig).max()) <= 1e-3 * float(np.abs(orig).max())


def test_offsets_and_counts_are_consistent(pack_dir):
    pf = PackedFeatures(pack_dir)
    assert (np.diff(pf.coco_id) > 0).all()
    assert np.array_equal(pf.offset, np.concatenate([[0], np.cumsum(pf.n_regions)[:-1]]))
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    size = os.path.getsize(os.path.join(pack_dir, "regions_fp16.bin"))
    assert size == int(pf.n_regions.sum()) * DIM * 2 == meta["total_regions"] * DIM * 2 and meta["n_images"] == N


def test_missing_id_raises_keyerror_with_the_id(pack_dir):
    pf = PackedFeatures(pack_dir)
    with pytest.raises(KeyError, match="123456789"):
        pf.get(123456789)
    with pytest.raises(KeyError):
        pf.get(int(pf.coco_id[0]) - 1)


def test_missing_source_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        P.pack(str(tmp_path), S1_DIR, str(tmp_path / "out"), limit=3, log=lambda *_: None)


def test_repack_is_byte_identical(pack_dir, tmp_path):
    again = P.pack(FEATURES, S1_DIR, str(tmp_path / "again"), limit=N, log=lambda *_: None)
    first = json.load(open(os.path.join(pack_dir, "meta.json")))
    assert again["sha256"] == first["sha256"]


def test_refuses_to_write_inside_the_project():
    with pytest.raises(SystemExit, match="refusing"):
        P.main(["--out", os.path.join(P.REPO, "packed_inside"), "--limit", "1"])


def test_tags_are_written_for_every_image(pack_dir):
    rows = [json.loads(l) for l in open(os.path.join(pack_dir, "tags.jsonl"), encoding="utf-8")]
    pf = PackedFeatures(pack_dir)
    assert [r["coco_id"] for r in rows] == pf.coco_id.tolist() and all("objects" in r and "attributes" in r for r in rows)


def packed_cfg(path):
    cfg = feature_cfg(0)
    cfg.FEATURE_FORMAT, cfg.PACKED_PATH = "packed", path
    return cfg


@pytest.fixture(scope="module")
def tiny_jsonl(pack_dir, tmp_path_factory):
    """S1 dev items whose coco_id is in the 20-image pack."""
    ids = set(PackedFeatures(pack_dir).coco_id.tolist())
    p = tmp_path_factory.mktemp("s1p") / "items.jsonl"
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for name in ("train", "calib", "dev", "test"):
            for line in open(os.path.join(S1_DIR, f"{name}.jsonl"), encoding="utf-8"):
                if json.loads(line)["coco_id"] in ids:
                    f.write(line)
                    n += 1
    assert n >= 20
    return str(p)


def test_dataset_packed_equals_npy_shapes_and_masks(vocab, pack_dir, tiny_jsonl):
    d_npy = ds.S1FeatureDataset(tiny_jsonl, vocab, feature_cfg(0))
    d_pk = ds.S1FeatureDataset(tiny_jsonl, vocab, packed_cfg(pack_dir))
    assert d_pk.feature_format == "packed" and d_npy.feature_format == "npy"
    for i in range(len(d_npy)):
        a, b = d_npy[i], d_pk[i]
        assert a.region_features.shape == b.region_features.shape and b.region_features.dtype == torch.float32
        assert float((a.region_features - b.region_features).abs().max()) <= 1e-3 * float(a.region_features.abs().max())
        for k in ("question_tokens", "space_mask", "target", "gold_idx", "cand_tokens", "type_id"):
            assert torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k]))
    batch = ds.s1_collate([d_pk[i] for i in range(6)])
    for i in range(6):
        r = d_pk[i].region_features.shape[0]
        assert bool(batch.region_mask[i, :r].all()) and not bool(batch.region_mask[i, r:].any())


def test_default_config_still_uses_npy():
    from configs.utils import get_config
    for f in ("configs/viso_vivqav2.yaml", "configs/viso_flat_vivqav2.yaml"):
        assert get_config(f).DATASET.FEATURE_DATASET.FEATURE_FORMAT == "npy"


def test_bad_feature_format_is_rejected(vocab, tiny_jsonl):
    cfg = feature_cfg(0)
    cfg.FEATURE_FORMAT = "parquet"
    with pytest.raises(ValueError):
        ds.S1FeatureDataset(tiny_jsonl, vocab, cfg)


def test_workers_can_read_the_packed_file(vocab, pack_dir, tiny_jsonl):
    d = ds.S1FeatureDataset(tiny_jsonl, vocab, packed_cfg(pack_dir))
    d[0]                                           # opens the memmap in the parent first: it must not be shipped to workers
    loader = DataLoader(d, batch_size=4, num_workers=2, collate_fn=ds.s1_collate)
    batches = list(loader)
    assert sum(b.id.shape[0] for b in batches) == len(d)
    ref = ds.s1_collate([d[i] for i in range(4)])
    assert torch.equal(batches[0].region_features, ref.region_features)
