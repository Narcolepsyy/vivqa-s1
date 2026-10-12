import torch

import data_utils.datasets.s1_dataset as ds
from utils.viso_s1 import TYPE_IDS, label_space


def test_sizes_and_types(dataset32):
    assert len(dataset32) >= 32
    assert {dataset32[i].type_id for i in range(len(dataset32))} == {0, 1, 2}


def test_sample_fields_and_masks(dataset32):
    for i in range(len(dataset32)):
        s, it = dataset32[i], dataset32.items[i]
        q = it["questions"]["q"]
        n = {"noul": 2, "score": 11, "choice": len(q["criteria"]) if q["type"] == "choice" else 0}[q["type"]]
        assert s.space_mask.shape == (16,) and int(s.space_mask.sum()) == n
        assert bool(s.space_mask[:n].all()) and not bool(s.space_mask[n:].any())      # contiguous prefix
        assert s.type_id == TYPE_IDS[q["type"]]
        assert s.region_features.dtype == torch.float32 and s.region_features.shape[1] == 2048
        assert s.cand_mask.sum() == (n if q["type"] == "choice" else 0)


def test_target_sums_to_one_and_is_zero_outside_mask(dataset32):
    for i in range(len(dataset32)):
        s = dataset32[i]
        assert abs(float(s.target.sum()) - 1.0) < 1e-6
        assert float(s.target[~s.space_mask].abs().sum()) == 0.0


def test_gold_idx_points_at_gold_label(dataset32):
    for i in range(len(dataset32)):
        s, it = dataset32[i], dataset32.items[i]
        space = label_space(it["questions"]["q"])
        assert space[s.gold_idx] == it["gold_label"]
        assert s.target[s.gold_idx] > 0


def test_label_space_order_is_the_items_own_order(dataset32):
    for i in range(len(dataset32)):
        it = dataset32.items[i]
        q = it["questions"]["q"]
        if q["type"] == "noul":
            assert label_space(q) == ["false", "true"]
        elif q["type"] == "score":
            assert label_space(q) == [str(k) for k in range(11)]
        else:
            assert label_space(q) == list(q["criteria"])           # Phase-1 shuffled order, not sorted


def test_candidate_tokens_follow_candidate_order(dataset32, vocab):
    for i in range(len(dataset32)):
        s, it = dataset32[i], dataset32.items[i]
        q = it["questions"]["q"]
        if q["type"] != "choice":
            assert int((s.cand_tokens != vocab.padding_idx).sum()) == 0
            continue
        for k, text in enumerate(q["criteria"]):
            ids = vocab.encode_candidate(text)
            assert s.cand_tokens[k, :len(ids)].tolist() == ids
            assert bool((s.cand_tokens[k, len(ids):] == vocab.padding_idx).all())


def test_flat_index_maps_into_flat_vocab(dataset32, vocab):
    for i in range(len(dataset32)):
        s, it = dataset32[i], dataset32.items[i]
        q = it["questions"]["q"]
        if q["type"] == "noul":
            assert [vocab.flat_itos[j] for j in s.flat_index[:2].tolist()] == ["<no>", "<yes>"]
        if q["type"] == "score":
            assert [vocab.flat_itos[j] for j in s.flat_index[:11].tolist()] == [f"#{k}" for k in range(11)]
        assert bool((s.flat_index[~s.space_mask] == -1).all())


def test_collate_pads_regions_and_masks(dataset32):
    samples = [dataset32[i] for i in range(6)]
    b = ds.s1_collate(samples)
    r_max = max(s.region_features.shape[0] for s in samples)
    assert b.region_features.shape == (6, r_max, 2048) and b.region_mask.shape == (6, r_max)
    for i, s in enumerate(samples):
        r = s.region_features.shape[0]
        assert bool(b.region_mask[i, :r].all()) and not bool(b.region_mask[i, r:].any())
        assert float(b.region_features[i, r:].abs().sum()) == 0.0
        assert torch.equal(b.region_features[i, :r], s.region_features)
    assert b.space_mask.shape == (6, 16) and b.cand_tokens.shape[:2] == (6, 16)


def test_vocab_uses_train_calib_words_only(vocab):
    from data_utils.utils import preprocess_sentence
    import json
    seen = set()
    for name in ("train", "calib"):
        for l in open(f"../vivqa_s1_v2/data/{name}.jsonl", encoding="utf-8"):
            it = json.loads(l)
            seen.update(preprocess_sentence(it["questions"]["q"]["instructions"], None))
    only_dev = None
    for l in open("../vivqa_s1_v2/data/dev.jsonl", encoding="utf-8"):
        for w in preprocess_sentence(json.loads(l)["questions"]["q"]["instructions"], None):
            if w not in seen and w not in vocab.stoi:
                only_dev = w
                break
        if only_dev:
            break
    assert only_dev is not None and vocab.encode_candidate(only_dev) == [vocab.unk_idx]
    assert vocab.total_answers == len(vocab.flat_itos) and vocab.flat_itos[:2] == ["<yes>", "<no>"]


def test_feature_cache_returns_same_values(vocab, subset32):
    from tests.viso.conftest import feature_cfg
    d0 = ds.S1FeatureDataset(subset32[0], vocab, feature_cfg(0))
    d1 = ds.S1FeatureDataset(subset32[0], vocab, feature_cfg(4))
    for i in range(8):
        assert torch.equal(d0[i].region_features, d1[i].region_features)
    assert len(d1._cache) <= 4
