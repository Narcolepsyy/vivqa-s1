"""Opt-in options added after the collapse diagnosis: MODEL.FUSION_PRODUCT, TRAINING.VISION_PATH_AS_HEADS / TEXT_PROJ_AS_HEADS (optimizer groups), STRICT_TEXT_MASK on in the SigLIP 2 configs."""
import types

import pytest
import torch

import data_utils.datasets.s1_dataset as ds
import models.viso  # noqa: F401
from builders.model_builder import META_ARCHITECTURE
from configs.utils import get_config
from tasks.typed_decision_task import TypedDecisionTask
from tests.viso.test_bamibert import caches  # noqa: F401  (fixture: synthetic SigLIP 2 + BamiBERT caches)


def cfg_for(caches, path="configs/viso_siglip2.yaml"):
    cfg = get_config(path)
    cfg.DATASET.FEATURE_DATASET.PACKED_PATH = caches[0]
    cfg.DATASET.FEATURE_DATASET.TEXT_CACHE = caches[1]
    cfg.MODEL.DEVICE = "cpu"
    return cfg


def batch_of(cfg, vocab, subset32):
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    by_type = {}
    for i in range(len(dset)):
        by_type.setdefault(dset[i].type_id, i)
    return ds.s1_collate([dset[by_type[0]], dset[by_type[1]], dset[by_type[2]], dset[by_type[2]]])


def test_siglip2_configs_have_strict_masks_on_and_the_new_options_off():
    for f in ("viso_siglip2", "viso_flat_siglip2", "viso_siglip2_bami", "viso_flat_siglip2_bami"):
        cfg = get_config(f"configs/{f}.yaml")
        assert cfg.MODEL.STRICT_TEXT_MASK is True and cfg.MODEL.FUSION_PRODUCT is False
        assert cfg.TRAINING.VISION_PATH_AS_HEADS is False and cfg.TRAINING.TEXT_PROJ_AS_HEADS is False
    legacy = get_config("configs/viso_vivqav2.yaml")                   # the paper's VinVL setup keeps the published behaviour
    assert not getattr(legacy.MODEL, "STRICT_TEXT_MASK", False) and not getattr(legacy.MODEL, "FUSION_PRODUCT", False)


def test_fusion_product_is_off_by_default_and_adds_a_trainable_hadamard_term(caches, vocab, subset32):
    cfg = cfg_for(caches)
    b = batch_of(cfg, vocab, subset32)
    torch.manual_seed(0)
    base = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab).eval()
    assert not base.encoder.fusion_product and not hasattr(base.encoder, "prod_q")
    cfg.MODEL.FUSION_PRODUCT = True
    torch.manual_seed(0)
    prod = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab).train()
    assert prod.encoder.fusion_product
    out = prod(b)
    out["logits"].masked_fill(~b.space_mask, 0).pow(2).mean().backward()
    for m in (prod.encoder.prod_q, prod.encoder.prod_v):
        assert float(m.weight.grad.abs().sum()) > 0
    with torch.no_grad():
        h_base, h_prod = base.encoder(b)[0], prod.eval().encoder(b)[0]
    assert torch.isfinite(h_prod).all() and not torch.allclose(h_base, h_prod)
    # the product term depends on BOTH sides: zeroing the vision features changes h even when only the product path would carry the interaction
    b0 = batch_of(cfg, vocab, subset32)
    b0["region_features"] = torch.zeros_like(b0["region_features"])
    with torch.no_grad():
        assert not torch.allclose(prod.encoder(b0)[0], h_prod)


def make_groups(cfg, vocab, **train_over):
    for k, v in train_over.items():
        setattr(cfg.TRAINING, k, v)
    model = META_ARCHITECTURE.get(cfg.MODEL.ARCHITECTURE)(cfg.MODEL, vocab)
    stub = types.SimpleNamespace(config=cfg, model=model)
    opt = TypedDecisionTask._make_optimizer(stub)
    names = {id(p): n for n, p in model.named_parameters()}
    return model, [{names[id(p)] for p in g["params"]} for g in opt.param_groups], [g["lr"] for g in opt.param_groups]


def test_lr_groups_default_and_vision_path_as_heads(caches, vocab):
    cfg = cfg_for(caches)
    model, (enc, heads), lrs = make_groups(cfg, vocab)
    assert lrs == [cfg.TRAINING.LR_ENCODER, cfg.TRAINING.LR_HEADS]
    assert any(n.startswith("encoder.vision_proj") for n in enc) and not any(n.startswith("noul_head") for n in enc)
    total = len(enc) + len(heads)
    cfg = cfg_for(caches)
    cfg.MODEL.FUSION_PRODUCT = True
    model, (enc2, heads2), _ = make_groups(cfg, vocab, VISION_PATH_AS_HEADS=True)
    for n in ("encoder.vision_embedding.proj.weight", "encoder.vision_proj.weight", "encoder.vision_attr_reduce.fc1.weight", "encoder.box_proj.weight", "encoder.prod_v.weight"):
        assert n in heads2 and n not in enc2, n
    for n in ("encoder.text_proj.weight", "encoder.prod_q.weight", "encoder.constituent_encoder.norm.weight"):
        assert n in enc2, n
    assert len(enc2) + len(heads2) == len(list(model.parameters())) and len(enc2) < len(enc) + 2 and total > 0


def test_text_proj_as_heads_moves_the_bert_projection(caches, vocab):
    cfg = cfg_for(caches, "configs/viso_siglip2_bami.yaml")
    model, (enc, heads), _ = make_groups(cfg, vocab, TEXT_PROJ_AS_HEADS=True)
    assert "encoder.text_embedding.proj.weight" in heads and "encoder.text_embedding.proj.weight" not in enc
    model, (enc, heads), _ = make_groups(cfg_for(caches, "configs/viso_siglip2_bami.yaml"), vocab)
    assert "encoder.text_embedding.proj.weight" in enc


def test_attr_reduce_init_std_is_off_by_default_and_makes_the_pooling_logits_vary(caches, vocab, subset32):
    cfg = cfg_for(caches)
    assert cfg.MODEL.ATTR_REDUCE_INIT_STD == 0.0
    torch.manual_seed(0)
    default = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab)
    cfg.MODEL.ATTR_REDUCE_INIT_STD = 0.5
    torch.manual_seed(0)
    wide = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab)
    for name in ("vision_attr_reduce", "text_attr_reduce"):
        assert abs(float(getattr(wide.encoder, name).fc2.weight.std()) - 0.5) < 0.1
        assert float(getattr(default.encoder, name).fc2.weight.std()) < 0.1


def test_type_weights_are_off_by_default_and_scale_the_training_loss_per_type(caches, vocab):
    cfg = cfg_for(caches)
    assert list(cfg.LOSS.TYPE_WEIGHTS) == [1.0, 1.0, 1.0]
    stub = types.SimpleNamespace(config=cfg)
    per = torch.ones(6)
    tid = torch.tensor([0, 0, 1, 1, 2, 2])
    assert TypedDecisionTask.weighted(stub, per, tid) is per                           # off: returned unchanged
    cfg.LOSS.TYPE_WEIGHTS = [3.0, 1.0, 0.5]
    assert TypedDecisionTask.weighted(stub, per, tid).tolist() == [3.0, 3.0, 1.0, 1.0, 0.5, 0.5]


# ------------------------------------------------------------------------------------------------------------- B-BUTD baseline, FUSION_DROPOUT, ENCODER_AS_HEADS
def butd_cfg(caches):
    cfg = get_config("configs/viso_butd_siglip2.yaml")
    cfg.DATASET.FEATURE_DATASET.PACKED_PATH = caches[0]
    cfg.MODEL.DEVICE = "cpu"
    return cfg


def test_butd_reuses_the_typed_heads_and_trains_end_to_end(caches, vocab, subset32):
    cfg = butd_cfg(caches)
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    by_type = {}
    for i in range(len(dset)):
        by_type.setdefault(dset[i].type_id, i)
    b = ds.s1_collate([dset[by_type[0]], dset[by_type[1]], dset[by_type[2]], dset[by_type[2]]])
    torch.manual_seed(0)
    m = META_ARCHITECTURE.get("BUTD")(cfg.MODEL, vocab)
    assert type(m.encoder).__name__ == "ButdEncoder" and hasattr(m, "noul_head") and hasattr(m, "score_head") and hasattr(m, "choice_head")
    out = m(b)
    assert out["logits"].shape == (4, 16) and torch.isfinite(out["h"]).all() and out["break_probs"] is None
    out["logits"].masked_fill(~b.space_mask, 0).pow(2).mean().backward()
    for mod in (m.encoder.gru, m.encoder.att_q, m.encoder.att_v, m.encoder.fuse_q, m.encoder.fuse_v, m.encoder.vision_proj, m.noul_head.mlp):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in mod.parameters())
    # the output depends on the question and on the image
    m.eval()
    with torch.no_grad():
        h0 = m.encoder(b)[0]
        b2 = ds.s1_collate([dset[by_type[0]], dset[by_type[1]], dset[by_type[2]], dset[by_type[2]]])
        b2["question_tokens"] = b.question_tokens.flip(0)
        assert not torch.allclose(h0, m.encoder(b2)[0])
        b3 = ds.s1_collate([dset[by_type[0]], dset[by_type[1]], dset[by_type[2]], dset[by_type[2]]])
        b3["region_features"] = b.region_features.flip(0)
        assert not torch.allclose(h0, m.encoder(b3)[0])
        cfg.MODEL.ZERO_VISION = True
        z = META_ARCHITECTURE.get("BUTD")(cfg.MODEL, vocab).eval()
        assert torch.allclose(z.encoder(b)[0], z.encoder(b3)[0])                       # A5-style: images are ignored


def test_butd_attention_ignores_padded_regions(caches, vocab, subset32):
    cfg = butd_cfg(caches)
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    m = META_ARCHITECTURE.get("BUTD")(cfg.MODEL, vocab).eval()
    b = ds.s1_collate([dset[0], dset[1]])
    with torch.no_grad():
        h1 = m.encoder(b)[0]
        b["region_features"] = torch.cat([b.region_features, torch.randn(2, 3, 768)], 1)
        b["region_mask"] = torch.cat([b.region_mask, torch.zeros(2, 3, dtype=torch.bool)], 1)
        b["region_boxes"] = torch.cat([b.region_boxes, torch.zeros(2, 3, 4)], 1)
        h2 = m.encoder(b)[0]
    assert torch.allclose(h1, h2, atol=1e-5)


def test_encoder_as_heads_gives_one_group_with_every_parameter(caches, vocab):
    cfg = butd_cfg(caches)
    assert cfg.TRAINING.ENCODER_AS_HEADS is True
    model, groups, lrs = make_groups(cfg, vocab)
    assert len(groups) == 1 and lrs == [cfg.TRAINING.LR_HEADS] and len(groups[0]) == len(list(model.parameters()))
    default = cfg_for(caches)
    assert default.TRAINING.ENCODER_AS_HEADS is False


def test_fusion_dropout_is_off_by_default_and_acts_in_training_only(caches, vocab, subset32):
    cfg = cfg_for(caches)
    assert cfg.MODEL.FUSION_DROPOUT == 0.0
    dset = ds.S1FeatureDataset(subset32[0], vocab, cfg.DATASET.FEATURE_DATASET)
    b = ds.s1_collate([dset[i] for i in range(4)])
    cfg.MODEL.FUSION_DROPOUT = 0.5
    torch.manual_seed(0)
    m = META_ARCHITECTURE.get("ViSO")(cfg.MODEL, vocab)
    m.train()
    torch.manual_seed(1); a1 = m.encoder(b)[0]
    torch.manual_seed(2); a2 = m.encoder(b)[0]
    assert not torch.allclose(a1, a2)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m.encoder(b)[0], m.encoder(b)[0])


# ------------------------------------------------------------------------------------------------------------- learning-curve subsets and EXTRA_TRAIN
def test_yes_no_train_fraction_is_seeded_stratified_and_train_only(caches, vocab, tmp_path):
    import json
    from utils.viso_s1 import yesno_form
    rows = []
    for i in range(400):                                   # synthetic train split: yes/no items of three forms and both labels + other types
        form = ("Có con nào trong ảnh không?", "Màu của cái áo là xanh không?", "Cái này tròn không?")[i % 3]
        rows.append({"id": i, "image": "x.jpg", "coco_id": 0, "state_text": "", "questions": {"q": {"type": "noul", "instructions": form}}, "gold": {"q": {"probabilities": {"false": 0.5, "true": 0.5}}},
                     "gold_label": "true" if i % 2 else "false", "gold_source": "human10", "human_subset": True, "meta": {"en_q": ""}})
    rows += [dict(rows[0], id=1000 + k, questions={"q": {"type": "score", "instructions": "Có bao nhiêu?", "levels": [str(j) for j in range(11)]}}, gold={"q": {"probabilities": {"1": 1.0}}}, gold_label="1") for k in range(50)]
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    q = tmp_path / "dev.jsonl"
    q.write_text(p.read_text())
    cfg = cfg_for(caches)
    fd = cfg.DATASET.FEATURE_DATASET
    fd.TEXT_CACHE = ""

    def load(path, frac, seed=0):
        fd.NOUL_TRAIN_FRACTION, fd.NOUL_SUBSET_SEED = frac, seed
        return ds.S1FeatureDataset.__new__(ds.S1FeatureDataset), path

    def items_after(path, frac, seed=0):
        fd.NOUL_TRAIN_FRACTION, fd.NOUL_SUBSET_SEED = frac, seed
        d = ds.S1FeatureDataset.__new__(ds.S1FeatureDataset)
        # run only the item-selection part of __init__ by calling it with a stub packed cache path (features are not read in __init__)
        d.__init__(str(path), vocab, fd)
        return d.items

    full = items_after(p, 1.0)
    half = items_after(p, 0.5)
    again = items_after(p, 0.5)
    other = items_after(p, 0.5, seed=1)
    assert len(full) == 450 and [i["id"] for i in half] == [i["id"] for i in again] and [i["id"] for i in half] != [i["id"] for i in other]
    yn = [i for i in half if i["questions"]["q"]["type"] == "noul"]
    assert abs(len(yn) - 200) <= 6 and sum(1 for i in half if i["questions"]["q"]["type"] == "score") == 50          # score items untouched
    from collections import Counter
    cnt = Counter((i["gold_label"], yesno_form(i)) for i in yn)
    full_cnt = Counter((i["gold_label"], yesno_form(i)) for i in full if i["questions"]["q"]["type"] == "noul")
    assert all(abs(cnt[k] - 0.5 * v) <= 1 for k, v in full_cnt.items())                                                  # stratified
    dev = items_after(q, 0.5)                                                                                            # not a train file: untouched
    assert len(dev) == 450


def test_extra_train_items_are_appended_to_train_only(caches, vocab, subset32, tmp_path):
    import json
    cfg = cfg_for(caches)
    fd = cfg.DATASET.FEATURE_DATASET
    base = [json.loads(l) for l in open(subset32[0], encoding="utf-8")]
    tr = tmp_path / "train.jsonl"
    tr.write_text("\n".join(json.dumps(b) for b in base) + "\n")
    extra_item = dict(base[0], id=100_000_001, meta={"extra_vqa": True, "en_q": "x"})
    ex = tmp_path / "extra.jsonl"
    ex.write_text(json.dumps(extra_item) + "\n")
    fd.TEXT_CACHE = ""
    fd.EXTRA_TRAIN = str(ex)
    d = ds.S1FeatureDataset(str(tr), vocab, fd)
    assert len(d.items) == len(base) + 1 and d.items[-1]["id"] == 100_000_001
    dv = tmp_path / "dev.jsonl"
    dv.write_text(tr.read_text())
    assert len(ds.S1FeatureDataset(str(dv), vocab, fd).items) == len(base)                                               # dev never gets the extra items
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(dict(extra_item, coco_id=987654321)) + "\n")
    fd.EXTRA_TRAIN = str(bad)
    with pytest.raises(ValueError, match="not in the packed features"):
        ds.S1FeatureDataset(str(tr), vocab, fd)


# ------------------------------------------------------------------------------------------------------------- knowledge distillation (teacher targets)
def test_teacher_targets_reach_the_batch_and_kd_loss_blends_human_and_teacher(caches, vocab, subset32, tmp_path):
    import json
    cfg = cfg_for(caches)
    fd = cfg.DATASET.FEATURE_DATASET
    fd.TEXT_CACHE = ""
    base = [json.loads(l) for l in open(subset32[0], encoding="utf-8")][:12]
    tr = tmp_path / "train.jsonl"
    tr.write_text("\n".join(json.dumps(b) for b in base) + "\n")
    teach = tmp_path / "teacher.jsonl"
    from utils.viso_s1 import label_space
    rows = []
    for b in base[:-1]:                                                                          # the last item is a 'refusal': no line
        n = len(label_space(b["questions"]["q"]))
        p = [0.0] * n
        p[-1] = 1.0
        rows.append({"id": b["id"], "p": p})
    teach.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    fd.TEACHER_TRAIN = str(teach)
    d = ds.S1FeatureDataset(str(tr), vocab, fd)
    b = ds.s1_collate([d[i] for i in range(len(d))])
    assert b.teacher_target.shape == (12, 16) and b.has_teacher.tolist() == [True] * 11 + [False]
    assert float(b.teacher_target[0].sum()) == 1.0 and float(b.teacher_target[11].sum()) == 0.0
    dv = tmp_path / "dev.jsonl"
    dv.write_text(tr.read_text())
    assert "teacher_target" not in ds.S1FeatureDataset(str(dv), vocab, fd)[0]                    # dev never gets teacher targets
    torch.manual_seed(0)
    lg = torch.randn(12, 16, requires_grad=True)
    out = {"logits": lg.masked_fill(~b.space_mask, -1e9)}
    stub = types.SimpleNamespace(config=cfg, flat=False)
    cfg.LOSS.KD_WEIGHT = 0.0
    l0 = TypedDecisionTask.item_loss(stub, out, b, "soft_ce")
    cfg.LOSS.KD_WEIGHT = 0.8
    l8 = TypedDecisionTask.item_loss(stub, out, b, "soft_ce")
    from utils import viso_rlcd as rlcd
    kd = rlcd.soft_ce(out["logits"], b.space_mask, b.teacher_target)
    assert torch.allclose(l8[:11], 0.2 * l0[:11] + 0.8 * kd[:11], atol=1e-5) and torch.allclose(l8[11], l0[11])        # refusal item: human loss only
    b2 = ds.s1_collate([ds.S1FeatureDataset(str(dv), vocab, fd)[i] for i in range(12)])
    assert torch.allclose(TypedDecisionTask.item_loss(stub, out, b2, "soft_ce"), l0)                                  # no teacher fields (dev / eval): plain soft CE
