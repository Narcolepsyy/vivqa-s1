"""Smoke tests for MODEL.TEXT_VISION_GROUNDING.

Tiny synthetic tensors only. They check shapes, padding, gradients, the config diff, and checkpoint
compatibility. They are not accuracy evidence and they do not use ViVQA-S1 items.
"""
import types

import pytest
import torch

import models.viso  # noqa: F401
from builders.model_builder import META_ARCHITECTURE
from configs.utils import get_config
from models.viso.encoder import TextVisionGrounding, require_text_vision_grounding
from tasks.typed_decision_task import TypedDecisionTask
from train_viso import load_config
from utils.instance import Instance
from utils.viso_efficiency import param_counts

D = 768
HEADS_D = 512
# 4 linear maps + LayerNorm + one gate, D_MODEL 512
EXPECTED_NEW_PARAMS = 4 * (HEADS_D * HEADS_D + HEADS_D) + 2 * HEADS_D + 1

OFF_CONFIGS = (
    "viso_siglip2",
    "viso_siglip2_bami",
    "viso_siglip2_bami_crossattn",
    "viso_siglip2_tsig",
    "viso_flat_siglip2",
    "viso_flat_siglip2_bami",
    "viso_flat_siglip2_tsig",
    "viso_ft_bami",
)


class StubVocab:
    padding_idx = 0
    total_answers = 13

    def __len__(self):
        return 40


def leaves(node, prefix=""):
    out = {}
    for key, value in node.items():
        name = f"{prefix}.{key}" if prefix else key
        if hasattr(value, "items"):
            out.update(leaves(value, name))
        else:
            out[name] = value
    return out


def build(flag, arch="ViSO", path="configs/viso_siglip2_bami.yaml", seed=0, **over):
    cfg = get_config(path)
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.TEXT_VISION_GROUNDING = flag
    for key, value in over.items():
        setattr(cfg.MODEL, key, value)
    torch.manual_seed(seed)
    model = META_ARCHITECTURE.get(arch)(cfg.MODEL, StubVocab())
    return cfg, model


def batch(b=4, regions=6, length=5, seed=0, cached=True):
    g = torch.Generator().manual_seed(seed)
    region_mask = torch.ones(b, regions, dtype=torch.bool)
    region_mask[:, -2:] = False
    type_id = torch.tensor([0, 1, 2, 2])
    space = torch.zeros(b, 16, dtype=torch.bool)
    space[0, :2] = True
    space[1, :11] = True
    space[2:] = True
    fields = dict(
        region_features=torch.randn(b, regions, D, generator=g),
        region_mask=region_mask,
        region_boxes=torch.rand(b, regions, 4, generator=g),
        type_id=type_id,
        space_mask=space,
        cand_tokens=torch.randint(1, 20, (b, 16, 6), generator=g),
        cand_mask=space & (type_id[:, None] == 2),
        target=torch.zeros(b, 16),
        gold_idx=torch.zeros(b, dtype=torch.long),
        image_id=torch.arange(b),
        flat_index=torch.arange(16).view(1, 16).expand(b, 16).contiguous().clamp(max=12),
    )
    if cached:
        qmask = torch.ones(b, length, dtype=torch.bool)
        qmask[:, -1] = False
        fields.update(
            question_states=torch.randn(b, length, D, generator=g),
            question_state_mask=qmask,
            cand_vecs=torch.randn(b, 16, D, generator=g),
        )
    else:
        tokens = torch.randint(1, 20, (b, length), generator=g)
        tokens[:, -1] = 0
        fields["question_tokens"] = tokens
    return Instance(**fields)


def clone(batch_, **over):
    data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_.items()}
    data.update(over)
    return Instance(**data)


def test_configs_default_off_and_the_example_flips_only_the_flag_and_the_run_name():
    for name in OFF_CONFIGS:
        cfg = get_config(f"configs/{name}.yaml")
        assert cfg.MODEL.TEXT_VISION_GROUNDING is False
    legacy = get_config("configs/viso_vivqav2.yaml")
    flat_legacy = get_config("configs/viso_flat_vivqav2.yaml")
    assert not getattr(legacy.MODEL, "TEXT_VISION_GROUNDING", False)
    assert not getattr(flat_legacy.MODEL, "TEXT_VISION_GROUNDING", False)
    butd = get_config("configs/viso_butd_siglip2.yaml")
    assert not hasattr(butd.MODEL, "TEXT_VISION_GROUNDING")

    base = leaves(get_config("configs/viso_siglip2_bami.yaml"))
    example = leaves(get_config("configs/viso_siglip2_bami_tvground.yaml"))
    changed = {k for k in base.keys() | example.keys() if base.get(k) != example.get(k)}
    assert changed == {"MODEL.NAME", "MODEL.TEXT_VISION_GROUNDING"}
    assert example["MODEL.NAME"] == "bami_tvground" and example["MODEL.TEXT_VISION_GROUNDING"] is True

    loaded = load_config(
        "configs/viso_siglip2_bami.yaml",
        ["MODEL.TEXT_VISION_GROUNDING", "True", "MODEL.DEVICE", "cpu"],
    )
    assert loaded.MODEL.TEXT_VISION_GROUNDING is True


def test_block_rejects_a_head_count_that_does_not_divide_and_a_foreign_encoder():
    with pytest.raises(ValueError, match="divide"):
        TextVisionGrounding(10, 3)
    with pytest.raises(ValueError, match="ViSOEncoder"):
        require_text_vision_grounding(types.SimpleNamespace(TEXT_VISION_GROUNDING=True), torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="SAAAEncoder"):
        build(True, **{"ENCODER_TYPE": "SAAA"})


def test_isolated_block_masks_pads_empty_rows_and_takes_gradient():
    torch.manual_seed(0)
    block = TextVisionGrounding(32, 4)
    text = torch.randn(2, 5, 32)
    vision = torch.randn(2, 7, 32)
    text_pad = torch.zeros(2, 5, dtype=torch.bool)
    text_pad[:, -2:] = True
    vision_pad = torch.zeros(2, 7, dtype=torch.bool)
    vision_pad[:, -3:] = True

    block.gate.data.zero_()
    assert torch.equal(block(text, vision, text_pad, vision_pad), text)

    with torch.no_grad():
        block.gate.fill_(1)
    text2 = text.clone()
    text2[:, -2:] = torch.randn(2, 2, 32) * 20
    vision2 = vision.clone()
    vision2[:, -3:] = torch.randn(2, 3, 32) * 20
    y = block(text, vision, text_pad, vision_pad)
    y_pad = block(text2, vision2, text_pad, vision_pad)
    assert torch.allclose(y[:, :-2], y_pad[:, :-2], atol=1e-6)
    assert torch.allclose(y[:, -2:], text[:, -2:], atol=1e-6)
    real_vision = vision.clone()
    real_vision[:, :-3] += 1
    assert not torch.allclose(y, block(text, real_vision, text_pad, vision_pad))

    empty = vision_pad.clone()
    empty[1] = True
    y_empty = block(text, vision, text_pad, empty)
    assert torch.isfinite(y_empty).all()
    assert torch.allclose(y_empty[1], text[1], atol=1e-6)
    assert torch.equal(block(text[:, :0], vision, text_pad[:, :0], vision_pad), text[:, :0])

    text_g = text.detach().requires_grad_(True)
    vision_g = vision.detach().requires_grad_(True)
    out = block(text_g, vision_g, text_pad, vision_pad)
    out[:, :-2].sum().backward()
    assert float(block.q_proj.weight.grad.abs().sum()) > 0
    assert float(block.out_proj.weight.grad.abs().sum()) > 0
    assert float(vision_g.grad[:, :-3].abs().sum()) > 0
    assert float(vision_g.grad[:, -3:].abs().sum()) == 0
    assert float(text_g.grad[:, :-2].abs().sum()) > 0
    assert float(text_g.grad[:, -2:].abs().sum()) == 0

    block.zero_grad(set_to_none=True)
    with torch.no_grad():
        block.gate.zero_()
    text_g2 = text.detach().requires_grad_(True)
    out0 = block(text_g2, vision.detach(), text_pad, vision_pad)
    out0.sum().backward()
    assert float(block.gate.grad.abs()) > 0
    assert block.q_proj.weight.grad is None or float(block.q_proj.weight.grad.abs().sum()) == 0


def test_flag_off_matches_flag_on_at_gate_zero_including_train_mode_dropout():
    b = batch()
    _, base = build(False)
    _, grounded = build(True)
    assert not hasattr(base.encoder, "tv_ground")
    shared = set(base.state_dict())
    assert shared <= set(grounded.state_dict())
    extra = set(grounded.state_dict()) - shared
    assert extra and all(k.startswith("encoder.tv_ground.") for k in extra)
    for key in shared:
        assert torch.equal(base.state_dict()[key], grounded.state_dict()[key])
    assert float(grounded.encoder.tv_ground.gate.detach()) == 0

    for train in (False, True):
        base.train(train)
        grounded.train(train)
        torch.manual_seed(7)
        base_out = base(b)
        torch.manual_seed(7)
        new_out = grounded(b)
        assert torch.equal(base_out["h"], new_out["h"])
        assert torch.equal(base_out["logits"], new_out["logits"])
        if base_out["break_probs"] is None:
            assert new_out["break_probs"] is None
        else:
            assert torch.equal(base_out["break_probs"], new_out["break_probs"])

    counts = param_counts(grounded)
    assert counts["text_vision_grounding"] == EXPECTED_NEW_PARAMS
    assert "text_vision_grounding" not in param_counts(base)
    assert counts["encoder"] - param_counts(base)["encoder"] == EXPECTED_NEW_PARAMS


def test_baseline_checkpoint_is_strict_compatible_only_when_the_flag_is_off():
    b = batch()
    _, base = build(False, seed=0)
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    reloaded = build(False, seed=1)[1]
    reloaded.load_state_dict(sd, strict=True)

    grounded = build(True, seed=4)[1]
    with pytest.raises(RuntimeError, match="Missing key"):
        grounded.load_state_dict(sd, strict=True)
    missing = grounded.load_state_dict(sd, strict=False)
    assert missing.unexpected_keys == []
    assert set(missing.missing_keys) == {k for k in grounded.state_dict() if k.startswith("encoder.tv_ground.")}
    base.eval()
    grounded.eval()
    assert torch.equal(base.encoder(b)[0], grounded.encoder(b)[0])

    with pytest.raises(RuntimeError, match="Unexpected key"):
        build(False, seed=2)[1].load_state_dict(build(True, seed=3)[1].state_dict(), strict=True)


def test_open_gate_uses_real_tokens_and_ignores_gold_pads_and_image_id():
    b = batch()
    _, base = build(False, seed=0)
    _, model = build(True, seed=0)
    with torch.no_grad():
        model.encoder.tv_ground.gate.fill_(1)
    base.eval()
    model.eval()
    h_base = base.encoder(b)
    h_open = model.encoder(b)
    assert torch.equal(h_base.vision_features, h_open.vision_features)
    assert torch.equal(h_base.break_probs, h_open.break_probs)
    assert not torch.allclose(h_base.h, h_open.h)

    garbage_vision = clone(b)
    garbage_vision.region_features = b.region_features.clone()
    garbage_vision.region_features[~b.region_mask] = torch.randn(int((~b.region_mask).sum()), D) * 50
    garbage_vision.region_boxes = b.region_boxes.clone()
    garbage_vision.region_boxes[~b.region_mask] = 9
    extra_q = clone(b)
    extra_q.question_states = torch.cat([b.question_states, torch.randn(4, 3, D) * 50], 1)
    extra_q.question_state_mask = torch.cat([b.question_state_mask, torch.zeros(4, 3, dtype=torch.bool)], 1)
    with torch.no_grad():
        h = model.encoder(b).h
        assert torch.allclose(h, model.encoder(garbage_vision).h, atol=1e-5)
        assert torch.allclose(h, model.encoder(extra_q).h, atol=1e-5)

    changed = clone(b)
    changed.region_features = b.region_features.clone()
    changed.region_features[b.region_mask] += 0.75
    # kill the vision summand so a change in h can only come from the grounded question stream
    with torch.no_grad():
        model.encoder.vision_proj.weight.zero_()
        model.encoder.vision_proj.bias.zero_()
        base.encoder.vision_proj.weight.zero_()
        base.encoder.vision_proj.bias.zero_()
    with torch.no_grad():
        assert torch.allclose(base.encoder(b).h, base.encoder(changed).h, atol=1e-5)
        assert not torch.allclose(model.encoder(b).h, model.encoder(changed).h, atol=1e-4)

    other_q = clone(b)
    other_q.question_states = b.question_states.clone()
    other_q.question_states[b.question_state_mask] += 0.5
    with torch.no_grad():
        assert not torch.allclose(model.encoder(b).h, model.encoder(other_q).h, atol=1e-4)

    logits = model(b)["logits"]
    relabeled = clone(b, gold_idx=torch.tensor([1, 4, 7, 0]), target=torch.rand(4, 16), image_id=torch.tensor([9, 8, 7, 6]))
    assert torch.equal(logits, model(relabeled)["logits"])
    assert logits.shape == (4, 16)
    assert bool((logits[~b.space_mask] == -1e9).all())
    assert torch.isfinite(logits[b.space_mask]).all()


def test_zero_vision_ignores_image_content_and_gradients_reach_the_block():
    b = batch()
    other = clone(b)
    other.region_features = torch.randn_like(b.region_features)
    other.region_boxes = torch.rand_like(b.region_boxes)
    _, model = build(True, seed=2, ZERO_VISION=True)
    with torch.no_grad():
        model.encoder.tv_ground.gate.fill_(1)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model.encoder(b).h, model.encoder(other).h)

    _, train_model = build(True, seed=3)
    with torch.no_grad():
        train_model.encoder.tv_ground.gate.fill_(0.5)
    train_model.train()
    out = train_model(b)
    out["logits"].masked_fill(~b.space_mask, 0).pow(2).mean().backward()
    block = train_model.encoder.tv_ground
    assert float(block.q_proj.weight.grad.abs().sum()) > 0
    assert float(block.k_proj.weight.grad.abs().sum()) > 0
    assert float(block.v_proj.weight.grad.abs().sum()) > 0
    assert float(block.gate.grad.abs()) > 0
    assert float(train_model.noul_head.mlp.net[0].weight.grad.abs().sum()) > 0
    assert float(train_model.score_head.mlp.net[0].weight.grad.abs().sum()) > 0
    assert float(train_model.choice_head.query.weight.grad.abs().sum()) > 0


def test_grounding_parameters_stay_on_the_encoder_learning_rate():
    cfg, model = build(True)
    stub = types.SimpleNamespace(config=cfg, model=model)
    opt = TypedDecisionTask._make_optimizer(stub)
    names = {id(p): n for n, p in model.named_parameters()}
    groups = [{names[id(p)] for p in g["params"]} for g in opt.param_groups]
    assert opt.param_groups[0]["lr"] == cfg.TRAINING.LR_ENCODER
    assert "encoder.tv_ground.gate" in groups[0]
    assert "encoder.tv_ground.q_proj.weight" in groups[0]
    assert all("tv_ground" not in n for n in groups[1])
    cfg.TRAINING.VISION_PATH_AS_HEADS = True
    opt = TypedDecisionTask._make_optimizer(types.SimpleNamespace(config=cfg, model=model))
    names = {id(p): n for n, p in model.named_parameters()}
    enc = {names[id(p)] for p in opt.param_groups[0]["params"]}
    assert "encoder.tv_ground.gate" in enc
    assert "encoder.vision_proj.weight" not in enc


def test_flat_and_lstm_paths_keep_the_gate_zero_identity():
    b = batch()
    _, flat_off = build(False, arch="FlatS1", path="configs/viso_flat_siglip2_bami.yaml")
    _, flat_on = build(True, arch="FlatS1", path="configs/viso_flat_siglip2_bami.yaml")
    assert not hasattr(flat_off.encoder, "tv_ground") and flat_on.encoder.text_vision_grounding
    flat_off.eval()
    flat_on.eval()
    with torch.no_grad():
        assert torch.equal(flat_off(b)["logits"], flat_on(b)["logits"])

    tokens = batch(cached=False)
    _, lstm_off = build(False, path="configs/viso_siglip2.yaml", seed=5)
    _, lstm_on = build(True, path="configs/viso_siglip2.yaml", seed=5)
    lstm_off.eval()
    lstm_on.eval()
    with torch.no_grad():
        assert torch.equal(lstm_off(tokens)["h"], lstm_on(tokens)["h"])
        lstm_on.encoder.tv_ground.gate.fill_(1)
        padded = clone(tokens)
        padded.question_tokens = torch.cat([tokens.question_tokens, torch.zeros(4, 4, dtype=torch.long)], 1)
        h = lstm_on.encoder(tokens).h
        assert torch.allclose(h, lstm_on.encoder(padded).h, atol=1e-5)
        assert torch.isfinite(h).all()
