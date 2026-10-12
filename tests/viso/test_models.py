import copy

import pytest
import torch

import data_utils.datasets.s1_dataset as ds
import models.viso  # noqa: F401  (registers ViSO / FlatS1)
from builders.model_builder import META_ARCHITECTURE
from configs.utils import get_config


def model_cfg(arch="ViSO", **over):
    cfg = get_config("configs/viso_vivqav2.yaml").MODEL
    cfg.DEVICE = "cpu"
    cfg.ARCHITECTURE = arch
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def batch4(dataset32):
    idx = [0, 1, len(dataset32) // 2, len(dataset32) - 1]
    # one item per type where possible
    by_type = {}
    for i in range(len(dataset32)):
        by_type.setdefault(dataset32[i].type_id, i)
    idx = [by_type[0], by_type[1], by_type[2], by_type[2]]
    return ds.s1_collate([dataset32[i] for i in idx])


def build(vocab, arch="ViSO", **over):
    torch.manual_seed(0)
    return META_ARCHITECTURE.get(arch)(model_cfg(arch, **over), vocab).eval()


def test_vision_embedding_mask_is_derived_from_zero_rows():
    """Documents WHY the encoder builds its own mask: FeatureEmbedding masks rows that sum to zero, so an all-zero
    (ZERO_VISION) input would mask every region."""
    from builders.vision_embedding_builder import build_vision_embedding
    cfg = model_cfg().VISION_EMBEDDING
    emb = build_vision_embedding(cfg).eval()
    f = torch.zeros(1, 5, 2048)
    f[0, :3] = torch.randn(3, 2048)
    _, mask = emb(f)
    assert (mask[0, 0, 0] != 0).tolist() == [False, False, False, True, True]
    _, mask0 = emb(torch.zeros(1, 5, 2048))
    assert bool((mask0 != 0).all())


def test_zero_row_padding_does_not_change_h(vocab, batch4):
    m = build(vocab)
    b = ds.S1Batch(**{k: v for k, v in batch4.items()})
    with torch.no_grad():
        h0, _ = m.encoder(b)
        pad = 7
        b2 = ds.S1Batch(**{k: v for k, v in batch4.items()})
        b2.region_features = torch.cat([b.region_features, torch.zeros(4, pad, 2048)], dim=1)
        b2.region_mask = torch.cat([b.region_mask, torch.zeros(4, pad, dtype=torch.bool)], dim=1)
        h1, _ = m.encoder(b2)
    assert float((h0 - h1).abs().max()) < 1e-5


def test_padding_changes_nothing_even_if_padding_rows_are_garbage(vocab, batch4):
    m = build(vocab)
    b = ds.S1Batch(**{k: v for k, v in batch4.items()})
    b2 = ds.S1Batch(**{k: v for k, v in batch4.items()})
    b2.region_features = torch.cat([b.region_features, torch.randn(4, 5, 2048) * 10], dim=1)
    b2.region_mask = torch.cat([b.region_mask, torch.zeros(4, 5, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        assert float((m.encoder(b)[0] - m.encoder(b2)[0]).abs().max()) < 1e-5


def test_head_layout_and_masking(vocab, batch4):
    m = build(vocab)
    with torch.no_grad():
        out = m(batch4)
    lg = out["logits"]
    assert lg.shape == (4, 16) and lg.dtype == torch.float32 and out["h"].shape == (4, 512)
    assert bool((lg[~batch4.space_mask] == -1e9).all()) and bool(torch.isfinite(lg[batch4.space_mask]).all())
    assert int(batch4.space_mask[0].sum()) == 2 and int(batch4.space_mask[1].sum()) == 11


def test_choice_head_is_permutation_equivariant_and_ignores_padding(vocab, batch4):
    m = build(vocab)
    ch = m.choice_head
    with torch.no_grad():
        h, _ = m.encoder(batch4)
        tokens, mask = batch4.cand_tokens[2:], batch4.cand_mask[2:]
        s = ch(h[2:], tokens, mask)
        g = torch.Generator().manual_seed(3)
        perm = torch.randperm(16, generator=g)
        s_perm = ch(h[2:], tokens[:, perm], mask[:, perm])
        assert torch.allclose(s[:, perm], s_perm, atol=1e-5)
        # 2..16 candidates: drop trailing candidates (masked out) -> the kept scores must not change
        for keep in (2, 5, 9):
            mk = mask.clone()
            mk[:, keep:] = False
            sk = ch(h[2:], tokens, mk)
            sub = ch(h[2:], tokens[:, :keep].contiguous().repeat(1, 1, 1), mk[:, :keep])
            assert torch.allclose(sk[:, :keep], sub, atol=1e-5)
            assert bool((sk[:, keep:] == -1e9).all())


def test_set_attention_flag_removes_the_layers(vocab):
    m = build(vocab, SET_ATTENTION=False)
    assert m.choice_head.set_attention is None
    assert build(vocab).choice_head.set_attention is not None


def test_flat_model_restricted_layout(vocab, batch4):
    m = build(vocab, "FlatS1")
    with torch.no_grad():
        out = m(batch4)
    assert out["flat_logits"].shape == (4, vocab.total_answers) and out["logits"].shape == (4, 16)
    assert bool((out["logits"][~batch4.space_mask] == -1e9).all())
    fl = out["flat_logits"][0]
    assert torch.allclose(out["logits"][0, :2], torch.stack([fl[vocab.flat_stoi["<no>"]], fl[vocab.flat_stoi["<yes>"]]]))


def test_forward_backward_is_finite(vocab, batch4):
    for arch in ("ViSO", "FlatS1"):
        m = build(vocab, arch).train()
        out = m(batch4)
        logp = torch.log_softmax(out["logits"], dim=-1)
        loss = -(batch4.target * logp.masked_fill(~batch4.space_mask, 0)).sum(-1).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_zero_vision_runs_without_nan(vocab, batch4):
    m = build(vocab, ZERO_VISION=True)
    with torch.no_grad():
        out = m(batch4)
    assert bool(torch.isfinite(out["logits"][batch4.space_mask]).all())
