import math

import torch
from torch import nn

from utils.viso_s1 import MAX_SPACE, NEG_INF


class _MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)


class NoulHead(nn.Module):
    """Boolean decision: logits for ["false", "true"] (columns 0..1)."""
    width = 2

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.mlp = _MLP(d_model, d_model, 2, dropout)

    def forward(self, h):
        return self.mlp(h)


class ScoreHead(nn.Module):
    """Ordinal count decision: logits for levels "0".."10" (column k = level k; level 10 means >= 10).
    When count_pool=True, computes question-conditioned sigmoid objectness gates and spatial attention
    entropy/cardinality statistics over unpooled visual tokens."""
    width = 11

    def __init__(self, d_model=512, dropout=0.1, count_pool=False, count_bins=64):
        super().__init__()
        self.mlp = _MLP(d_model, d_model, 11, dropout)
        self.count_pool_enabled = bool(count_pool)
        if self.count_pool_enabled:
            with torch.random.fork_rng():
                torch.manual_seed(1337)
                self.count_q = nn.Linear(d_model, d_model)
                self.count_v = nn.Linear(d_model, d_model)
                self.count_gate = nn.Linear(d_model, count_bins)
                # count_bins (patch sum) + count_bins (patch max) + count_bins (attention-weighted variance) = count_bins * 3
                self.count_proj = nn.Sequential(
                    nn.LayerNorm(count_bins * 3),
                    nn.Linear(count_bins * 3, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 11),
                )
                nn.init.normal_(self.count_proj[-1].weight, std=0.01)
                nn.init.zeros_(self.count_proj[-1].bias)
                self.count_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, h, vision_features=None, vision_mask=None):
        logits = self.mlp(h)
        if self.count_pool_enabled and vision_features is not None:
            # Exclude index 0 (global MAP token in SigLIP-2 65-token layout) when 65 tokens are present so we sum over the 64 spatial cells
            if vision_features.size(1) == 65:
                vf = vision_features[:, 1:, :]
                vm = vision_mask[:, 1:] if vision_mask is not None else None
            else:
                vf, vm = vision_features, vision_mask
            q_ctx = torch.nn.functional.gelu(self.count_q(h)).unsqueeze(1)
            v_ctx = torch.nn.functional.gelu(self.count_v(vf))
            raw_scores = self.count_gate(v_ctx * q_ctx)  # (B, R, count_bins)
            gates = torch.sigmoid(raw_scores)
            if vm is not None:
                gates = gates * vm.unsqueeze(-1).to(gates.dtype)
                raw_masked = raw_scores.masked_fill(~vm.unsqueeze(-1), NEG_INF)
            else:
                raw_masked = raw_scores
            # 1) Sum-pooled sigmoid activations (proportional to object count)
            patch_sum = gates.sum(dim=1) / 8.0
            # 2) Max-pooled activation (existence of at least 1 match)
            patch_max = gates.max(dim=1).values
            # 3) Softmax spatial entropy across patches per channel (measures spatial dispersion across multiple objects)
            attn_p = torch.softmax(raw_masked, dim=1)
            spatial_entropy = -(attn_p * (attn_p.clamp(min=1e-7).log())).sum(dim=1) / math.log(max(vf.size(1), 2))
            count_logits = self.count_proj(torch.cat([patch_sum, patch_max, spatial_entropy], dim=-1))
            logits = logits + self.count_scale * count_logits
        return logits


class SetAttention(nn.Module):
    """Permutation-equivariant set encoder over the candidates: Transformer encoder layers WITHOUT positional
    encoding and with a key-padding mask, so candidate order and padding slots cannot change real scores."""

    def __init__(self, d_model=512, layers=2, heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, c, cand_mask):                     # c (B,K,D), cand_mask (B,K) True = real candidate
        return self.encoder(c, src_key_padding_mask=~cand_mask)


class ChoiceHead(nn.Module):
    """Pointer-style choice decision: score_i = <W_q h, c_i> / sqrt(d) over the candidate texts.
    Candidate encoder = word embedding (optionally the question embedding table, SHARE_EMBEDDING) -> masked mean
    over the candidate's tokens -> Linear(d_emb -> d) -> LayerNorm -> SetAttention (optional)."""

    width = MAX_SPACE

    def __init__(self, embedding: nn.Embedding, padding_idx: int, d_model=512, set_attention=True,
                 set_layers=2, set_heads=8, set_ff=1024, dropout=0.1, d_in=768,
                 cross_attention=False, cross_heads=8, cross_dropout=0.1):
        super().__init__()
        self.embedding, self.padding_idx, self.d_model = embedding, padding_idx, d_model
        self.proj = nn.Linear(d_in if embedding is None else embedding.embedding_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.set_attention = SetAttention(d_model, set_layers, set_heads, set_ff, dropout) if set_attention else None
        self.query = nn.Linear(d_model, d_model)
        self.cross_attention_enabled = bool(cross_attention)
        if self.cross_attention_enabled:
            with torch.random.fork_rng():
                torch.manual_seed(2026)
                self.cross_q_ctx = nn.Linear(d_model, d_model)
                self.cross_q_norm = nn.LayerNorm(d_model)
                self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=cross_heads, dropout=cross_dropout, batch_first=True)
                self.cross_norm = nn.LayerNorm(d_model)
                self.cross_cand_proj = nn.Linear(d_model, d_model)
                self.cross_gate = nn.Parameter(torch.tensor(0.2))

    def encode_candidates(self, cand_tokens, cand_mask, cand_vecs=None):
        if cand_vecs is not None:                                           # frozen pooled BamiBERT vector per candidate string (embedding=None)
            pooled = cand_vecs
        else:
            tok_mask = (cand_tokens != self.padding_idx)                    # (B,K,L)
            emb = self.embedding(cand_tokens) * tok_mask.unsqueeze(-1)
            pooled = emb.sum(2) / tok_mask.sum(2).clamp(min=1).unsqueeze(-1)    # (B,K,d_emb)
        c = self.norm(self.proj(pooled))
        if self.set_attention is not None:
            c = self.set_attention(c, cand_mask)
        return c

    def forward(self, h, cand_tokens, cand_mask, cand_vecs=None, vision_features=None, vision_mask=None):
        c = self.encode_candidates(cand_tokens, cand_mask, cand_vecs)
        q = self.query(h)
        scores = torch.einsum("bd,bkd->bk", q, c) / math.sqrt(self.d_model)
        if self.cross_attention_enabled and vision_features is not None:
            key_pad_mask = ~vision_mask if vision_mask is not None else None
            q_ctx = self.cross_q_ctx(h).unsqueeze(1)
            c_query = self.cross_q_norm(c + q_ctx)
            attn_out, _ = self.cross_attn(query=c_query, key=vision_features, value=vision_features, key_padding_mask=key_pad_mask)
            v_grounded = self.cross_norm(attn_out)
            c_modulated = self.cross_cand_proj(c * torch.sigmoid(q_ctx))
            # 1) Cross-attended visual token verification score
            attn_scores = torch.einsum("bkd,bkd->bk", c_modulated, v_grounded) / math.sqrt(self.d_model)
            # 2) Direct FILIP-style max-patch alignment score across spatial tokens
            patch_sim = torch.einsum("bkd,brd->bkr", c_modulated, vision_features) / math.sqrt(self.d_model)
            if vision_mask is not None:
                patch_sim = patch_sim.masked_fill(~vision_mask.unsqueeze(1), NEG_INF)
            max_patch_scores = patch_sim.max(dim=-1).values
            scores = scores + self.cross_gate * (0.5 * attn_scores + 0.5 * max_patch_scores)
        return scores.masked_fill(~cand_mask, NEG_INF)


class EscalationHead:
    """Post-hoc System-1 -> System-2 deferral features; the logistic regression itself lives in
    utils/viso_calibration.py (it is fitted on calibrated probabilities, not trained with the network)."""
    FEATURES = ("max_prob", "entropy_norm", "margin", "type_onehot(3)", "max_prob*type_onehot(3)")
