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
    """Ordinal count decision: logits for levels "0".."10" (column k = level k; level 10 means >= 10)."""
    width = 11

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.mlp = _MLP(d_model, d_model, 11, dropout)

    def forward(self, h):
        return self.mlp(h)


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
            self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=cross_heads, dropout=cross_dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(d_model)
            self.cross_gate = nn.Parameter(torch.tensor(1.0))

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
        if self.cross_attention_enabled and vision_features is not None:
            key_pad_mask = ~vision_mask if vision_mask is not None else None
            attn_out, _ = self.cross_attn(query=c, key=vision_features, value=vision_features, key_padding_mask=key_pad_mask)
            c = self.cross_norm(c + self.cross_gate * attn_out)
        q = self.query(h)
        scores = torch.einsum("bd,bkd->bk", q, c) / math.sqrt(self.d_model)
        return scores.masked_fill(~cand_mask, NEG_INF)


class EscalationHead:
    """Post-hoc System-1 -> System-2 deferral features; the logistic regression itself lives in
    utils/viso_calibration.py (it is fitted on calibrated probabilities, not trained with the network)."""
    FEATURES = ("max_prob", "entropy_norm", "margin", "type_onehot(3)", "max_prob*type_onehot(3)")
