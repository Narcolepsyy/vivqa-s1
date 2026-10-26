"""Lightweight VQA System-One Baselines on cached SigLIP-2 + BamiBERT (or scratch) features.

Provides a unified family of classical VQA encoders adapted to the ViVQA-S1 interface:
  - `BUTD_Bami` (`ENCODER_TYPE: BUTD_Bami`): Bottom-Up and Top-Down Attention (Anderson et al., CVPR 2018)
  - `SAAA` (`ENCODER_TYPE: SAAA`): Show, Ask, Attend, and Answer (Kazemi & Elqursh, CVPR 2017)
  - `SAN` (`ENCODER_TYPE: SAN`): Stacked Attention Networks (Yang et al., CVPR 2016)
  - `BAN` (`ENCODER_TYPE: BAN`): Bilinear Attention Networks (Kim et al., NeurIPS 2018)
  - `HieCoAtt` (`ENCODER_TYPE: HieCoAtt`): Hierarchical Question-Image Co-Attention (Lu et al., NeurIPS 2016)
  - `VisualBERT` (`ENCODER_TYPE: VisualBERT`): Single-stream Early-Fusion Transformer (Li et al., 2019)
  - `ViLBERT` (`ENCODER_TYPE: ViLBERT`): Two-stream Co-Attention Transformer (Lu et al., NeurIPS 2019)
  - `LXMERT` (`ENCODER_TYPE: LXMERT`): Cross-Modality Transformer (Tan & Bansal, EMNLP 2019)
  - `MCAN` (`ENCODER_TYPE: MCAN`): Deep Modular Co-Attention Network without Constituent parser (Yu et al., CVPR 2019)
  - `PhrasalMCAN` (`ENCODER_TYPE: PhrasalMCAN`): Phrasal Lexeme Co-Text MCAN (ViWordFormer)

Every encoder returns `EncoderOutput(h, None, vision_features=vision_features, vision_mask=region_mask)`
and supports both:
  1. `ViSO` mode (Typed heads: NoulHead, ScoreHead, ChoiceHead with SetAttention / CrossAttention+FILIP)
  2. `FlatS1` mode (3,500-way flat answer vocabulary softmax restricted at inference to the S1 candidate space)
"""

import torch
from torch import nn
from torch.nn import functional as F

from builders.encoder_builder import build_encoder
from builders.model_builder import META_ARCHITECTURE
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from models.modules.attentions import MultiHeadAttention
from models.modules.encoders import Encoder, GuidedAttentionEncoder
from models.modules.pos_embeddings import SinusoidPositionalEmbedding
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.phrasal_mcan.encoder import PhrasalEncoder, PhrasalGuidedAttentionEncoder

from .encoder import MASK_FILL, EncoderOutput


def _cfg(config, name, default):
    return getattr(config, name, default)


class AttrReduceMLP(nn.Module):
    """2-layer attention-pooling MLP matching MCAN / ViSO (`fc1 -> ReLU -> Dropout -> fc2`)."""

    def __init__(self, d_model: int, dropout: float = 0.1, init_std: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, 1)
        if init_std > 0:
            nn.init.normal_(self.fc2.weight, std=init_std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.relu(self.fc1(features))))


class BaseS1Encoder(nn.Module):
    """Shared input adapter for cached SigLIP-2 region tokens + cached BamiBERT (or scratch) text tokens.

    Names (`vision_embedding`, `vision_proj`, `vision_attr_reduce`, `box_proj`, `feature_norm`, `prod_v`,
    `text_embedding.proj`) intentionally match `ViSOEncoder` so `TypedDecisionTask._make_optimizer` applies
    `VISION_PATH_AS_HEADS` and `TEXT_PROJ_AS_HEADS` identically across all backbones.
    """

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__()
        self.d_model = int(config.D_MODEL)
        self.zero_vision = bool(_cfg(config, "ZERO_VISION", False))
        self.cached_text = config.TEXT_EMBEDDING.ARCHITECTURE == "CachedBertEmbedding"
        self.pretrained_residual = bool(_cfg(config, "PRETRAINED_RESIDUAL", False))
        self.fusion_dropout = nn.Dropout(float(_cfg(config, "FUSION_DROPOUT", 0.0)))
        self.fusion_product = bool(_cfg(config, "FUSION_PRODUCT", False))

        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)
        self.feature_norm = (
            nn.LayerNorm(config.VISION_EMBEDDING.D_FEATURE)
            if bool(_cfg(config.VISION_EMBEDDING, "FEATURE_NORM", False))
            else None
        )
        self.use_boxes = bool(_cfg(config.VISION_EMBEDDING, "USE_BOXES", False))
        if self.use_boxes:
            self.box_proj = nn.Linear(4, config.VISION_EMBEDDING.D_MODEL)

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)

        if keep_classify:
            self.classify = nn.Linear(self.d_model, vocab.total_answers)

    def embed_inputs(self, batch):
        """Return `(vision_features, vision_padding_mask, region_mask, text_features, text_padding_mask, text_bool_mask)`."""
        regions, region_mask = batch.region_features, batch.region_mask
        if self.zero_vision:
            regions = torch.zeros_like(regions)
        if self.feature_norm is not None:
            regions = self.feature_norm(regions)
        ve = self.vision_embedding
        projected = ve.proj(regions)
        if self.use_boxes:
            boxes = batch.region_boxes
            projected = projected + self.box_proj(torch.zeros_like(boxes) if self.zero_vision else boxes)
        vision_features = ve.dropout(ve.gelu(projected))
        vision_padding_mask = (~region_mask).float().mul(MASK_FILL)[:, None, None, :]

        if self.cached_text:
            text_features, (text_padding_mask, _) = self.text_embedding(
                batch.question_states, batch.question_state_mask
            )
            text_bool_mask = batch.question_state_mask
        else:
            text_features, (text_padding_mask, _) = self.text_embedding(batch.question_tokens)
            text_bool_mask = (text_padding_mask[:, 0, 0, :] == 0)

        return vision_features, vision_padding_mask, region_mask, text_features, text_padding_mask, text_bool_mask


class DualPoolS1Encoder(BaseS1Encoder):
    """Base class for two-stream / co-attention encoders that pool `(vision_features, text_features)` into `h`."""

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        d = self.d_model
        std = float(_cfg(config, "ATTR_REDUCE_INIT_STD", 0.0))
        v_drop = float(getattr(config.VISION_ATTR_REDUCE, "DROPOUT", 0.1))
        t_drop = float(getattr(config.TEXT_ATTR_REDUCE, "DROPOUT", 0.1))
        self.vision_attr_reduce = AttrReduceMLP(d, v_drop, std)
        self.text_attr_reduce = AttrReduceMLP(d, t_drop, std)
        self.vision_proj = nn.Linear(d, d)
        self.text_proj = nn.Linear(d, d)
        if self.fusion_product:
            self.prod_q = nn.Linear(d, d)
            self.prod_v = nn.Linear(d, d)
        self.layer_norm = nn.LayerNorm(d)

    def pool_and_fuse(self, vision_features, vision_padding_mask, region_mask, text_features, text_padding_mask):
        v_logits = self.vision_attr_reduce(vision_features) + vision_padding_mask[:, 0, 0, :, None]
        attended_vision = F.softmax(v_logits, dim=1)
        t_logits = self.text_attr_reduce(text_features) + text_padding_mask[:, 0, 0, :, None]
        attended_text = F.softmax(t_logits, dim=1)
        weighted_vision = (vision_features * attended_vision).sum(dim=1)
        weighted_text = (text_features * attended_text).sum(dim=1)
        fused = self.vision_proj(weighted_vision) + self.text_proj(weighted_text)
        if self.fusion_product:
            fused = fused + self.prod_q(weighted_text) * self.prod_v(weighted_vision)
        h = self.fusion_dropout(self.layer_norm(fused))
        return EncoderOutput(h, None, vision_features=vision_features, vision_mask=region_mask)


# ---------------------------------------------------------------------------
# 1. MCAN (Yu et al., CVPR 2019) — Deep Modular Co-Attention without Constituent module
# ---------------------------------------------------------------------------
class MCANEncoder(DualPoolS1Encoder):
    """Standard MCAN (Self-Attention on text + Guided-Attention on vision) without Constituent tree parser."""

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        self.self_encoder = Encoder(config.SELF_ENCODER)
        self.guided_encoder = GuidedAttentionEncoder(config.GUIDED_ENCODER)

    def forward(self, batch):
        v, v_mask, r_mask, t, t_mask, _ = self.embed_inputs(batch)
        v_init, t_init = v, t
        t = self.self_encoder(features=t, padding_mask=t_mask)
        if self.pretrained_residual:
            t = t + t_init
        v = self.guided_encoder(
            vision_features=v, vision_padding_mask=v_mask,
            language_features=t, language_padding_mask=t_mask,
        )
        if self.pretrained_residual:
            v = v + v_init
        return self.pool_and_fuse(v, v_mask, r_mask, t, t_mask)


# ---------------------------------------------------------------------------
# 2. PhrasalMCAN (ViWordFormer) — Bilinear Phrasal Lexeme + Co-Text Gating MCAN
# ---------------------------------------------------------------------------
def _make_phrasal_cfg(base_enc_cfg, is_guided: bool = False):
    """Clone `SELF_ENCODER` / `GUIDED_ENCODER` CfgNode with `PhrasalScaledDotProductAttention`."""
    from copy import deepcopy
    cfg = deepcopy(base_enc_cfg)
    cfg.SELF_ATTENTION.ARCHITECTURE = "PhrasalScaledDotProductAttention"
    if is_guided and hasattr(cfg, "GUIDED_ATTENTION"):
        cfg.GUIDED_ATTENTION.ARCHITECTURE = "PhrasalScaledDotProductAttention"
    return cfg


class PhrasalMCANEncoder(DualPoolS1Encoder):
    """PhrasalMCAN (ViWordFormer): Bilinear Phrasal Score propagation + Co-Text gating across SA and SGA layers."""

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        self.self_encoder = PhrasalEncoder(_make_phrasal_cfg(config.SELF_ENCODER, is_guided=False))
        self.guided_encoder = PhrasalGuidedAttentionEncoder(_make_phrasal_cfg(config.GUIDED_ENCODER, is_guided=True))

    def forward(self, batch):
        v, v_mask, r_mask, t, t_mask, _ = self.embed_inputs(batch)
        v_init, t_init = v, t
        t = self.self_encoder(features=t, padding_mask=t_mask)
        if self.pretrained_residual:
            t = t + t_init
        v = self.guided_encoder(
            vision_features=v, vision_padding_mask=v_mask,
            language_features=t, language_padding_mask=t_mask,
        )
        if self.pretrained_residual:
            v = v + v_init
        return self.pool_and_fuse(v, v_mask, r_mask, t, t_mask)


# ---------------------------------------------------------------------------
# 3. BUTD_Bami (Anderson et al., CVPR 2018) — Bottom-Up Top-Down Attention on cached features
# ---------------------------------------------------------------------------
class ButdBamiEncoder(BaseS1Encoder):
    """Bottom-Up Top-Down Attention using the shared SigLIP-2 + BamiBERT feature interface:
    1-layer BiGRU (or attention pool) over token embeddings -> question-guided multi-glimpse ($G=2$)
    attention over visual tokens -> Hadamard fusion `ReLU(W_q q) * ReLU(W_v v_att)` -> 2-layer MLP + LayerNorm.
    """

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        d = self.d_model
        self.glimpses = int(_cfg(config, "BUTD_GLIMPSES", 2))
        self.gru = nn.GRU(d, d, batch_first=True)
        self.att_q = nn.Linear(d, d)
        self.att_v = nn.Linear(d, d)
        self.att_out = nn.Linear(d, self.glimpses)
        self.fuse_q = nn.Linear(d, d)
        self.fuse_v = nn.Linear(d * self.glimpses, d)
        drop = max(float(_cfg(config, "FUSION_DROPOUT", 0.0)), 0.1)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, batch):
        v, _, r_mask, t, _, t_bool = self.embed_inputs(batch)
        out, _ = self.gru(t)
        last = (t_bool.sum(1) - 1).clamp(min=0)
        q = out[torch.arange(out.shape[0], device=out.device), last]
        if self.pretrained_residual:
            denom = t_bool.sum(1, keepdim=True).clamp(min=1).to(t.dtype)
            q = q + (t * t_bool.unsqueeze(-1)).sum(1) / denom
        joint = F.relu(self.att_q(q))[:, None, :] * F.relu(self.att_v(v))
        logits = self.att_out(joint).masked_fill(~r_mask[..., None], MASK_FILL)
        att = torch.softmax(logits, dim=1)
        v_att = torch.einsum("brg,brd->bgd", att, v).flatten(1)
        h = F.relu(self.fuse_q(q)) * F.relu(self.fuse_v(v_att))
        h = self.fusion_dropout(self.layer_norm(self.mlp(h)))
        return EncoderOutput(h, None, vision_features=v, vision_mask=r_mask)


# ---------------------------------------------------------------------------
# 4. SAAA (Kazemi & Elqursh, CVPR 2017) — Show, Ask, Attend, and Answer
# ---------------------------------------------------------------------------
class SAAAEncoder(BaseS1Encoder):
    """Show, Ask, Attend, and Answer (Kazemi & Elqursh, CVPR 2017):
    L2-normalized visual features + 1-layer LSTM/GRU question processor + $G=2$ co-attention glimpses
    + concatenated `[v_glimpses; q]` 2-layer MLP + LayerNorm.
    """

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        d = self.d_model
        self.glimpses = int(_cfg(config, "SAAA_GLIMPSES", 2))
        self.lstm = nn.LSTM(d, d, num_layers=1, batch_first=True)
        self.v_conv = nn.Linear(d, d, bias=False)
        self.q_lin = nn.Linear(d, d)
        self.x_conv = nn.Linear(d, self.glimpses)
        drop = max(float(_cfg(config, "FUSION_DROPOUT", 0.0)), 0.1)
        self.drop = nn.Dropout(drop)
        self.vision_proj = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear((self.glimpses + 1) * d, 2 * d),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(2 * d, d),
        )
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, batch):
        v, _, r_mask, t, _, t_bool = self.embed_inputs(batch)
        v_norm = F.normalize(v, p=2, dim=-1)
        out, _ = self.lstm(torch.tanh(self.drop(t)))
        last = (t_bool.sum(1) - 1).clamp(min=0)
        q = out[torch.arange(out.shape[0], device=out.device), last]
        if self.pretrained_residual:
            denom = t_bool.sum(1, keepdim=True).clamp(min=1).to(t.dtype)
            q = q + (t * t_bool.unsqueeze(-1)).sum(1) / denom
        x = F.relu(self.v_conv(self.drop(v_norm)) + self.q_lin(self.drop(q))[:, None, :])
        att_logits = self.x_conv(self.drop(x)).masked_fill(~r_mask[..., None], MASK_FILL)
        att = F.softmax(att_logits, dim=1)
        v_att = torch.einsum("brg,brd->bgd", att, v_norm).flatten(1)
        combined = torch.cat([v_att, q], dim=-1)
        h = self.fusion_dropout(self.layer_norm(self.vision_proj(combined)))
        return EncoderOutput(h, None, vision_features=v, vision_mask=r_mask)


# ---------------------------------------------------------------------------
# 5. SAN (Yang et al., CVPR 2016) — Stacked Attention Networks
# ---------------------------------------------------------------------------
class SANEncoder(BaseS1Encoder):
    """Stacked Attention Networks (Yang et al., CVPR 2016):
    Multi-step ($K=2$) iterative visual attention refinement conditioned on evolving query vector $u_k$.
    """

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        d = self.d_model
        self.hops = int(_cfg(config, "SAN_HOPS", 2))
        self.text_attr_reduce = AttrReduceMLP(d, float(getattr(config.TEXT_ATTR_REDUCE, "DROPOUT", 0.1)))
        self.w_v = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.hops)])
        self.w_u = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(self.hops)])
        self.w_p = nn.ModuleList([nn.Linear(d, 1, bias=True) for _ in range(self.hops)])
        self.hop_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.hops)])
        drop = max(float(_cfg(config, "FUSION_DROPOUT", 0.0)), 0.1)
        self.drop = nn.Dropout(drop)
        self.vision_proj = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, batch):
        v, v_mask, r_mask, t, t_mask, _ = self.embed_inputs(batch)
        t_logits = self.text_attr_reduce(t) + t_mask[:, 0, 0, :, None]
        u = (t * F.softmax(t_logits, dim=1)).sum(dim=1)
        for k in range(self.hops):
            joint = torch.tanh(self.w_v[k](self.drop(v)) + self.w_u[k](self.drop(u))[:, None, :])
            scores = self.w_p[k](self.drop(joint)) + v_mask[:, 0, 0, :, None]
            att = F.softmax(scores, dim=1)
            v_ctx = (v * att).sum(dim=1)
            u = self.hop_norms[k](u + v_ctx)
        h = self.fusion_dropout(self.layer_norm(u + self.vision_proj(u)))
        return EncoderOutput(h, None, vision_features=v, vision_mask=r_mask)


# ---------------------------------------------------------------------------
# 6. BAN (Kim et al., NeurIPS 2018) — Bilinear Attention Networks
# ---------------------------------------------------------------------------
class BANLayer(nn.Module):
    """Single low-rank bilinear attention glimpse with residual language update."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.u_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_net = nn.Linear(d_model, d_model)
        self.v_net = nn.Linear(d_model, d_model)
        self.p_net = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, v: torch.Tensor, q_bool_mask: torch.Tensor, r_bool_mask: torch.Tensor):
        q_att = F.relu(self.u_proj(self.drop(q)))
        v_att = F.relu(self.v_proj(self.drop(v)))
        scores = torch.bmm(q_att, v_att.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        pair_mask = q_bool_mask.unsqueeze(2) & r_bool_mask.unsqueeze(1)
        scores = scores.masked_fill(~pair_mask, MASK_FILL)
        att = F.softmax(scores, dim=-1) * q_bool_mask.unsqueeze(-1).to(scores.dtype)
        v_aligned = torch.bmm(att, F.relu(self.v_net(self.drop(v))))
        joint = F.relu(self.q_net(self.drop(q))) * v_aligned
        return self.norm(q + self.p_net(self.drop(joint))) * q_bool_mask.unsqueeze(-1).to(q.dtype)


class BANEncoder(DualPoolS1Encoder):
    """Bilinear Attention Networks (Kim et al., NeurIPS 2018) with $G=3$ stacked bilinear glimpses."""

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        d = self.d_model
        glimpses = int(_cfg(config, "BAN_GLIMPSES", 3))
        drop = float(getattr(config.SELF_ENCODER.SELF_ATTENTION, "DROPOUT", 0.1))
        self.self_encoder = Encoder(config.SELF_ENCODER)
        self.ban_layers = nn.ModuleList([BANLayer(d, drop) for _ in range(glimpses)])

    def forward(self, batch):
        v, v_mask, r_mask, t, t_mask, t_bool = self.embed_inputs(batch)
        t_init = t
        t = self.self_encoder(features=t, padding_mask=t_mask)
        if self.pretrained_residual:
            t = t + t_init
        for layer in self.ban_layers:
            t = layer(t, v, t_bool, r_mask)
        return self.pool_and_fuse(v, v_mask, r_mask, t, t_mask)


# ---------------------------------------------------------------------------
# 7. HieCoAtt (Lu et al., NeurIPS 2016) — Hierarchical Question-Image Co-Attention
# ---------------------------------------------------------------------------
class FastHierarchicalExtractor(nn.Module):
    """Vectorized 1D-Conv unigram/bigram/trigram hierarchical phrase extractor (Lu et al., NeurIPS 2016)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.proj = nn.Linear(3 * d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, bool_mask: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        x_m = x * bool_mask.unsqueeze(-1).to(x.dtype)
        xt = x_m.transpose(1, 2)
        u1 = F.relu(self.conv1(xt)[:, :, :L]).transpose(1, 2)
        u2 = F.relu(self.conv2(xt)[:, :, :L]).transpose(1, 2)
        u3 = F.relu(self.conv3(xt)[:, :, :L]).transpose(1, 2)
        out = self.norm(x + self.drop(self.proj(torch.cat([u1, u2, u3], dim=-1))))
        return out * bool_mask.unsqueeze(-1).to(x.dtype)


class SymmetricCoAttnBlock(nn.Module):
    """Symmetric Vision<->Language Co-Attention + Self-Attention + FFN block (used by HieCoAtt, ViLBERT, LXMERT)."""

    def __init__(self, attn_cfg):
        super().__init__()
        self.vl_attn = MultiHeadAttention(attn_cfg)
        self.lv_attn = MultiHeadAttention(attn_cfg)
        self.vv_attn = MultiHeadAttention(attn_cfg)
        self.ll_attn = MultiHeadAttention(attn_cfg)
        self.v_ffn = PositionWiseFeedForward(attn_cfg)
        self.l_ffn = PositionWiseFeedForward(attn_cfg)

    def forward(self, v, v_mask, l, l_mask):
        v_cross = self.vl_attn(queries=v, keys=l, values=l, attention_mask=l_mask)
        l_cross = self.lv_attn(queries=l, keys=v, values=v, attention_mask=v_mask)
        v_self = self.vv_attn(queries=v_cross, keys=v_cross, values=v_cross, attention_mask=v_mask)
        l_self = self.ll_attn(queries=l_cross, keys=l_cross, values=l_cross, attention_mask=l_mask)
        return self.v_ffn(v_self), self.l_ffn(l_self)


class HieCoAttEncoder(DualPoolS1Encoder):
    """Hierarchical Question-Image Co-Attention (Lu et al., NeurIPS 2016):
    Unigram/Bigram/Trigram hierarchical phrase convs + symmetric co-attention layers.
    """

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        d = self.d_model
        layers = int(getattr(config.SELF_ENCODER, "LAYERS", 3))
        attn_cfg = config.SELF_ENCODER.SELF_ATTENTION
        self.hier = FastHierarchicalExtractor(d, float(getattr(attn_cfg, "DROPOUT", 0.1)))
        self.pos_embedding = SinusoidPositionalEmbedding(d)
        self.v_norm = nn.LayerNorm(d)
        self.l_norm = nn.LayerNorm(d)
        self.co_layers = nn.ModuleList([SymmetricCoAttnBlock(attn_cfg) for _ in range(layers)])

    def forward(self, batch):
        v, v_mask, r_mask, t, t_mask, t_bool = self.embed_inputs(batch)
        v_init, t_init = v, t
        t = self.hier(t, t_bool)
        v = self.v_norm(v) + self.pos_embedding(v)
        t = self.l_norm(t) + self.pos_embedding(t)
        for layer in self.co_layers:
            v, t = layer(v, v_mask, t, t_mask)
        if self.pretrained_residual:
            v = v + v_init
            t = t + t_init
        return self.pool_and_fuse(v, v_mask, r_mask, t, t_mask)


# ---------------------------------------------------------------------------
# 8. VisualBERT (Li et al., 2019) — Single-Stream Early-Fusion Transformer
# ---------------------------------------------------------------------------
class VisualBERTEncoder(DualPoolS1Encoder):
    """Single-Stream Early-Fusion Transformer (VisualBERT / VanillaTransformer):
    Concatenates `[text_tokens; vision_tokens]` with learned modality embeddings and runs joint self-attention.
    """

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        d = self.d_model
        self.modality_emb = nn.Embedding(2, d)
        self.unified_encoder = Encoder(config.SELF_ENCODER)

    def forward(self, batch):
        v, v_mask, r_mask, t, t_mask, _ = self.embed_inputs(batch)
        v_init, t_init = v, t
        L = t.shape[1]
        mod_t = self.modality_emb(torch.zeros((1, L), dtype=torch.long, device=t.device))
        mod_v = self.modality_emb(torch.ones((1, v.shape[1]), dtype=torch.long, device=v.device))
        joint = torch.cat([t + mod_t, v + mod_v], dim=1)
        joint_mask = torch.cat([t_mask, v_mask], dim=-1)
        joint_out = self.unified_encoder(features=joint, padding_mask=joint_mask)
        t_out, v_out = joint_out[:, :L, :], joint_out[:, L:, :]
        if self.pretrained_residual:
            t_out = t_out + t_init
            v_out = v_out + v_init
        return self.pool_and_fuse(v_out, v_mask, r_mask, t_out, t_mask)


# ---------------------------------------------------------------------------
# 9. ViLBERT (Lu et al., NeurIPS 2019) — Two-Stream Self-Attention + Co-Attention
# ---------------------------------------------------------------------------
class ViLBERTEncoder(DualPoolS1Encoder):
    """Two-Stream Transformer (ViLBERT): modality-specific self-attention streams followed by co-attention."""

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        attn_cfg = config.SELF_ENCODER.SELF_ATTENTION
        self.text_encoder = Encoder(config.SELF_ENCODER)
        self.vision_encoder = Encoder(config.SELF_ENCODER)
        co_layers = max(1, int(getattr(config.GUIDED_ENCODER, "LAYERS", 3)) // 2)
        self.co_layers = nn.ModuleList([SymmetricCoAttnBlock(attn_cfg) for _ in range(co_layers)])

    def forward(self, batch):
        v, v_mask, r_mask, t, t_mask, _ = self.embed_inputs(batch)
        v_init, t_init = v, t
        t = self.text_encoder(features=t, padding_mask=t_mask)
        v = self.vision_encoder(features=v, padding_mask=v_mask)
        for layer in self.co_layers:
            v, t = layer(v, v_mask, t, t_mask)
        if self.pretrained_residual:
            v = v + v_init
            t = t + t_init
        return self.pool_and_fuse(v, v_mask, r_mask, t, t_mask)


# ---------------------------------------------------------------------------
# 10. LXMERT / CrossModalityTransformer (Tan & Bansal, EMNLP 2019)
# ---------------------------------------------------------------------------
class LXMERTEncoder(DualPoolS1Encoder):
    """Cross-Modality Transformer (LXMERT-style): stacked bidirectional cross-attention + self-attention layers."""

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab, keep_classify=keep_classify)
        d = self.d_model
        layers = int(getattr(config.SELF_ENCODER, "LAYERS", 3))
        attn_cfg = config.SELF_ENCODER.SELF_ATTENTION
        self.pos_embedding = SinusoidPositionalEmbedding(d)
        self.v_norm = nn.LayerNorm(d)
        self.l_norm = nn.LayerNorm(d)
        self.x_layers = nn.ModuleList([SymmetricCoAttnBlock(attn_cfg) for _ in range(layers)])

    def forward(self, batch):
        v, v_mask, r_mask, t, t_mask, _ = self.embed_inputs(batch)
        v_init, t_init = v, t
        v = self.v_norm(v) + self.pos_embedding(v)
        t = self.l_norm(t) + self.pos_embedding(t)
        for layer in self.x_layers:
            v, t = layer(v, v_mask, t, t_mask)
        if self.pretrained_residual:
            v = v + v_init
            t = t + t_init
        return self.pool_and_fuse(v, v_mask, r_mask, t, t_mask)


ENCODER_REGISTRY = {
    "MCAN": MCANEncoder,
    "PhrasalMCAN": PhrasalMCANEncoder,
    "BUTD_Bami": ButdBamiEncoder,
    "SAAA": SAAAEncoder,
    "SAN": SANEncoder,
    "BAN": BANEncoder,
    "HieCoAtt": HieCoAttEncoder,
    "VisualBERT": VisualBERTEncoder,
    "ViLBERT": ViLBERTEncoder,
    "LXMERT": LXMERTEncoder,
}


def build_s1_encoder(config, vocab, keep_classify: bool = False):
    enc_type = str(_cfg(config, "ENCODER_TYPE", "ViSO"))
    if enc_type == "ViSO":
        from .encoder import ViSOEncoder
        return ViSOEncoder(config, vocab, keep_classify=keep_classify)
    if enc_type == "BUTD":
        from .butd import ButdEncoder
        enc = ButdEncoder(config, vocab)
        if keep_classify:
            enc.classify = nn.Linear(config.D_MODEL, vocab.total_answers)
        return enc
    if enc_type not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown MODEL.ENCODER_TYPE={enc_type!r}; available: {sorted(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[enc_type](config, vocab, keep_classify=keep_classify)
