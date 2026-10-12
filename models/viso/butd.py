import torch
from torch import nn
from torch.nn import functional as F

from builders.model_builder import META_ARCHITECTURE

from .viso_model import ViSO, _cfg


class WordEmbedding(nn.Module):
    """holder of the question word-embedding table (the ChoiceHead of ViSO shares it with the candidate texts, SHARE_EMBEDDING)"""

    def __init__(self, vocab, d_embedding):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), d_embedding, padding_idx=vocab.padding_idx)


class ButdEncoder(nn.Module):
    """Bottom-Up Top-Down style fusion (Anderson et al. 2018; Teney et al. 2018 'tips and tricks') on the cached SigLIP 2 tokens: GRU question encoder (from scratch, the benchmark vocabulary),
    question-guided soft attention over the 65 visual tokens (pooled token + 8 x 8 grid; G glimpses), Hadamard product fusion of the question and the attended visual vector, a 2-layer MLP.
    Returns (h, None) like ViSOEncoder so the typed heads of ViSO (noul / score / choice set head) are reused unchanged: B-BUTD is scored in System One mode exactly like ViSO."""

    def __init__(self, config, vocab):
        super().__init__()
        d = config.D_MODEL
        d_emb = config.TEXT_EMBEDDING.D_EMBEDDING
        self.pad = vocab.padding_idx
        self.cached_text = False
        self.zero_vision = bool(_cfg(config, "ZERO_VISION", False))
        self.use_boxes = bool(_cfg(config.VISION_EMBEDDING, "USE_BOXES", False))
        self.glimpses = int(_cfg(config, "BUTD_GLIMPSES", 2))
        self.text_embedding = WordEmbedding(vocab, d_emb)
        self.q_dropout = nn.Dropout(config.TEXT_EMBEDDING.DROPOUT)
        self.gru = nn.GRU(d_emb, d, batch_first=True)
        self.vision_proj = nn.Linear(config.VISION_EMBEDDING.D_FEATURE, d)               # named like ViSO's so TRAINING.VISION_PATH_AS_HEADS applies to it
        if self.use_boxes:
            self.box_proj = nn.Linear(4, d)
        self.v_dropout = nn.Dropout(config.VISION_EMBEDDING.DROPOUT)
        self.att_q = nn.Linear(d, d)
        self.att_v = nn.Linear(d, d)
        self.att_out = nn.Linear(d, self.glimpses)
        self.fuse_q = nn.Linear(d, d)
        self.fuse_v = nn.Linear(d * self.glimpses, d)
        drop = float(_cfg(config, "FUSION_DROPOUT", 0.0))
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(max(drop, 0.1)), nn.Linear(2 * d, d))
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, batch):
        tokens = batch.question_tokens
        mask = tokens != self.pad                                                          # (B,L) True = real token (the <bos> / <eos> tokens are real)
        out, _ = self.gru(self.q_dropout(self.text_embedding.embedding(tokens)))
        last = (mask.sum(1) - 1).clamp(min=0)
        q = out[torch.arange(out.shape[0], device=out.device), last]                       # GRU state at the last real token
        regions, region_mask = batch.region_features, batch.region_mask
        if self.zero_vision:
            regions = torch.zeros_like(regions)
        v = self.vision_proj(regions)
        if self.use_boxes:
            boxes = batch.region_boxes
            v = v + self.box_proj(torch.zeros_like(boxes) if self.zero_vision else boxes)
        v = self.v_dropout(F.relu(v))                                                      # (B,R,d)
        joint = F.relu(self.att_q(q))[:, None, :] * F.relu(self.att_v(v))                  # question-guided attention logits
        logits = self.att_out(joint).masked_fill(~region_mask[..., None], -1e4)            # (B,R,G)
        att = torch.softmax(logits, dim=1)
        v_att = torch.einsum("brg,brd->bgd", att, v).flatten(1)                            # (B,G*d)
        h = F.relu(self.fuse_q(q)) * F.relu(self.fuse_v(v_att))                            # Hadamard fusion
        return self.layer_norm(self.mlp(h)), None


@META_ARCHITECTURE.register()
class BUTD(ViSO):
    """B-BUTD baseline: ButdEncoder + ViSO's typed heads (same heads, same losses, same evaluation)."""

    def _make_encoder(self, config, vocab):
        return ButdEncoder(config, vocab)
