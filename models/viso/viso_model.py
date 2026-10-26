import torch
from torch import nn

from builders.model_builder import META_ARCHITECTURE
from utils.viso_s1 import MAX_SPACE, NEG_INF

from .encoder import ViSOEncoder
from .heads import ChoiceHead, NoulHead, ScoreHead


def _cfg(config, name, default):
    return getattr(config, name, default)


@META_ARCHITECTURE.register()
class ViSO(nn.Module):
    """ViSO: ViCompreVQA encoder (ViSOEncoder) + typed heads. The question type is part of the interface
    (batch.type_id: 0 noul, 1 score, 2 choice); there is NO type predictor. Every head writes into the shared
    16-wide logit layout (masked columns = NEG_INF in fp32)."""

    def __init__(self, config, vocab):
        super().__init__()
        self.d_model = config.D_MODEL
        self.encoder = self._make_encoder(config, vocab)
        drop = _cfg(config, "HEAD_DROPOUT", 0.1)
        self.noul_head = NoulHead(config.D_MODEL, drop)
        self.score_head = ScoreHead(
            config.D_MODEL, drop,
            count_pool=_cfg(config, "SCORE_COUNT_POOL", False),
            count_bins=_cfg(config, "SCORE_COUNT_BINS", 64),
        )
        self.cached_text = self.encoder.cached_text
        if self.cached_text:                       # candidates come as cached pooled BamiBERT vectors (batch.cand_vecs), no embedding table
            emb, d_in = None, config.TEXT_EMBEDDING.D_EMBEDDING
        else:
            emb, d_in = self.encoder.text_embedding.embedding, 768
            if not _cfg(config, "SHARE_EMBEDDING", True):
                emb = nn.Embedding(len(vocab), emb.embedding_dim, padding_idx=vocab.padding_idx)
        self.choice_head = ChoiceHead(
            emb, vocab.padding_idx, config.D_MODEL, set_attention=_cfg(config, "SET_ATTENTION", True),
            set_layers=_cfg(config, "SET_LAYERS", 2), set_heads=_cfg(config, "SET_HEADS", 8),
            set_ff=_cfg(config, "SET_FF", 1024), dropout=_cfg(config, "SET_DROPOUT", 0.1), d_in=d_in,
            cross_attention=_cfg(config, "CHOICE_CROSS_ATTENTION", False),
            cross_heads=_cfg(config, "CHOICE_CROSS_HEADS", 8),
            cross_dropout=_cfg(config, "CHOICE_CROSS_DROPOUT", 0.1))

    def _make_encoder(self, config, vocab):                      # hook: the B-BUTD baseline (models/viso/butd.py) or MODEL.ENCODER_TYPE swaps the encoder and keeps the typed heads
        enc_type = _cfg(config, "ENCODER_TYPE", "ViSO")
        if enc_type != "ViSO":
            from .vqa_baselines import build_s1_encoder
            return build_s1_encoder(config, vocab, keep_classify=False)
        return ViSOEncoder(config, vocab)

    def forward(self, batch):
        enc_out = self.encoder(batch)
        h, break_probs = enc_out[0], enc_out[1]
        vision_features = getattr(enc_out, "vision_features", None)
        vision_mask = getattr(enc_out, "vision_mask", None)
        bs = h.shape[0]
        logits = torch.full((bs, MAX_SPACE), NEG_INF, dtype=h.dtype, device=h.device)
        rows = (batch.type_id == 0).nonzero(as_tuple=True)[0]
        if rows.numel():
            logits[rows, :self.noul_head.width] = self.noul_head(h[rows]).to(logits.dtype)
        rows = (batch.type_id == 1).nonzero(as_tuple=True)[0]
        if rows.numel():
            vf = vision_features[rows] if vision_features is not None else None
            vm = vision_mask[rows] if vision_mask is not None else None
            logits[rows, :self.score_head.width] = self.score_head(h[rows], vf, vm).to(logits.dtype)
        rows = (batch.type_id == 2).nonzero(as_tuple=True)[0]
        if rows.numel():
            cv = batch.cand_vecs[rows] if self.cached_text else None
            vf = vision_features[rows] if vision_features is not None else None
            vm = vision_mask[rows] if vision_mask is not None else None
            logits[rows] = self.choice_head(h[rows], batch.cand_tokens[rows], batch.cand_mask[rows], cv, vf, vm).to(logits.dtype)
        logits = logits.masked_fill(~batch.space_mask, NEG_INF)
        return {"logits": logits, "h": h, "break_probs": break_probs}


@META_ARCHITECTURE.register()
class FlatS1(nn.Module):
    """Baseline A0: flat classifier over the answer vocabulary on any System-One encoder body (`MODEL.ENCODER_TYPE`).
    `logits` are the flat logits RESTRICTED to the item's label space (noul: [<no>,<yes>], score:
    #0..#10, choice: the candidate texts; a candidate absent from the flat vocab gets min(row logits) - 5), so
    evaluation of FlatS1 is always over the same space as ViSO."""

    def __init__(self, config, vocab):
        super().__init__()
        self.d_model = config.D_MODEL
        self.encoder = self._make_encoder(config, vocab)

    def _make_encoder(self, config, vocab):
        enc_type = _cfg(config, "ENCODER_TYPE", "ViSO")
        if enc_type != "ViSO":
            from .vqa_baselines import build_s1_encoder
            return build_s1_encoder(config, vocab, keep_classify=True)
        return ViSOEncoder(config, vocab, keep_classify=True)

    def restrict(self, flat_logits, batch):
        idx = batch.flat_index
        g = flat_logits.gather(1, idx.clamp(min=0))
        absent = (idx < 0) & batch.space_mask
        g = torch.where(absent, flat_logits.min(dim=1, keepdim=True).values - 5.0, g)
        return g.masked_fill(~batch.space_mask, NEG_INF)

    def forward(self, batch):
        h, break_probs = self.encoder(batch)
        flat_logits = self.encoder.classify(h)
        return {"logits": self.restrict(flat_logits, batch), "flat_logits": flat_logits, "h": h,
                "break_probs": break_probs}
