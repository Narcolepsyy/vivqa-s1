import math

import torch
from torch.nn import functional as F

from models.constituent_mcan.mcan_constituent import ConstituentMCAN

MASK_FILL = -10e4   # additive mask value used by the repo's attention layers


class EncoderOutput(tuple):
    """2-tuple (h, break_probs) subclass that exposes unpooled visual tokens and mask without breaking
    code that unpacks `h, break_probs = encoder(batch)` or indexes `encoder(batch)[0]`."""
    def __new__(cls, h, break_probs, vision_features=None, vision_mask=None):
        instance = super().__new__(cls, (h, break_probs))
        instance.h = h
        instance.break_probs = break_probs
        instance.vision_features = vision_features
        instance.vision_mask = vision_mask
        return instance


class TextVisionGrounding(torch.nn.Module):
    """One question-to-vision cross-attention residual, inserted before text pooling.

    The guided encoder lets every visual token read the question. The question tokens themselves never read the
    image, so the text vector added in the sum fusion is image-invariant. This block is the other direction:
    question tokens query the question-conditioned visual tokens. No extra positional encoding is added; positions
    already live in the two streams.

    The scalar gate is initialized at 0. The residual is then exactly zero, and the fused vector matches a baseline
    encoder with the same other weights, in eval and in train (the block has no dropout, so it does not consume
    random state). Padded question positions keep their incoming features. Padded visual tokens are removed from
    the softmax. A row with no real visual token gets a zero residual, because a softmax over an empty key set is
    undefined. Gold labels are not an input.
    """

    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if heads < 1 or d_model % heads != 0:
            raise ValueError(f"self-attention heads must divide D_MODEL, got heads={heads} D_MODEL={d_model}")
        self.heads = int(heads)
        self.d_head = d_model // self.heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
        self.gate = torch.nn.Parameter(torch.zeros(()))

    def forward(self, text, vision, text_pad, vision_pad):
        # text (B, L, D), vision (B, R, D); text_pad / vision_pad bool, True = ignore
        if text.size(1) == 0 or vision.size(1) == 0:
            return text
        batch, length, dim = text.shape
        n_reg = vision.size(1)
        heads, d_head = self.heads, self.d_head
        query = self.q_proj(text).view(batch, length, heads, d_head).transpose(1, 2)
        key = self.k_proj(vision).view(batch, n_reg, heads, d_head).transpose(1, 2)
        value = self.v_proj(vision).view(batch, n_reg, heads, d_head).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_head)
        scores = scores.masked_fill(vision_pad[:, None, None, :], MASK_FILL)
        empty = vision_pad.all(dim=-1)
        if bool(empty.any()):
            scores = scores.masked_fill(empty[:, None, None, None], 0)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(vision_pad[:, None, None, :], 0)
        if bool(empty.any()):
            attn = attn.masked_fill(empty[:, None, None, None], 0)
        context = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch, length, dim)
        residual = self.norm(self.out_proj(context))
        residual = residual.masked_fill(text_pad.unsqueeze(-1), 0)
        if bool(empty.any()):
            residual = residual.masked_fill(empty[:, None, None], 0)
        return text + torch.tanh(self.gate) * residual


def require_text_vision_grounding(config, encoder):
    """MODEL.TEXT_VISION_GROUNDING is a ViSOEncoder option. Other encoders must not ignore it silently."""
    if bool(getattr(config, "TEXT_VISION_GROUNDING", False)) and not isinstance(encoder, ViSOEncoder):
        raise ValueError(
            "MODEL.TEXT_VISION_GROUNDING is implemented on ViSOEncoder only "
            f"(this config built {type(encoder).__name__}). Set it false for other encoders."
        )


class ViSOEncoder(ConstituentMCAN):
    """ViCompreVQA's ConstituentMCAN encoder returning the 512-d fused vector instead of class scores.

    The forward below copies ConstituentMCAN.forward step by step (vision embedding, LSTM text embedding,
    constituent module, self / guided attention encoders, attention pooling, layer_norm(vision_proj + text_proj))
    with two deliberate differences (documented in docs/viso_README.md):
      * the vision padding mask is built from the explicit `region_mask` of the batch instead of "rows that sum to
        zero", and the vision attention pooling is masked with it, so zero-padded regions cannot change the output
        (the original pools over padded rows too);
      * `MODEL.ZERO_VISION: true` feeds all-zero region features but keeps the explicit mask (question-only ablation).
    Opt-in (`MODEL.TEXT_VISION_GROUNDING`, default false): after the guided encoder, question tokens cross-attend to
    the visual tokens and a zero-init gate adds that residual before text pooling. Off, the module is not created and
    the state dict matches the baseline encoder. See docs/viso_text_vision_grounding.md.
    Returns (h, break_probs); `classify` is deleted unless keep_classify=True (baseline A0 / FlatS1)."""

    def __init__(self, config, vocab, keep_classify: bool = False):
        super().__init__(config, vocab)
        if not keep_classify:
            del self.classify
        self.zero_vision = bool(getattr(config, "ZERO_VISION", False))
        self.cached_text = config.TEXT_EMBEDDING.ARCHITECTURE == "CachedBertEmbedding"
        self.strict_text_mask = bool(getattr(config, "STRICT_TEXT_MASK", False))
        # FUSION_PRODUCT: add a Hadamard term proj_q(pooled text) * proj_v(pooled vision) to the sum fusion (BUTD / question x image product recipe). Off by default.
        # ATTR_REDUCE_INIT_STD > 0: re-initialise the last layer of the two attention-pooling MLPs with N(0, std) so the pooling is NOT uniform at the start (F3 probe); 0 = default init
        std = float(getattr(config, "ATTR_REDUCE_INIT_STD", 0.0))
        if std > 0:
            for m in (self.vision_attr_reduce, self.text_attr_reduce):
                torch.nn.init.normal_(m.fc2.weight, std=std)
        self.fusion_dropout = torch.nn.Dropout(float(getattr(config, "FUSION_DROPOUT", 0.0)))          # FUSION_DROPOUT: dropout on the fused vector h (regularisation variant); 0 = off
        self.fusion_product = bool(getattr(config, "FUSION_PRODUCT", False))
        if self.fusion_product:
            self.prod_q = torch.nn.Linear(config.D_MODEL, config.D_MODEL)
            self.prod_v = torch.nn.Linear(config.D_MODEL, config.D_MODEL)
        # SigLIP 2 configs: VISION_EMBEDDING.USE_BOXES adds a linear embedding of the patch-cell box (x1, y1, x2, y2) to the projected feature (the pooled token has the full-image box)
        # FEATURE_NORM: LayerNorm over the raw feature dim before the projection (SigLIP 2 features have outlier dims up to ~30; VinVL needed none). Off by default.
        self.feature_norm = torch.nn.LayerNorm(config.VISION_EMBEDDING.D_FEATURE) if bool(getattr(config.VISION_EMBEDDING, "FEATURE_NORM", False)) else None
        self.use_boxes = bool(getattr(config.VISION_EMBEDDING, "USE_BOXES", False))
        if self.use_boxes:
            self.box_proj = torch.nn.Linear(4, config.VISION_EMBEDDING.D_MODEL)
        self.pretrained_residual = bool(getattr(config, "PRETRAINED_RESIDUAL", False))
        # TEXT_VISION_GROUNDING: question tokens query visual tokens before text pooling. The block is initialized
        # under fork_rng so it does not change the initialization of any other parameter (the heads are built after
        # the encoder). Off: the module is not created and a baseline checkpoint loads unchanged.
        self.text_vision_grounding = bool(getattr(config, "TEXT_VISION_GROUNDING", False))
        if self.text_vision_grounding:
            with torch.random.fork_rng():
                self.tv_ground = TextVisionGrounding(config.D_MODEL, int(config.SELF_ENCODER.SELF_ATTENTION.HEAD))

    def forward(self, batch):
        regions, region_mask = batch.region_features, batch.region_mask          # (B,R,2048), (B,R) bool
        if self.zero_vision:
            regions = torch.zeros_like(regions)
        ve = self.vision_embedding
        if self.feature_norm is not None:
            regions = self.feature_norm(regions)
        projected = ve.proj(regions)
        if self.use_boxes:
            boxes = batch.region_boxes
            projected = projected + self.box_proj(torch.zeros_like(boxes) if self.zero_vision else boxes)
        vision_features = ve.dropout(ve.gelu(projected))                         # == FeatureEmbedding minus its mask
        vision_init = vision_features
        vision_padding_mask = (~region_mask).float().mul(MASK_FILL)[:, None, None, :]   # (B,1,1,R) additive

        if self.cached_text:                                                     # frozen BamiBERT token states instead of the from-scratch word embedding + LSTM
            text_features, (text_padding_mask, _) = self.text_embedding(batch.question_states, batch.question_state_mask)
        else:
            text_features, (text_padding_mask, _) = self.text_embedding(batch.question_tokens)
        text_init = text_features
        mask = text_padding_mask
        if mask is not None and mask.dim() in (3, 4):
            mask = mask.view(mask.size(0), mask.size(-1))
        if self.strict_text_mask:                                                 # bool, True = real token (see constituent_module.GroupAttention)
            mask = mask == 0
        text_features, break_probs = self.constituent_encoder(text_features, mask)

        text_features = self.self_encoder(features=text_features, padding_mask=text_padding_mask)
        if self.pretrained_residual:
            text_features = text_features + text_init
        vision_features = self.guided_encoder(
            vision_features=vision_features, vision_padding_mask=vision_padding_mask,
            language_features=text_features, language_padding_mask=text_padding_mask)
        if self.pretrained_residual:
            vision_features = vision_features + vision_init
        if self.text_vision_grounding:
            text_pad = text_padding_mask.reshape(text_features.shape[0], -1) != 0
            if text_pad.shape[1] != text_features.shape[1]:
                raise RuntimeError(
                    f"text padding mask length {text_pad.shape[1]} != text length {text_features.shape[1]}"
                )
            text_features = self.tv_ground(text_features, vision_features, text_pad, ~region_mask)

        v_logits = self.vision_attr_reduce(vision_features) + vision_padding_mask[:, 0, 0, :, None]   # mask the pooling
        attended_vision = F.softmax(v_logits, dim=1)
        t_logits = self.text_attr_reduce(text_features)
        if self.cached_text or self.strict_text_mask:   # cached states are padded to the batch maximum (not a fixed length): mask the pooling so predictions do not depend on the batch
            t_logits = t_logits + text_padding_mask[:, 0, 0, :, None]
        attended_text = F.softmax(t_logits, dim=1)                                # unchanged otherwise (fixed-length questions)
        weighted_vision = (vision_features * attended_vision).sum(dim=1)
        weighted_text = (text_features * attended_text).sum(dim=1)
        fused = self.vision_proj(weighted_vision) + self.text_proj(weighted_text)
        if self.fusion_product:
            fused = fused + self.prod_q(weighted_text) * self.prod_v(weighted_vision)
        h = self.fusion_dropout(self.layer_norm(fused))
        return EncoderOutput(h, break_probs, vision_features=vision_features, vision_mask=region_mask)
