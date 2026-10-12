"""ViSO-FT: ViSO with the two frozen-cache encoders replaced by LIVE encoders whose top blocks are fine-tuned.
  vision: google/siglip2-base-patch16-256 @3f9f96cb (the vision tower, the SAME 65 x 768 tokens as the cache pipeline: row 0 = the pooled MAP token, rows 1..64 = the 2x2-average-pooled 8x8 patch grid,
          boxes = scripts/siglip2_extract.grid_boxes) computed per batch from uint8 images, WITH gradients in the unfrozen part;
  text:   Qualcomm-AI-Research/BamiBERT @57bc1340 (token states of the question; mean-pooled token states of every candidate string = the cache's 'mean' pool), live.
Unfrozen: the top `TOP_BLOCKS` (4) transformer blocks of each tower, SigLIP's MAP head and its post-LayerNorm (the final LayerNorm of the vision stack); BamiBERT's blocks carry their own output LayerNorms,
so "BamiBERT's final LayerNorm" is the one of the last unfrozen block. Everything below stays frozen AND in eval mode (no dropout), and no activation of it is stored (autograd starts at the first
trainable block). Learning rates: TRAINING.LR_FT (1e-5) for the unfrozen encoder parameters; the ViSO layers keep LR_ENCODER / LR_HEADS (the task builds the groups from `finetune_parameters()`).
Normalisation: the cache pipeline z-scores the states with train-set statistics; live states drift, so the statistics become the INITIAL value of a learnable per-dimension affine (NORM: stats; at step 0
the live path reproduces the cached features) or a LayerNorm is used instead (NORM: layernorm). `fit_stats` computes the statistics once from the frozen towers (train texts only).
Optional (not run by default): LORA_R > 0 injects LoRA adapters into both towers instead of unfreezing blocks (peft).
Checkpoints hold only the trainable parameters and buffers (the frozen tower weights are re-loaded from the pinned Hugging Face revision), so best_model.pth / last.pth stay small."""
import torch
import torch.nn.functional as F
from torch import nn

from builders.model_builder import META_ARCHITECTURE
from utils.image_u8_cache import to_pixel_values

from .viso_model import ViSO

N_GRID = 8
MIN_SD = 0.05                                           # same floor as utils/viso_text_cache.py


def grid_boxes():
    b = [[0.0, 0.0, 1.0, 1.0]]
    for r in range(N_GRID):
        for c in range(N_GRID):
            b.append([c / N_GRID, r / N_GRID, (c + 1) / N_GRID, (r + 1) / N_GRID])
    return torch.tensor(b, dtype=torch.float32)


def pool_candidates(h, pm):
    """the cache's 'mean' pool of a candidate string (scripts/bamibert_extract.py): the mean of the token states WITHOUT <s> and </s> (all tokens when the string has at most 2)"""
    n = pm.sum(1, keepdim=True)
    pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    body = pm & (pos > 0) & (pos < n - 1)
    body = torch.where(n > 2, body, pm).unsqueeze(-1).to(h.dtype)
    return (h * body).sum(1) / body.sum(1).clamp(min=1)


def _cfg(c, k, d):
    return getattr(c, k, d)


class Standardize(nn.Module):
    """per-dimension (x - mu) / sd with trainable scale / shift initialised from the statistics (mode 'stats'), or a LayerNorm (mode 'layernorm')"""

    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        if mode == "layernorm":
            self.ln = nn.LayerNorm(dim)
        else:
            self.scale = nn.Parameter(torch.ones(dim))
            self.shift = nn.Parameter(torch.zeros(dim))

    def set_stats(self, mu, sd):
        if self.mode == "stats":
            with torch.no_grad():
                self.scale.copy_(1.0 / sd)
                self.shift.copy_(-mu / sd)

    def forward(self, x):
        return self.ln(x) if self.mode == "layernorm" else x * self.scale + self.shift


class LiveEncoders(nn.Module):
    def __init__(self, ft):
        super().__init__()
        from transformers import AutoModel
        self.top = int(_cfg(ft, "TOP_BLOCKS", 4))
        self.amp = _cfg(ft, "AMP", "bf16")
        self.lora_r = int(_cfg(ft, "LORA_R", 0))
        sig = AutoModel.from_pretrained(ft.VISION_MODEL, revision=ft.VISION_REVISION, dtype=torch.float32)
        self.vision = sig.vision_model
        del sig
        self.text = AutoModel.from_pretrained(ft.TEXT_MODEL, revision=ft.TEXT_REVISION, dtype=torch.float32, add_pooling_layer=False)
        for p in list(self.vision.parameters()) + list(self.text.parameters()):
            p.requires_grad_(False)
        if self.lora_r > 0:
            self._inject_lora()
        else:
            for blk in self.vision.encoder.layers[-self.top:]:
                blk.requires_grad_(True)
            for m in (self.vision.post_layernorm, self.vision.head):
                m.requires_grad_(True)
            for blk in self.text.encoder.layer[-self.top:]:
                blk.requires_grad_(True)
        self._trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        mode = _cfg(ft, "NORM", "stats")
        self.q_norm, self.c_norm = Standardize(768, mode), Standardize(768, mode)
        self.register_buffer("boxes", grid_boxes(), persistent=False)

    def _inject_lora(self):
        from peft import LoraConfig, inject_adapter_in_model
        cfg_v = LoraConfig(r=self.lora_r, lora_alpha=self.lora_r, lora_dropout=0.0, target_modules=r".*encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))")
        cfg_t = LoraConfig(r=self.lora_r, lora_alpha=self.lora_r, lora_dropout=0.0, target_modules=r".*encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)")
        self.vision = inject_adapter_in_model(cfg_v, self.vision)
        self.text = inject_adapter_in_model(cfg_t, self.text)
        for n, p in self.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)

    # ---- modes: the frozen lower part stays in eval mode (no dropout) even when the model trains
    def train(self, mode=True):
        super().train(mode)
        if mode and self.lora_r == 0:
            self.vision.embeddings.eval()
            for blk in self.vision.encoder.layers[:-self.top]:
                blk.eval()
            self.text.embeddings.eval()
            for blk in self.text.encoder.layer[:-self.top]:
                blk.eval()
        return self

    def finetune_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and n in self._trainable_names]

    def counts(self):
        tower = sum(p.numel() for n, p in self.named_parameters() if n.startswith(("vision.", "text.")))
        train = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and n.startswith(("vision.", "text.")))
        return {"towers_total": tower, "towers_trainable": train}

    def _ctx(self, device):
        ok = device.type == "cuda" and self.amp in ("bf16", "fp16")
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if self.amp == "bf16" else torch.float16, enabled=ok)

    # ---- vision
    def image_tokens(self, image_u8):
        px = to_pixel_values(image_u8)
        with self._ctx(px.device):
            out = self.vision(pixel_values=px)
        h = out.last_hidden_state.float()                                               # (B, 256, 768)
        b = h.shape[0]
        g = F.avg_pool2d(h.view(b, 16, 16, 768).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, N_GRID * N_GRID, 768)
        return torch.cat([out.pooler_output.float().unsqueeze(1), g], 1)                # (B, 65, 768): pooled token first

    # ---- text
    def token_states(self, ids, mask):
        with self._ctx(ids.device):
            return self.text(input_ids=ids, attention_mask=mask.long()).last_hidden_state.float()

    def candidate_vectors(self, cand_ids, cand_pad_mask, cand_mask, device):
        """(B,16,Lc) ids -> (B,16,768) mean-pooled token states of the valid candidates (zeros elsewhere), NOT standardised"""
        b, k, lc = cand_ids.shape
        out = torch.zeros(b, k, 768, device=device)
        rows = cand_mask.nonzero(as_tuple=False)
        if rows.numel() == 0:
            return out
        ids = cand_ids[rows[:, 0], rows[:, 1]]
        pm = cand_pad_mask[rows[:, 0], rows[:, 1]]
        h = self.token_states(ids, pm)
        out[rows[:, 0], rows[:, 1]] = pool_candidates(h, pm.bool())
        return out

    @torch.no_grad()
    def fit_stats(self, q_batches, c_batches, device):
        """statistics of the FROZEN initial towers on train texts: q_batches / c_batches yield (ids, mask) tensors. Returns and installs (q_mu, q_sd, c_mu, c_sd)."""
        was = self.training
        self.eval()
        qs = [self.token_states(i.to(device), m.to(device))[m.to(device).bool()] for i, m in q_batches]
        q = torch.cat(qs)
        cs = []
        for i, m in c_batches:
            i, m = i.to(device), m.to(device)
            cs.append(pool_candidates(self.token_states(i, m), m.bool()))
        c = torch.cat(cs)
        q_mu, q_sd = q.mean(0), q.std(0).clamp(min=MIN_SD)
        c_mu, c_sd = c.mean(0), c.std(0).clamp(min=MIN_SD)
        self.q_norm.set_stats(q_mu, q_sd)
        self.c_norm.set_stats(c_mu, c_sd)
        self.train(was)
        return q_mu.cpu(), q_sd.cpu(), c_mu.cpu(), c_sd.cpu()

    def forward(self, batch):
        dev = batch.image_u8.device
        tok = self.image_tokens(batch.image_u8)
        b = tok.shape[0]
        batch.region_features = tok
        batch.region_mask = torch.ones(b, tok.shape[1], dtype=torch.bool, device=dev)
        batch.region_boxes = self.boxes.to(dev).unsqueeze(0).expand(b, -1, -1)
        batch.question_states = self.q_norm(self.token_states(batch.q_ids, batch.q_mask))
        batch.question_state_mask = batch.q_mask.bool()
        cv = self.candidate_vectors(batch.cand_ids, batch.cand_pad_mask, batch.cand_mask, dev)
        batch.cand_vecs = self.c_norm(cv) * batch.cand_mask.unsqueeze(-1)
        return batch


@META_ARCHITECTURE.register()
class ViSOFT(ViSO):
    """ViSO (cached-text variant: CachedBertEmbedding + pooled candidate vectors) fed by LiveEncoders; needs batch.image_u8, q_ids / q_mask, cand_ids / cand_pad_mask (s1_live_dataset)."""

    def __init__(self, config, vocab):
        super().__init__(config, vocab)
        self.live = LiveEncoders(config.FT)
        self._frozen_keys = {"live." + n for n, p in self.live.named_parameters() if not p.requires_grad}

    def forward(self, batch):
        return super().forward(self.live(batch))

    def finetune_parameters(self):
        return self.live.finetune_parameters()

    # ---- checkpoints without the frozen tower weights (re-loaded from the pinned revision at construction)
    def state_dict(self, *args, **kwargs):                    # ViSOFT is always the root module (the task saves model.state_dict())
        sd = super().state_dict(*args, **kwargs)
        return type(sd)((k, v) for k, v in sd.items() if k not in self._frozen_keys)

    def load_state_dict(self, state_dict, strict=True):
        res = super().load_state_dict(state_dict, strict=False)
        missing = [k for k in res.missing_keys if k not in self._frozen_keys]
        if strict and (missing or res.unexpected_keys):
            raise RuntimeError(f"ViSOFT checkpoint mismatch: missing {missing[:5]}, unexpected {res.unexpected_keys[:5]}")
        return res

    def parameter_report(self):
        c = self.live.counts()
        total = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"params_total_incl_frozen_towers": total, "params_trained": trained, "towers_total": c["towers_total"], "towers_trainable": c["towers_trainable"],
                "viso_layers_trainable": trained - c["towers_trainable"]}
