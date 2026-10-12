"""Model adapters of the VLM System-One harness (scripts/s1_vlm_eval.py): everything that differs between models lives here, the protocol (prompt, one forward pass, letter logits, shards) does not.
  StdAdapter      any standard transformers image-text model (AutoModelForImageTextToText + chat template): Qwen3.5 / Intern-Decision / OneJev / jpt / Qwen3-VL ... trust_remote_code is NEVER used.
  VinternAdapter  5CD-AI/Vintern-1B-v3_5 (InternVLChatModel): the ONLY model loaded with trust_remote_code=True, and only at the pinned revision whose five remote .py files were reviewed; their sha256 are
                  checked against the table below before anything is imported and printed at load. Any other revision (or a changed file) is refused.
Interface: load(a) in __init__, .tok, .letter_ids, .W (16 x d letter rows of the output embeddings, float32), .n_params, .meta_extra, .prepare_image(PIL) (runs in the DataLoader workers),
.encode(rows, device) -> hidden state at the answer position (B, d) float32, .needs_linear_attention (for the fla backend check)."""
import hashlib
import os

import torch

LETTERS = "ABCDEFGHIJKLMNOP"

VINTERN = {
    "repo": "5CD-AI/Vintern-1B-v3_5",
    "revision": "b98f263eab246eb5269ade64edbdca8a887dc44d",
    "licence": "MIT",
    "remote_files_sha256": {                                                            # reviewed: no network / process / file-system access, only model code (config, conversation template, ViT, InternVL chat model)
        "configuration_intern_vit.py": "22dd687a5f142985407675fc6b1dc688c2ded93e54192543ecf28ef495a6efa8",
        "configuration_internvl_chat.py": "f88c46b4abd60767396ee251bac17a02adb19a2a5d69c72d5e9ad93173f09803",
        "conversation.py": "7e457a0fc08304deeb386ed1932e9db88dfbca888e0649ff76d488ae6c8aebfe",
        "modeling_intern_vit.py": "619c2b96987bf45018a32d8abd16e7f7036995c16e1bd762c3507d58b8fdb672",
        "modeling_internvl_chat.py": "4935c1f1d205167150d4449cbaf4bf340d8d5d0fd96ca97be5d15b26d489ead1",
    },
}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def letter_ids_of(tok):
    ids = []
    for L in LETTERS:
        e = tok.encode(L, add_special_tokens=False)
        assert len(e) == 1, f"option letter {L!r} is not a single token in this tokenizer: {e}"
        ids.append(e[0])
    return ids


class StdAdapter:
    needs_linear_attention = True
    meta_extra = {"chat_template_kwargs": {"enable_thinking": False}}

    def __init__(self, a, dtype):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(a.model, revision=a.revision, dtype=dtype, trust_remote_code=False).to(a.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(a.model, revision=a.revision)
        self.proc.tokenizer.padding_side = "left"
        if a.max_pixels:
            try:                                                                  # tile cap: documented in meta.json
                self.proc.image_processor.size = {"shortest_edge": a.min_pixels, "longest_edge": a.max_pixels}
            except Exception as e:                                                # pragma: no cover
                print("could not set max_pixels:", e)
        self.tok = self.proc.tokenizer
        self.letter_ids = letter_ids_of(self.tok)
        self.W = self.model.get_output_embeddings().weight[self.letter_ids].float()          # (16, d): the unembedding rows of the letter tokens
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.device = a.device

    def prepare_image(self, im):
        return im

    @torch.no_grad()
    def encode(self, rows, device):
        texts = []
        for _, _, im, prompt in rows:
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))
        inp = self.proc(text=texts, images=[r[2] for r in rows], return_tensors="pt", padding=True).to(device)
        return self.model.model(**inp).last_hidden_state[:, -1].float()


# ------------------------------------------------------------------------------------------------------------------- Vintern (InternVL), the one remote-code model
def require_pinned_vintern(repo, revision):
    if repo != VINTERN["repo"] or revision != VINTERN["revision"]:
        raise SystemExit(f"trust_remote_code is allowed ONLY for {VINTERN['repo']}@{VINTERN['revision']}; refusing {repo}@{revision}")


def verify_remote_code(repo, revision):
    """sha256 of the 5 remote .py files of the pinned snapshot vs the reviewed table; prints them; raises on any difference (before transformers imports a line of them)"""
    require_pinned_vintern(repo, revision)
    from huggingface_hub import hf_hub_download
    for name, want in VINTERN["remote_files_sha256"].items():
        p = hf_hub_download(repo, name, revision=revision)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"remote code {name}: sha256 {got}" + ("" if got == want else f"  !!! EXPECTED {want}"), flush=True)
        if got != want:
            raise SystemExit(f"remote code file {name} differs from the reviewed version: refusing to run it")


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_ratio_diff:
            best_ratio_diff, best_ratio = diff, ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=4, image_size=448, use_thumbnail=True):
    """InternVL's dynamic tiling (model card): the aspect ratio is matched to a grid of 448 px tiles with at most `max_num` tiles (+ a thumbnail of the whole image when more than one)"""
    w, h = image.size
    aspect = w / h
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    r = find_closest_aspect_ratio(aspect, ratios, w, h, image_size)
    tw, th = image_size * r[0], image_size * r[1]
    blocks = r[0] * r[1]
    resized = image.resize((tw, th))
    tiles = []
    for i in range(blocks):
        box = ((i % (tw // image_size)) * image_size, (i // (tw // image_size)) * image_size, ((i % (tw // image_size)) + 1) * image_size, ((i // (tw // image_size)) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class VinternAdapter:
    needs_linear_attention = False                                                       # Qwen2.5-0.5B language model: full attention, no fla kernels

    def __init__(self, a, dtype):
        from torchvision import transforms as T
        from transformers import AutoModel, AutoTokenizer
        verify_remote_code(a.model, a.revision)
        self.max_tiles = int(getattr(a, "max_tiles", 4))
        self.model = self._load_remote(a, dtype).to(a.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, trust_remote_code=True, use_fast=False)
        self.tok.padding_side = "right"                                                   # RIGHT padding: with the eager attention this model uses, left-padded rows have fully masked query rows -> NaN in fp16
        self.letter_ids = letter_ids_of(self.tok)
        self.lm = self.model.language_model
        self.W = self.lm.get_output_embeddings().weight[self.letter_ids].float()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.img_context_id = self.tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = self.img_context_id
        self.image_size = int(self.model.config.force_image_size or 448)
        self.transform = T.Compose([T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.device, self.dtype = a.device, dtype
        self.meta_extra = {"trust_remote_code": True, "remote_code_sha256": VINTERN["remote_files_sha256"], "max_tiles": self.max_tiles, "use_thumbnail": True, "image_tokens_per_tile": self.model.num_image_token,
                           "template": self.model.template, "dynamic_tiling": "InternVL dynamic_preprocess (448 px tiles, <= max_tiles + thumbnail)",
                           "chat_template_kwargs": None}

    @staticmethod
    def _load_remote(a, dtype):
        """Load the (hash-verified) remote InternVLChatModel class and instantiate it. The reviewed code predates transformers 5 and never calls `post_init()`, so the attribute v5's loader reads
        (`all_tied_weights_keys`) does not exist: a class-level empty dict is its correct value here (this checkpoint ties no weights: llm_config.tie_word_embeddings is False). Nothing in the remote files changes."""
        from transformers import AutoModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module("modeling_internvl_chat.InternVLChatModel", a.model, revision=a.revision)
        if not hasattr(cls, "all_tied_weights_keys"):
            cls.all_tied_weights_keys = {}
        return AutoModel.from_pretrained(a.model, revision=a.revision, dtype=dtype, trust_remote_code=True, use_flash_attn=False)

    def prepare_image(self, im):
        tiles = dynamic_preprocess(im.convert("RGB"), 1, self.max_tiles, self.image_size, True)
        return torch.stack([self.transform(t) for t in tiles])                           # (n_tiles, 3, 448, 448) float32

    def _query(self, prompt, n_tiles):
        tpl = __import__("copy").deepcopy(self.model.conv_template)
        tpl.system_message = self.model.system_message
        tpl.append_message(tpl.roles[0], "<image>\n" + prompt)
        tpl.append_message(tpl.roles[1], None)
        query = tpl.get_prompt()
        return query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token * n_tiles + "</img>", 1)

    @torch.no_grad()
    def encode(self, rows, device):
        pix = [r[2] for r in rows]
        queries = [self._query(r[3], p.shape[0]) for r, p in zip(rows, pix)]
        enc = self.tok(queries, return_tensors="pt", padding=True)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        vit = self.model.extract_feature(torch.cat(pix).to(device=device, dtype=self.dtype))              # (sum tiles, 256, d_llm)
        emb = self.lm.get_input_embeddings()(ids)
        b, n, c = emb.shape
        sel = (ids.reshape(-1) == self.img_context_id)
        flat = emb.reshape(b * n, c).clone()
        flat[sel] = vit.reshape(-1, c).to(flat.dtype)
        out = self.lm.model(inputs_embeds=flat.reshape(b, n, c), attention_mask=mask)                      # the decoder body: final norm applied, no lm_head; positions 0..n-1 (right padding)
        last = mask.sum(1) - 1                                                                              # the answer position = the last REAL token of every row
        return out.last_hidden_state[torch.arange(b, device=device), last].float()


def make_adapter(a, dtype):
    if getattr(a, "trust_remote_code", False) or a.model == VINTERN["repo"]:
        require_pinned_vintern(a.model, a.revision)
        return VinternAdapter(a, dtype)
    return StdAdapter(a, dtype)
