"""ViVQA-S1 items for ViSO-FT (live encoders): the same targets / label spaces / masks as S1FeatureDataset, but instead of cached features each item carries RAW inputs:
  image_u8  uint8 (3, 256, 256)  the SigLIP-preprocessed image (utils/image_u8_cache.py; memmap cache, or decoded on the fly when no cache exists: tiny / smoke runs)
  q_ids     long (L,)            BamiBERT token ids of the question (<s> ... </s>, max 48 tokens, the same tokenizer call as scripts/bamibert_extract.py)
  cand_ids  long (16, Lc)        BamiBERT token ids of every candidate string of a choice item (padded with the pad id)
`s1_collate` pads them to the batch maximum and adds q_mask / cand_pad_mask. The base class is built with FEATURE_FORMAT npy / USE_BOXES False / TEXT_CACHE '' (no cached features are read)."""
import os

import numpy as np
import torch

from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from utils.viso_s1 import MAX_SPACE, label_space, read_jsonl

from .s1_dataset import S1FeatureDataset

MAX_TOK = 48
_SPLITS = ("train", "calib", "dev", "test")


def all_coco_ids(s1_dir):
    ids = set()
    for sp in _SPLITS:
        path = os.path.join(s1_dir, f"{sp}.jsonl")
        if os.path.isfile(path):
            ids.update(int(it["coco_id"]) for it in read_jsonl(path))
    return sorted(ids)


def load_tokenizer(model_id, revision):
    """BamiBERT's tokenizer: the shipped tokenizer.json (byte-level BPE; <s>=0 <pad>=1 </s>=2) exactly as scripts/bamibert_extract.py loads it (tokenizer_config.json names a class that transformers >= 5.6 cannot build)"""
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(model_id, "tokenizer.json", revision=revision), bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                   sep_token="</s>", pad_token="<pad>", cls_token="<s>", mask_token="<mask>")


class _OnTheFly:
    """no cache: decode + preprocess an image when it is asked for (smoke tests)"""

    def __init__(self, store):
        from transformers import AutoProcessor
        from utils.image_u8_cache import MODEL_ID, REVISION
        self.store = store
        self.proc = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION).image_processor

    def get(self, coco_id):
        px = self.proc(images=self.store.open(coco_id), return_tensors="np")["pixel_values"][0]
        return torch.from_numpy(np.rint((px * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8))


@META_DATASET.register()
class S1LiveDataset(S1FeatureDataset):
    def __init__(self, json_path, vocab, config):
        super().__init__(json_path, vocab, config)
        live = config.LIVE
        self.tok = load_tokenizer(live.TEXT_MODEL, live.TEXT_REVISION)
        self.pad_id = int(self.tok.pad_token_id)
        self.images = None
        self._live = live
        self._json = json_path

    def _image_source(self):
        if self.images is None:
            from utils.coco_images import ImageStore
            live = self._live
            base = os.path.expanduser(live.IMAGE_CACHE_DIR) if getattr(live, "IMAGE_CACHE_DIR", "") else ""
            from utils.image_u8_cache import ImageU8Cache
            found = None
            if base and os.path.isdir(base):
                import glob
                for done in sorted(glob.glob(os.path.join(base, "images_*.done"))):
                    c = ImageU8Cache(done[:-len(".done")])
                    ids = {int(it["coco_id"]) for it in self.items}
                    if ids <= set(c.row):
                        found = c
                        break
            if found is None:
                store = ImageStore(os.path.expanduser(live.SHA_LIST), roots=[os.path.expanduser(r) for r in live.IMAGE_ROOTS], verify=bool(getattr(live, "VERIFY", True)))
                found = _OnTheFly(store)
            self.images = found
        return self.images

    def __getstate__(self):
        d = dict(self.__dict__)
        return d

    def load_regions(self, coco_id):
        return np.zeros((1, 768), np.float32)                  # placeholder: the live tower replaces region_features

    def __getitem__(self, idx):
        inst = super().__getitem__(idx)
        it = self.items[idx]
        q = it["questions"]["q"]
        qi = self.tok(q["instructions"], truncation=True, max_length=MAX_TOK)["input_ids"]
        cand = torch.full((MAX_SPACE, 1), self.pad_id, dtype=torch.long)
        if q["type"] == "choice":
            enc = [self.tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in label_space(q)]
            lc = max(len(e) for e in enc)
            cand = torch.full((MAX_SPACE, lc), self.pad_id, dtype=torch.long)
            for k, e in enumerate(enc):
                cand[k, :len(e)] = torch.tensor(e, dtype=torch.long)
        inst.update(image_u8=self._image_source().get(it["coco_id"]), q_ids=torch.tensor(qi, dtype=torch.long), cand_ids=cand, pad_id=self.pad_id)
        return inst
