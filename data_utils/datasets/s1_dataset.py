import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils import data

from builders.dataset_builder import META_DATASET
from data_utils.utils import preprocess_sentence
from utils.instance import Instance
from utils.viso_s1 import MAX_SPACE, TYPE_IDS, label_space, read_jsonl

LC = 6  # max tokens per candidate text


class S1Batch(Instance):
    """Collated batch with attribute access and .to(device) (the old InstanceList API)."""

    def to(self, device):
        for k, v in list(self.items()):
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self


@META_DATASET.register()
class S1FeatureDataset(data.Dataset):
    """ViVQA-S1 items (jsonl) + VinVL region features. Every sample is an Instance with:
    id, image_id, region_features (R,2048 float32), question_tokens (L), type_id (0 noul / 1 score / 2 choice),
    space_mask (bool[16]), target (float[16], soft gold padded), gold_idx, human_flag, cand_tokens (int[16,LC]),
    cand_mask (bool[16]; choice only) and flat_index (long[16], index of each label in the flat answer vocab,
    -1 if absent). The label space keeps the item's own order (noul [false,true], score 0..10, choice =
    `criteria` order from Phase 1); nothing is sorted."""

    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__()
        self.vocab = vocab
        self.items = read_jsonl(json_path)
        limit = int(getattr(config, "LIMIT", 0))
        if limit > 0:      # tiny smoke runs: type-balanced deterministic subset (first items of each type, in file order)
            per_type = -(-limit // 3)
            taken, counts = [], {"noul": 0, "score": 0, "choice": 0}
            for it in self.items:
                t = it["questions"]["q"]["type"]
                if counts[t] < per_type:
                    counts[t] += 1
                    taken.append(it)
            self.items = taken[:limit]
        # TRAIN-only data options (the other splits are never touched):
        #   NOUL_TRAIN_FRACTION < 1: keep that fraction of the REAL yes/no train items (seeded, stratified by gold label x question form; score / choice stay full) - the learning-curve runs
        #   EXTRA_TRAIN: a jsonl of extra train items appended after that (translated real VQA v2 questions, meta.extra_vqa = true; their images must be in the packed features)
        if os.path.basename(json_path).startswith("train"):
            frac = float(getattr(config, "NOUL_TRAIN_FRACTION", 1.0))
            if frac < 1.0:
                import random
                from utils.viso_s1 import yesno_form
                groups = {}
                for k, it in enumerate(self.items):
                    if it["questions"]["q"]["type"] == "noul":
                        groups.setdefault((it["gold_label"], yesno_form(it)), []).append(k)
                drop = set()
                for key, ks in groups.items():
                    keep = set(random.Random(f"{int(getattr(config, 'NOUL_SUBSET_SEED', 0))}:{key}").sample(ks, max(1, round(frac * len(ks)))))
                    drop |= set(ks) - keep
                self.items = [it for k, it in enumerate(self.items) if k not in drop]
            extra = getattr(config, "EXTRA_TRAIN", "") or ""
            if extra:
                self.items = self.items + read_jsonl(os.path.expanduser(extra))
        # TEACHER_TRAIN: jsonl {id, p} of teacher distributions for the TRAIN split only (knowledge distillation, LOSS.KD_WEIGHT); items without a line (the teacher refused) have has_teacher = False
        self.teacher = {}
        if os.path.basename(json_path).startswith("train") and (getattr(config, "TEACHER_TRAIN", "") or ""):
            self.teacher = {r["id"]: r["p"] for r in read_jsonl(os.path.expanduser(config.TEACHER_TRAIN))}
        self.features_path = config.FEATURE_PATH.FEATURES
        # FEATURE_FORMAT: "npy" (default, one dict per image) | "packed" (one fp16 array, scripts/viso_pack_features.py)
        self.feature_format = getattr(config, "FEATURE_FORMAT", "npy")
        if self.feature_format not in ("npy", "packed"):
            raise ValueError(f"FEATURE_FORMAT must be npy or packed, got {self.feature_format!r}")
        self.packed = None
        if self.feature_format == "packed":
            from utils.viso_packed_features import PackedFeatures
            self.packed = PackedFeatures(getattr(config, "PACKED_PATH", "~/data/vivqa_packed"))
        if self.packed is not None and os.path.basename(json_path).startswith("train") and (getattr(config, "EXTRA_TRAIN", "") or ""):
            missing = {it["coco_id"] for it in self.items if it.get("meta", {}).get("extra_vqa")}
            missing = [c for c in missing if not self.packed.has(c)]
            if missing:
                raise ValueError(f"EXTRA_TRAIN: {len(missing)} images are not in the packed features, e.g. {missing[:5]}")
        self.use_boxes = bool(getattr(config, "USE_BOXES", False))      # SigLIP 2 caches: return the shared patch-cell boxes as `region_boxes` (R,4); needs FEATURE_FORMAT packed + a boxes.npy
        if self.use_boxes and (self.packed is None or self.packed.boxes is None):
            raise ValueError("USE_BOXES needs FEATURE_FORMAT packed and a boxes.npy in the packed directory")
        # TEXT_CACHE: frozen BamiBERT features (scripts/bamibert_extract.py): the batch then also carries question_states (L,768) and, for choice items, cand_vecs (16,768)
        tc = getattr(config, "TEXT_CACHE", None)
        self.text_cache = None
        if tc:
            from utils.viso_text_cache import TextCache
            self.text_cache = TextCache(tc, getattr(config, "TEXT_CACHE_POOL", "mean"), bool(getattr(config, "TEXT_CACHE_STANDARDIZE", False)))
        self.cache_size = int(getattr(config, "FEATURE_CACHE", 0))
        self._cache = OrderedDict()
        self.index = {it["id"]: i for i, it in enumerate(self.items)}   # id -> position (no O(N*M) lookups)

    def __len__(self):
        return len(self.items)

    def load_regions(self, coco_id: int) -> np.ndarray:
        if self.packed is not None:                 # lazy memmap per process; fp16 -> float32
            return self.packed.get_array(coco_id)
        if self.cache_size and coco_id in self._cache:
            self._cache.move_to_end(coco_id)
            return self._cache[coco_id]
        feats = np.load(os.path.join(self.features_path, f"{coco_id}.npy"), allow_pickle=True)[()]
        regions = np.asarray(feats["region_features"], dtype=np.float32)
        if self.cache_size:
            self._cache[coco_id] = regions
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return regions

    def __getitem__(self, idx: int):
        it = self.items[idx]
        q = it["questions"]["q"]
        space = label_space(q)
        n, vocab = len(space), self.vocab
        assert n <= MAX_SPACE, f"label space of item {it['id']} has {n} > {MAX_SPACE} entries"
        probs = it["gold"]["q"]["probabilities"]
        target = torch.zeros(MAX_SPACE)
        for k, key in enumerate(space):
            target[k] = probs.get(key, 0.0)
        space_mask = torch.zeros(MAX_SPACE, dtype=torch.bool)
        space_mask[:n] = True
        cand_tokens = torch.full((MAX_SPACE, LC), vocab.padding_idx, dtype=torch.long)
        cand_mask = torch.zeros(MAX_SPACE, dtype=torch.bool)
        if q["type"] == "choice":
            for k, text in enumerate(space):
                ids = vocab.encode_candidate(text, LC)
                cand_tokens[k, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            cand_mask[:n] = True
        flat_index = torch.full((MAX_SPACE,), -1, dtype=torch.long)
        for k, key in enumerate(space):
            flat_index[k] = vocab.flat_index(q["type"], key) if hasattr(vocab, "flat_index") else -1
        question_tokens = vocab.encode_question(preprocess_sentence(q["instructions"], vocab.tokenizer))
        regions = self.load_regions(it["coco_id"])
        extra = {"region_boxes": torch.from_numpy(self.packed.get_boxes(regions.shape[0]).copy())} if self.use_boxes else {}
        if self.text_cache is not None:
            tcx = self.text_cache
            cand_vecs = torch.zeros(MAX_SPACE, tcx.dim)
            if q["type"] == "choice":
                for k, text in enumerate(space):
                    cand_vecs[k] = torch.from_numpy(tcx.candidate(text))
            extra.update(question_states=torch.from_numpy(tcx.question(q["instructions"])), cand_vecs=cand_vecs)
        if self.teacher:
            tp = self.teacher.get(it["id"])
            tt = torch.zeros(MAX_SPACE)
            if tp is not None:
                assert len(tp) == n, f"teacher vector of item {it['id']} has {len(tp)} entries, the label space has {n}"
                tt[:n] = torch.tensor(tp, dtype=torch.float32)
            extra.update(teacher_target=tt, has_teacher=torch.tensor(tp is not None))
        return Instance(
            id=it["id"], image_id=it["coco_id"],
            region_features=torch.from_numpy(regions), **extra,
            question_tokens=question_tokens, type_id=TYPE_IDS[q["type"]],
            space_mask=space_mask, target=target, gold_idx=space.index(it["gold_label"]),
            human_flag=bool(it["human_subset"]), cand_tokens=cand_tokens, cand_mask=cand_mask,
            flat_index=flat_index)


def s1_collate(samples):
    """Pad region_features to the batch maximum with zero rows and build the explicit boolean `region_mask`
    (True = real region); everything else is stacked."""
    bs = len(samples)
    r_max = max(s.region_features.shape[0] for s in samples)
    d = samples[0].region_features.shape[1]
    regions = torch.zeros(bs, r_max, d)
    region_mask = torch.zeros(bs, r_max, dtype=torch.bool)
    for i, s in enumerate(samples):
        r = s.region_features.shape[0]
        regions[i, :r] = s.region_features
        region_mask[i, :r] = True
    stack = lambda key: torch.stack([s[key] for s in samples])
    extra = {}
    if "region_boxes" in samples[0]:
        boxes = torch.zeros(bs, r_max, 4)
        for i, s in enumerate(samples):
            boxes[i, :s.region_boxes.shape[0]] = s.region_boxes
        extra["region_boxes"] = boxes
    if "teacher_target" in samples[0]:
        extra.update(teacher_target=stack("teacher_target"), has_teacher=stack("has_teacher"))
    if "question_states" in samples[0]:
        l_max = max(s.question_states.shape[0] for s in samples)
        states = torch.zeros(bs, l_max, samples[0].question_states.shape[1])
        smask = torch.zeros(bs, l_max, dtype=torch.bool)
        for i, s in enumerate(samples):
            n_i = s.question_states.shape[0]
            states[i, :n_i] = s.question_states
            smask[i, :n_i] = True
        extra.update(question_states=states, question_state_mask=smask, cand_vecs=stack("cand_vecs"))
    if "image_u8" in samples[0]:                                  # ViSO-FT (data_utils/datasets/s1_live_dataset.py): raw image / token inputs for the live encoders
        pad = int(samples[0]["pad_id"])
        l_max = max(s["q_ids"].shape[0] for s in samples)
        q_ids = torch.full((bs, l_max), pad, dtype=torch.long)
        q_mask = torch.zeros(bs, l_max, dtype=torch.bool)
        lc_max = max(s["cand_ids"].shape[1] for s in samples)
        cand_ids = torch.full((bs, samples[0]["cand_ids"].shape[0], lc_max), pad, dtype=torch.long)
        for i, s in enumerate(samples):
            n_i = s["q_ids"].shape[0]
            q_ids[i, :n_i] = s["q_ids"]
            q_mask[i, :n_i] = True
            cand_ids[i, :, :s["cand_ids"].shape[1]] = s["cand_ids"]
        extra.update(image_u8=stack("image_u8"), q_ids=q_ids, q_mask=q_mask, cand_ids=cand_ids, cand_pad_mask=cand_ids != pad)
    return S1Batch(
        id=torch.tensor([s.id for s in samples], dtype=torch.long),
        image_id=torch.tensor([s.image_id for s in samples], dtype=torch.long),
        region_features=regions, region_mask=region_mask, question_tokens=stack("question_tokens"),
        type_id=torch.tensor([s.type_id for s in samples], dtype=torch.long),
        space_mask=stack("space_mask"), target=stack("target"),
        gold_idx=torch.tensor([s.gold_idx for s in samples], dtype=torch.long),
        human_flag=torch.tensor([s.human_flag for s in samples], dtype=torch.bool),
        cand_tokens=stack("cand_tokens"), cand_mask=stack("cand_mask"), flat_index=stack("flat_index"), **extra)
