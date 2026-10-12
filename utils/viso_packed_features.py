"""Packed VinVL region features: ONE half-precision array for all images (see scripts/viso_pack_features.py).

Directory layout (little-endian):
  regions_fp16.bin  float16 [total_regions, 2048], images concatenated in ascending coco_id order, exact lengths
  index.npz         coco_id int64 (sorted), offset int64 (first row of the image), n_regions int32
  tags.jsonl        {"coco_id", "objects", "attributes"} per image, same order
  meta.json         dtype, dim, counts, source, sha256 of the three files above, ...
`PackedFeatures` opens the memmap lazily on first access and drops it when pickled, so every DataLoader worker
(fork or spawn) maps the file itself."""
import json
import os

import numpy as np
import torch

DIM = 2048
BIN, INDEX, TAGS, META = "regions_fp16.bin", "index.npz", "tags.jsonl", "meta.json"
DEFAULT_PATH = "~/data/vivqa_packed"
BOXES = "boxes.npy"                          # SigLIP 2 caches: one shared [n_regions, 4] normalised box array (the same for every image); absent in the VinVL cache


class PackedFeatures:
    def __init__(self, path=DEFAULT_PATH):
        self.path = os.path.abspath(os.path.expanduser(path))
        with np.load(os.path.join(self.path, INDEX)) as z:
            self.coco_id, self.offset, self.n_regions = z["coco_id"], z["offset"], z["n_regions"]
        self.total_regions = int(self.offset[-1] + self.n_regions[-1]) if len(self.coco_id) else 0
        self.dim = DIM                       # 2048 (VinVL) unless meta.json says otherwise (SigLIP 2: 768)
        mp = os.path.join(self.path, META)
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                self.dim = int(json.load(f).get("dim", DIM))
        bp = os.path.join(self.path, BOXES)
        self.boxes = np.load(bp).astype(np.float32) if os.path.exists(bp) else None
        self._mm = None

    def __getstate__(self):                # never ship a memmap to a worker process
        state = dict(self.__dict__)
        state["_mm"] = None
        return state

    def __len__(self):
        return len(self.coco_id)

    def _memmap(self):
        if self._mm is None:
            self._mm = np.memmap(os.path.join(self.path, BIN), dtype="<f2", mode="r", shape=(self.total_regions, self.dim))
        return self._mm

    def get_boxes(self, n):
        """normalised boxes (n, 4) float32 of the first n rows of an image (SigLIP 2 caches), or None when the cache has no boxes"""
        return None if self.boxes is None else self.boxes[:n]

    def row(self, coco_id) -> int:
        i = int(np.searchsorted(self.coco_id, coco_id))
        if i >= len(self.coco_id) or self.coco_id[i] != coco_id:
            raise KeyError(f"coco_id {coco_id} is not in the packed features at {self.path}")
        return i

    def has(self, coco_id) -> bool:
        try:
            self.row(coco_id)
            return True
        except KeyError:
            return False

    def get_array(self, coco_id) -> np.ndarray:
        """region_features of one image as a float32 numpy array (n, dim) (cast from fp16); dim = 2048 for VinVL, 768 for SigLIP 2."""
        i = self.row(coco_id)
        o, n = int(self.offset[i]), int(self.n_regions[i])
        return np.asarray(self._memmap()[o:o + n], dtype=np.float32)

    def get(self, coco_id):
        """-> (float32 torch tensor (n, dim), n_regions)."""
        arr = self.get_array(coco_id)
        return torch.from_numpy(arr), arr.shape[0]
