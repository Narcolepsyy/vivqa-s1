"""Decoded, SigLIP-preprocessed images as ONE uint8 memmap [N, 256, 256, 3] (ViSO-FT: the vision tower runs live on every batch, so JPEG decoding must not be the bottleneck).
The pixels are exactly what the SigLIP 2 image processor produces for the cache extraction (scripts/siglip2_extract.py): the processor squashes the image to 256 x 256, rescales by 1/255 and normalises
with mean = std = 0.5; resize happens on uint8, so (px * 0.5 + 0.5) * 255 is an integer and the uint8 array loses nothing. `to_pixel_values` reverses it on the device.
  cache = build_or_load(coco_ids, store, cache_dir)       # ImageStore (utils/coco_images.py) reads + sha256-verifies every JPEG once
  u8 = cache.get(coco_id)                                  # uint8 tensor (3, 256, 256)
The file name carries N and a hash of the id list: another id set builds another file. Size: 196,608 bytes per image (34,893 images = 6.9 GB), shared by every DataLoader worker through the page cache."""
import hashlib
import io
import json
import os
import time

import numpy as np
import torch

SIDE = 256
MODEL_ID, REVISION = "google/siglip2-base-patch16-256", "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"


def to_pixel_values(u8):
    """uint8 (B, 3, 256, 256) on any device -> float32 normalised pixel values (mean = std = 0.5), the processor's output"""
    return (u8.float() / 255.0 - 0.5) / 0.5


class ImageU8Cache:
    def __init__(self, base):
        self.base = base
        z = np.load(base + ".index.npz")
        self.ids = z["coco_id"]
        self.row = {int(c): i for i, c in enumerate(self.ids.tolist())}
        self._mm = None

    def __getstate__(self):                                   # DataLoader workers re-open the memmap lazily
        d = dict(self.__dict__)
        d["_mm"] = None
        return d

    def _memmap(self):
        if self._mm is None:
            self._mm = np.memmap(self.base + ".u8", dtype=np.uint8, mode="r", shape=(len(self.ids), SIDE, SIDE, 3))
        return self._mm

    def __len__(self):
        return len(self.ids)

    def get(self, coco_id):
        return torch.from_numpy(np.array(self._memmap()[self.row[int(coco_id)]])).permute(2, 0, 1).contiguous()


class _Imgs(torch.utils.data.Dataset):
    def __init__(self, ids, store, processor):
        self.ids, self.store, self.proc = ids, store, processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(io.BytesIO(self.store.read_bytes(self.ids[i]))) as im:
            px = self.proc(images=im.convert("RGB"), return_tensors="np")["pixel_values"][0]            # (3, 256, 256) float32
        u8 = np.rint((px * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        return i, torch.from_numpy(u8.transpose(1, 2, 0).copy())


def build_or_load(coco_ids, store, cache_dir, workers=8, log=print):
    ids = sorted({int(c) for c in coco_ids})
    h = hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:10]
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, f"images_{SIDE}_{len(ids)}_{h}")
    if os.path.isfile(base + ".done"):
        return ImageU8Cache(base)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION).image_processor
    log(f"image cache: decoding {len(ids)} images to {base}.u8 ({len(ids) * SIDE * SIDE * 3 / 2 ** 30:.1f} GB), {workers} workers")
    tmp = base + ".u8.tmp"
    mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(len(ids), SIDE, SIDE, 3))
    dl = torch.utils.data.DataLoader(_Imgs(ids, store, proc), batch_size=64, shuffle=False, num_workers=workers, collate_fn=lambda b: b)
    t0, n = time.time(), 0
    for rows in dl:
        for i, arr in rows:
            mm[i] = arr.numpy()
        n += len(rows)
        if (n // 64) % 50 == 0:
            log(f"image cache: {n}/{len(ids)} ({n / max(time.time() - t0, 1e-9):.0f} images/s)")
    mm.flush()
    del mm
    os.replace(tmp, base + ".u8")
    np.savez(base + ".index.npz", coco_id=np.asarray(ids, np.int64))
    json.dump({"n": len(ids), "model": MODEL_ID, "revision": REVISION, "seconds": round(time.time() - t0, 1)}, open(base + ".meta.json", "w"))
    open(base + ".done", "w").close()
    return ImageU8Cache(base)
