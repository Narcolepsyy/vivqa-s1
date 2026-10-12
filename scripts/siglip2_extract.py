"""Extract frozen SigLIP 2 vision features for every image of ViVQA-S1 and pack them like the VinVL cache (Phase 2c; plans/viso_phase2c_siglip2_plan.md).
    ~/.venvs/siglip2-gpu/bin/python scripts/siglip2_extract.py --probe 200                   # throughput probe (200 random images, nothing is written to --out)
    ~/.venvs/siglip2-gpu/bin/python scripts/siglip2_extract.py --out ~/data/siglip2_packed   # full extraction (resumable), then --verify
    ~/.venvs/siglip2-gpu/bin/python scripts/siglip2_extract.py --out ~/data/siglip2_packed --verify
The vision tower is FROZEN and used for inference only (eval mode, torch.no_grad). Compute is fp32 by default (--amp enables fp16 autocast: on the GTX 1650 Ti, which has no tensor cores, fp16 is 3.5x SLOWER than fp32: 9 vs 32 img/s); outputs are STORED as fp16. Model: google/siglip2-base-patch16-256 at a pinned revision (Apache-2.0); the checkpoint loads as
SiglipModel, its vision tower (SiglipVisionModel) takes a 256x256 image (the official processor squashes the image to 256x256, mean/std 0.5) -> 16x16 = 256 patch tokens of 768 + the pooled (MAP head) embedding.
Per image we store 65 tokens x 768 (fp16): row 0 = the pooled token (box [0,0,1,1]), rows 1..64 = the 2x2 average-pooled 8x8 patch grid in row-major order (box = the cell of the 8x8 grid).
Only this 8x8 + pooled set is stored (the full 16x16 grid is NOT saved). Layout (little-endian), the same idea as scripts/viso_pack_features.py:
  regions_fp16.bin  float16 [n_images * 65, 768], images in ascending coco_id order
  index.npz         coco_id int64 (sorted), offset int64 (first row), n_regions int32 (= 65)
  boxes.npy         float32 [65, 4] normalised (x1, y1, x2, y2), identical for every image
  meta.json, verify.json   model id / revision / licence, preprocessing, counts, sha256, throughput, coverage, re-extraction check
No API keys involved; weights come from Hugging Face (network is needed once)."""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_ID, REVISION, LICENSE = "google/siglip2-base-patch16-256", "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab", "apache-2.0"
N_TOK, DIM, GRID = 65, 768, 8
BIN, INDEX, META, VERIFY, BOXES, DONE = "regions_fp16.bin", "index.npz", "meta.json", "verify.json", "boxes.npy", "done.npy"
SPLITS = ("train", "calib", "dev", "test")
VERSION = "siglip2_extract v1"


def s1_images(s1_dir):
    """{coco_id: image file name} of every item of the benchmark"""
    out = {}
    for sp in SPLITS:
        with open(os.path.join(s1_dir, f"{sp}.jsonl"), encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    it = json.loads(l)
                    if out.setdefault(it["coco_id"], it["image"]) != it["image"]:
                        raise SystemExit(f"coco_id {it['coco_id']} has two image files")
    return dict(sorted(out.items()))


def grid_boxes():
    b = [[0.0, 0.0, 1.0, 1.0]]
    for r in range(GRID):
        for c in range(GRID):
            b.append([c / GRID, r / GRID, (c + 1) / GRID, (r + 1) / GRID])
    return np.asarray(b, dtype=np.float32)


class Imgs(Dataset):
    def __init__(self, ids, files, img_dir, processor):
        self.ids, self.files, self.dir, self.proc = ids, files, img_dir, processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(os.path.join(self.dir, self.files[self.ids[i]])) as im:
            px = self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return i, px


def load_model(device):
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32).vision_model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)                       # frozen: a feature provider only
    return model, AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION).image_processor


@torch.no_grad()
def tokens_of(model, px, device, fp16):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda" and fp16)):
        out = model(pixel_values=px.to(device))
    h = out.last_hidden_state.float()                                   # (B, 256, 768)
    b = h.shape[0]
    g = F.avg_pool2d(h.view(b, 16, 16, DIM).permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).reshape(b, GRID * GRID, DIM)
    return torch.cat([out.pooler_output.float().unsqueeze(1), g], 1)    # (B, 65, 768) float32


def run(model, ids, files, img_dir, processor, device, fp16, batch, workers, sink, desc=""):
    dl = DataLoader(Imgs(ids, files, img_dir, processor), batch_size=batch, shuffle=False, num_workers=workers, pin_memory=(device == "cuda"))
    t0, n = None, 0
    for k, (idx, px) in enumerate(dl):
        t = tokens_of(model, px, device, fp16).half().cpu().numpy()
        sink(idx.numpy(), t)
        n += len(idx)
        if k == 0:                                                      # the first batch includes worker start-up and CUDA warm-up: exclude it from the throughput
            if device == "cuda":
                torch.cuda.synchronize()
            t0, n0 = time.time(), n
        if desc and (k + 1) % 50 == 0:
            print(f"{desc} {n}/{len(ids)}", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) if t0 else 0.0
    return (n - n0) / dt if dt > 0 and n > n0 else float("nan")


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/siglip2_packed")
    ap.add_argument("--s1-dir", default=os.path.join(REPO, "..", "vivqa_s1_v2", "data"))
    ap.add_argument("--images", default=os.path.join(REPO, "..", "vivqav2_dataset", "images"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe", type=int, default=0, help="throughput probe on N random images; writes nothing")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA (slower on a GTX 1650 Ti)")
    ap.add_argument("--efficiency", action="store_true", help="params / FLOPs / latency of the frozen vision tower (batch 1) -> <out>/vision_tower_efficiency.json")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(a.out))
    files = s1_images(a.s1_dir)
    ids = list(files)
    if a.limit:
        ids = ids[:a.limit]
    fp16 = a.device == "cuda" and a.amp
    if a.efficiency:
        from torch.utils.flop_counter import FlopCounterMode
        model, _ = load_model("cpu")
        tower = sum(p.numel() for p in model.parameters())
        head = sum(p.numel() for n, p in model.named_parameters() if n.startswith("head."))
        x = torch.randn(1, 3, 256, 256, generator=torch.Generator().manual_seed(0))
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(pixel_values=x)
        import statistics
        ts = []
        with torch.no_grad():
            for _ in range(2):
                model(pixel_values=x)
            for _ in range(5):
                t = time.perf_counter()
                model(pixel_values=x)
                ts.append((time.perf_counter() - t) * 1e3)
        eff = {"model_id": MODEL_ID, "revision": REVISION, "frozen": True, "once_per_image": True, "params_total": tower, "params_map_head": head, "params_encoder": tower - head,
               "flops_batch1": int(fc.get_total_flops()), "input": "1 x 3 x 256 x 256 (fp32)", "cpu_latency_ms_median_batch1": round(statistics.median(ts), 1), "cpu_threads": torch.get_num_threads(),
               "note": "the 8x8-pooled + pooled-token cache is read from disk at training / inference time; this tower runs ONCE per image"}
        os.makedirs(out, exist_ok=True)
        json.dump(eff, open(os.path.join(out, "vision_tower_efficiency.json"), "w"), indent=1)
        print(json.dumps(eff, indent=1))
        return
    if a.probe:
        sel = sorted(random.Random(0).sample(ids, a.probe))
        model, proc = load_model(a.device)
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        buf = np.zeros((len(sel), N_TOK, DIM), dtype=np.float16)

        def sink(idx, t):
            buf[idx] = t
        ips = run(model, sel, files, a.images, proc, a.device, fp16, a.batch, a.workers, sink)
        n_all = len(files)
        vram = torch.cuda.max_memory_allocated() / 2 ** 20 if a.device == "cuda" else None
        rep = {"device": a.device, "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None, "batch": a.batch, "workers": a.workers, "probe_images": len(sel), "img_per_s_steady": round(ips, 2),
               "projected_total_minutes": round(n_all / ips / 60, 1), "n_images_total": n_all, "vram_peak_mib": None if vram is None else round(vram),
               "storage_gb_fp16": round(n_all * N_TOK * DIM * 2 / 1e9, 2), "finite": bool(np.isfinite(buf.astype(np.float32)).all())}
        print(json.dumps(rep, indent=1))
        return
    os.makedirs(out, exist_ok=True)
    model, proc = load_model(a.device)
    n = len(ids)
    binp, donep = os.path.join(out, BIN), os.path.join(out, DONE)
    mm = np.memmap(binp, dtype="<f2", mode="r+" if os.path.exists(binp) else "w+", shape=(n * N_TOK, DIM))
    done = np.load(donep) if os.path.exists(donep) else np.zeros(n, dtype=bool)
    if a.verify:
        return verify(out, ids, files, a, model, proc, mm, fp16)
    todo = [i for i in range(n) if not done[i]]
    print(f"{n} images, {len(todo)} to extract on {a.device}", flush=True)
    sub = [ids[i] for i in todo]
    t0 = time.time()
    counter = [0]

    def sink(idx, t):
        for j, row in zip(idx, t):
            g = todo[int(j)]
            mm[g * N_TOK:(g + 1) * N_TOK] = row
            done[g] = True
        counter[0] += len(idx)
        if counter[0] % 640 < len(idx):                                  # checkpoint about every 640 images
            mm.flush()
            np.save(donep, done)
    ips = run(model, sub, files, a.images, proc, a.device, fp16, a.batch, a.workers, sink, desc="extract") if sub else float("nan")
    mm.flush()
    np.save(donep, done)
    assert done.all(), f"{int((~done).sum())} images missing"
    np.savez(os.path.join(out, INDEX), coco_id=np.asarray(ids, dtype=np.int64), offset=np.arange(n, dtype=np.int64) * N_TOK, n_regions=np.full(n, N_TOK, dtype=np.int32))
    np.save(os.path.join(out, BOXES), grid_boxes())
    meta = {"version": VERSION, "model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "dtype": "float16 little-endian", "dim": DIM, "tokens_per_image": N_TOK,
            "layout": "row 0 = pooled (MAP head) token; rows 1..64 = 2x2 average-pooled 8x8 patch grid, row-major; boxes.npy holds the normalised cell boxes", "n_images": n, "regions_shape": [n * N_TOK, DIM],
            "preprocessing": "SiglipImageProcessor: resize to 256x256 (bilinear, aspect ratio NOT kept), rescale 1/255, mean = std = 0.5", "frozen": True, "inference": f"eval, no_grad, {'fp16 autocast' if fp16 else 'fp32 compute'}; stored as fp16",
            "device": a.device, "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None, "torch": torch.__version__, "s1_dir": os.path.abspath(a.s1_dir), "images_dir": os.path.abspath(a.images),
            "img_per_s": None if ips != ips else round(ips, 2), "seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(out, META), "w"), indent=1)
    print("extraction done; run with --verify")


def verify(out, ids, files, a, model, proc, mm, fp16):
    rng = random.Random(1)
    pick = sorted(rng.sample(range(len(ids)), 20))
    sel = [ids[i] for i in pick]
    stored = np.stack([np.asarray(mm[i * N_TOK:(i + 1) * N_TOK]) for i in pick]).astype(np.float32)
    again = np.zeros_like(stored)

    def sink(idx, t):
        again[idx] = t.astype(np.float32)
    run(model, sel, files, a.images, proc, a.device, fp16, 5, 2, sink)                  # a different batch composition than the extraction
    d_re = float(np.abs(stored - again).max())
    cpu_model, _ = load_model("cpu")
    cpu = np.zeros_like(stored)

    def sink_cpu(idx, t):
        cpu[idx] = t.astype(np.float32)
    run(cpu_model, sel, files, a.images, proc, "cpu", False, 5, 2, sink_cpu)
    d_cpu = float(np.abs(stored - cpu).max())
    cos = float((stored * cpu).sum(-1).mean() / (np.linalg.norm(stored, axis=-1) * np.linalg.norm(cpu, axis=-1)).mean())
    z = np.load(os.path.join(out, INDEX))
    missing = sorted(set(files) - set(int(x) for x in z["coco_id"]))
    finite = bool(np.isfinite(np.asarray(mm[:1000 * N_TOK], dtype=np.float32)).all())
    rep = {"images_in_benchmark": len(files), "images_in_cache": int(len(z["coco_id"])), "missing": len(missing), "n_regions_all_65": bool((z["n_regions"] == N_TOK).all()),
           "re_extraction_20_images_max_abs_diff": d_re, "re_extraction_passed": d_re < 1e-3, "cpu_fp32_vs_stored_20_images_max_abs_diff": d_cpu, "cpu_fp32_vs_stored_cosine_mean": cos,
           "first_1000_images_finite": finite, "sha256": {k: sha256_file(os.path.join(out, k)) for k in (BIN, INDEX, BOXES)}, "passed_all": bool(not missing and d_re < 1e-3 and finite)}
    json.dump(rep, open(os.path.join(out, VERIFY), "w"), indent=1)
    meta = json.load(open(os.path.join(out, META)))
    meta["sha256"] = rep["sha256"]
    json.dump(meta, open(os.path.join(out, META), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
