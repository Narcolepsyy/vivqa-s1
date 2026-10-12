"""Pack the VinVL region features of every image used by ViVQA-S1 into ONE float16 array (outside the repo).
    nice -n 10 ionice -c3 python scripts/viso_pack_features.py --out ~/data/vivqa_packed            # pack
    python scripts/viso_pack_features.py --out ~/data/vivqa_packed --verify                          # verify.json
Streamed image by image (< 1 GB RAM), ids sorted, deterministic (byte-identical files on rerun; meta.json holds the
creation date and is excluded from that comparison). Source files are only read; page cache of the source and the output
is dropped periodically so the machine stays responsive."""
import argparse
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import time
import zipfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from utils.viso_packed_features import BIN, DIM, INDEX, META, TAGS, PackedFeatures  # noqa: E402

VERSION = "viso_pack_features v1"
SPLITS = ("train", "calib", "dev", "test")
DEFAULT_SRC = os.path.join(REPO, "..", "vivqav2_dataset", "vinvl_vinvl")
DEFAULT_S1 = os.path.join(REPO, "..", "vivqa_s1_v2", "data")


def s1_ids(s1_dir):
    ids = set()
    for sp in SPLITS:
        with open(os.path.join(s1_dir, f"{sp}.jsonl"), encoding="utf-8") as f:
            ids.update(json.loads(l)["coco_id"] for l in f if l.strip())
    return sorted(ids)


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def load_source(src, coco_id):
    """np.load of one source dict, then drop its pages from the cache. (The project's own pipeline loads these
    files with allow_pickle as well.)"""
    with open(os.path.join(src, f"{coco_id}.npy"), "rb") as f:
        d = np.load(f, allow_pickle=True)[()]
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return d


def write_index(path, coco_id, offset, n_regions):
    """Deterministic .npz (fixed zip timestamps): np.savez would embed the current time and change the bytes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, arr in (("coco_id", coco_id), ("offset", offset), ("n_regions", n_regions)):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arr, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            z.writestr(info, buf.getvalue())


def pack(src, s1_dir, out, limit=0, log=print):
    ids = s1_ids(s1_dir)
    if limit:
        ids = ids[:limit]
    missing = [i for i in ids if not os.path.isfile(os.path.join(src, f"{i}.npy"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source .npy files are missing, e.g. {missing[:20]}")
    os.makedirs(out, exist_ok=True)
    offsets, counts = [], []
    row, t0 = 0, time.time()
    h_bin = hashlib.sha256()
    with open(os.path.join(out, BIN), "wb") as fb, open(os.path.join(out, TAGS), "w", encoding="utf-8", newline="\n") as ft:
        pending = 0
        for k, cid in enumerate(ids):
            d = load_source(src, cid)
            rf = np.asarray(d["region_features"])
            if rf.ndim != 2 or rf.shape[1] != DIM or rf.dtype not in (np.float32, np.float64, np.float16) or rf.shape[0] < 1:
                raise ValueError(f"image {cid}: region_features has shape {rf.shape} dtype {rf.dtype}, expected (n, {DIM}) float")
            h16 = rf.astype("<f2")
            if not np.isfinite(h16).all() or not np.isfinite(rf).all():
                raise ValueError(f"image {cid}: non-finite values after the float16 cast (overflow?)")
            raw = h16.tobytes(order="C")
            fb.write(raw)
            h_bin.update(raw)
            offsets.append(row)
            counts.append(h16.shape[0])
            row += h16.shape[0]
            ft.write(json.dumps({"coco_id": cid, "objects": _plain(d.get("objects")), "attributes": _plain(d.get("attributes"))},
                                ensure_ascii=False) + "\n")
            pending += len(raw)
            if pending >= (256 << 20):                      # keep the page cache small
                fb.flush(); os.fsync(fb.fileno())
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fb.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                pending = 0
            if (k + 1) % 2000 == 0:
                log(f"packed {k + 1}/{len(ids)} images, {row} regions, {time.time() - t0:.0f}s")
        fb.flush(); os.fsync(fb.fileno())
    write_index(os.path.join(out, INDEX), np.asarray(ids, np.int64), np.asarray(offsets, np.int64), np.asarray(counts, np.int32))
    meta = {"version": VERSION, "dtype": "float16 little-endian", "dim": DIM, "n_images": len(ids), "total_regions": row,
            "regions_shape": [row, DIM], "source_dir": os.path.abspath(src), "s1_dir": os.path.abspath(s1_dir),
            "split_files_used": [f"{s}.jsonl" for s in SPLITS], "limit": limit,
            "regions_per_image": {"min": int(min(counts)), "max": int(max(counts)), "mean": round(float(np.mean(counts)), 3)},
            "sha256": {BIN: h_bin.hexdigest(), INDEX: sha256_file(os.path.join(out, INDEX)), TAGS: sha256_file(os.path.join(out, TAGS))},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "pack_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, META), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log(f"done: {len(ids)} images, {row} regions, {os.path.getsize(os.path.join(out, BIN)) / 2 ** 30:.2f} GiB, {time.time() - t0:.0f}s")
    return meta


def _plain(x):
    """objects / attributes -> JSON-able python lists."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


# --------------------------------------------------------------------------------------------- verification
def verify(src, s1_dir, out, seed=0, n_spot=1000, n_tags=100, n_time=2000, log=print):
    res, ok = {}, True
    meta = json.load(open(os.path.join(out, META), encoding="utf-8"))
    pf = PackedFeatures(out)
    ids = s1_ids(s1_dir)
    size = os.path.getsize(os.path.join(out, BIN))
    res["1_counts"] = {"n_images_pack": len(pf), "n_images_s1_union": len(ids), "sum_n_regions": int(pf.n_regions.sum()),
                       "total_regions_meta": meta["total_regions"], "file_rows": size // (2 * DIM), "file_size_bytes": size,
                       "size_divisible": size % (2 * DIM) == 0, "sorted_ids": bool((np.diff(pf.coco_id) > 0).all()),
                       "ids_equal_s1_union": bool(np.array_equal(pf.coco_id, np.asarray(ids))),
                       "offsets_consistent": bool(np.array_equal(pf.offset, np.concatenate([[0], np.cumsum(pf.n_regions)[:-1]])))}
    c = res["1_counts"]
    c["passed"] = (c["n_images_pack"] == c["n_images_s1_union"] and c["sum_n_regions"] == c["total_regions_meta"] == c["file_rows"]
                   and c["size_divisible"] and c["sorted_ids"] and c["ids_equal_s1_union"] and c["offsets_consistent"])
    rng = random.Random(seed)
    spot = rng.sample(ids, min(n_spot, len(ids)))
    worst_rel, worst_id, n_mismatch = 0.0, None, 0
    for cid in spot:
        rf = np.asarray(load_source(src, cid)["region_features"], dtype=np.float32)
        a = pf.get_array(cid)
        if a.shape != rf.shape:
            n_mismatch += 1
            continue
        rel = float(np.abs(a - rf).max() / max(float(np.abs(rf).max()), 1e-12))
        if rel > worst_rel:
            worst_rel, worst_id = rel, cid
    res["2_spot_check"] = {"n_ids": len(spot), "seed": seed, "shape_mismatches": n_mismatch, "worst_rel_err_vs_max_abs": worst_rel,
                           "worst_id": worst_id, "tolerance": 1e-3, "passed": n_mismatch == 0 and worst_rel <= 1e-3}
    tag_rows = {}
    with open(os.path.join(out, TAGS), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tag_rows[r["coco_id"]] = r
    bad = 0
    for cid in rng.sample(ids, min(n_tags, len(ids))):
        d = load_source(src, cid)
        if tag_rows[cid]["objects"] != _plain(d.get("objects")) or tag_rows[cid]["attributes"] != _plain(d.get("attributes")):
            bad += 1
    res["3_tags"] = {"n_ids": min(n_tags, len(ids)), "n_tag_lines": len(tag_rows), "n_unequal": bad,
                     "passed": bad == 0 and len(tag_rows) == len(ids)}
    res["4_forward_equivalence"] = forward_equivalence(s1_dir, out)
    # timing: items/s for packed vs npy over the same random ids
    t_ids = rng.sample(ids, min(n_time, len(ids)))
    t = time.perf_counter()
    for cid in t_ids:
        pf.get_array(cid)
    t_packed = time.perf_counter() - t
    t = time.perf_counter()
    for cid in t_ids:
        np.asarray(load_source(src, cid)["region_features"], dtype=np.float32)
    t_npy = time.perf_counter() - t
    res["5_timing"] = {"n_items": len(t_ids), "packed_items_per_s": round(len(t_ids) / t_packed, 1),
                       "npy_items_per_s": round(len(t_ids) / t_npy, 1), "speedup": round(t_npy / t_packed, 2),
                       "note": "single process, warm-ish page cache for the packed file; the npy loads drop their pages"}
    res["6_determinism"] = determinism(src, s1_dir, out, meta, log)
    res["passed_all"] = all(res[k]["passed"] for k in ("1_counts", "2_spot_check", "3_tags", "4_forward_equivalence", "6_determinism"))
    with open(os.path.join(out, "verify.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    log(json.dumps(res, indent=1))
    return res


def forward_equivalence(s1_dir, out, n_items=8, tol=5e-3):
    """ViSO and FlatS1 forward on the same real dev items, npy vs packed features: max |dh| of the fused vector."""
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import data_utils.datasets.s1_dataset as ds
    import data_utils.vocabs.s1_vocab  # noqa: F401
    import models.viso  # noqa: F401
    from builders.model_builder import build_model
    from builders.vocab_builder import build_vocab
    from train_viso import load_config
    res = {"n_items": n_items, "tolerance": tol}
    for arch, cfgfile in (("ViSO", "configs/viso_vivqav2.yaml"), ("FlatS1", "configs/viso_flat_vivqav2.yaml")):
        cfg = load_config(os.path.join(REPO, cfgfile), ["MODEL.DEVICE", "cpu", "DATASET.FEATURE_DATASET.LIMIT", "24"])
        torch.manual_seed(0)
        vocab = build_vocab(cfg.DATASET.VOCAB)
        model = build_model(cfg.MODEL, vocab).eval()
        fd_npy, fd_pk = cfg.DATASET.FEATURE_DATASET.clone(), cfg.DATASET.FEATURE_DATASET.clone()
        fd_pk.FEATURE_FORMAT, fd_pk.PACKED_PATH = "packed", out
        path = os.path.join(s1_dir, "dev.jsonl")
        d1, d2 = ds.S1FeatureDataset(path, vocab, fd_npy), ds.S1FeatureDataset(path, vocab, fd_pk)
        idx = list(range(0, len(d1), max(1, len(d1) // n_items)))[:n_items]
        with torch.no_grad():
            h1 = model(ds.s1_collate([d1[i] for i in idx]))["h"]
            h2 = model(ds.s1_collate([d2[i] for i in idx]))["h"]
        res[arch] = {"max_abs_dh": float((h1 - h2).abs().max()), "max_abs_h": float(h1.abs().max())}
    res["passed"] = all(res[a]["max_abs_dh"] <= tol for a in ("ViSO", "FlatS1"))
    return res


def determinism(src, s1_dir, out, meta, log, n=500):
    """Pack the first n images twice into temp dirs: the three sha256 must be equal (and equal to the full pack's
    own sha256 only when n covers all images, so they are compared with each other)."""
    quiet = lambda *_: None
    with tempfile.TemporaryDirectory(prefix="vivqa_pack_det_", dir=os.path.dirname(os.path.abspath(out))) as tmp:
        a = pack(src, s1_dir, os.path.join(tmp, "a"), limit=n, log=quiet)
        b = pack(src, s1_dir, os.path.join(tmp, "b"), limit=n, log=quiet)
    equal = {k: a["sha256"][k] == b["sha256"][k] for k in a["sha256"]}
    return {"limit": n, "sha256_run_a": a["sha256"], "sha256_run_b": b["sha256"], "equal": equal, "passed": all(equal.values())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--s1-dir", default=DEFAULT_S1)
    ap.add_argument("--out", default=os.path.expanduser("~/data/vivqa_packed"))
    ap.add_argument("--limit", type=int, default=0, help="pack only the first N image ids (tests / determinism)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    out = os.path.abspath(os.path.expanduser(args.out))
    repo = REPO
    if os.path.commonpath([out, repo]) == repo or os.path.commonpath([out, os.path.abspath(os.path.join(repo, ".."))]) == os.path.abspath(os.path.join(repo, "..")):
        raise SystemExit(f"refusing to write inside the project folder: {out}")
    if args.verify:
        res = verify(args.src, args.s1_dir, out)
        raise SystemExit(0 if res["passed_all"] else 1)
    pack(args.src, args.s1_dir, out, limit=args.limit)


if __name__ == "__main__":
    main()
