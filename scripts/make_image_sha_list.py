"""sha256 list of every image of ViVQA-S1 (the local copies are byte-identical to the original MS COCO 2014 files): kaggle/s1_images_sha256.tsv.gz
    python scripts/make_image_sha_list.py [--images ../vivqav2_dataset/images]
Columns: coco_id, our file name, COCO file name (COCO_{train,val}2014_%012d.jpg), size in bytes, sha256. The kernels read the COCO images from a public Kaggle COCO dataset (or download them from
images.cocodataset.org) and VERIFY every file against this list: a mismatch stops the run (utils/coco_images.py)."""
import argparse
import gzip
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJ = os.path.abspath(os.path.join(REPO, ".."))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest(), os.path.getsize(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(PROJ, "vivqav2_dataset", "images"))
    ap.add_argument("--s1-data", default=os.path.join(PROJ, "vivqa_s1_v2", "data"))
    ap.add_argument("--vqa", default=os.path.join(PROJ, "external", "vqav2"))
    ap.add_argument("--out", default=os.path.join(REPO, "kaggle", "s1_images_sha256.tsv.gz"))
    a = ap.parse_args(argv)
    used = {}
    for sp in ("train", "calib", "dev", "test"):
        for l in open(os.path.join(a.s1_data, f"{sp}.jsonl"), encoding="utf-8"):
            it = json.loads(l)
            used[it["coco_id"]] = it["image"]
    split_of = {}
    for sub in ("train", "val"):
        for q in json.load(open(os.path.join(a.vqa, f"v2_OpenEnded_mscoco_{sub}2014_questions.json")))["questions"]:
            split_of.setdefault(q["image_id"], sub)
    ids = sorted(used)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda i: sha(os.path.join(a.images, used[i])), ids))
    with gzip.open(a.out, "wt", encoding="utf-8", newline="\n") as f:
        f.write("coco_id\tour_name\tcoco_name\tbytes\tsha256\n")
        for i, (h, n) in zip(ids, res):
            f.write(f"{i}\t{used[i]}\tCOCO_{split_of[i]}2014_{i:012d}.jpg\t{n}\t{h}\n")
    print(f"{len(ids)} images -> {a.out} ({os.path.getsize(a.out) / 2 ** 20:.2f} MB)")


if __name__ == "__main__":
    main()
