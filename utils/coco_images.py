"""Image access for the kernels without uploading our 6 GB image folder: the benchmark images are the original MS COCO 2014 files (byte-identical, checked against images.cocodataset.org).
ImageStore finds an image by its COCO name (COCO_{train,val}2014_%012d.jpg) or by our name (%06d.jpg) under the given roots (a public Kaggle COCO dataset mounted anywhere under /kaggle/input), optionally
downloads the missing ones from images.cocodataset.org, and VERIFIES every file read against kaggle/s1_images_sha256.tsv.gz (scripts/make_image_sha_list.py): a mismatch or a missing image raises
(fail loudly; a wrong image would silently corrupt a baseline)."""
import csv
import gzip
import hashlib
import io
import os
import urllib.request

COCO_URL = "http://images.cocodataset.org/{sub}/{name}"


def resolve_sha_list(path):
    """Kaggle AUTO-DECOMPRESSES uploaded *.gz files, so a downloaded dataset has kaggle/s1_images_sha256.tsv, not .tsv.gz (same content): if `path` is missing, try the name with / without the .gz"""
    for p in (path, path[:-3] if path.endswith(".gz") else path + ".gz"):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"image sha list not found: {path} (nor its .gz / plain twin)")


def open_sha_list(path):
    """text handle of the sha list, plain or gzip (decided by the magic bytes, not by the name)"""
    p = resolve_sha_list(path)
    with open(p, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    return gzip.open(p, "rt", encoding="utf-8") if gz else open(p, "rt", encoding="utf-8")


class ImageStore:
    def __init__(self, sha_tsv, roots=(), cache_dir=None, allow_download=False, url_template=COCO_URL, verify=True):
        with open_sha_list(sha_tsv) as f:
            self.meta = {int(r["coco_id"]): r for r in csv.DictReader(f, delimiter="\t")}
        self.cache_dir, self.allow_download, self.url_template, self.verify = cache_dir, allow_download, url_template, verify
        wanted = {r["coco_name"] for r in self.meta.values()} | {r["our_name"] for r in self.meta.values()}
        self.index = {}                                                          # file name -> path (first hit wins)
        for root in roots:
            for d, _, fs in os.walk(os.path.expanduser(root)):
                for f in fs:
                    if f in wanted and f not in self.index:
                        self.index[f] = os.path.join(d, f)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.meta)

    def path(self, coco_id):
        r = self.meta[int(coco_id)]
        return self.index.get(r["coco_name"]) or self.index.get(r["our_name"])

    def read_bytes(self, coco_id):
        r = self.meta[int(coco_id)]
        p = self.path(coco_id)
        if p is not None:
            data = open(p, "rb").read()
        elif self.allow_download:
            data = self._download(r)
        else:
            raise FileNotFoundError(f"image {r['coco_name']} (coco_id {coco_id}) not found under the image roots and downloading is off")
        if self.verify and hashlib.sha256(data).hexdigest() != r["sha256"]:
            raise RuntimeError(f"image {r['coco_name']} (coco_id {coco_id}) does not match the benchmark sha256 list: got {hashlib.sha256(data).hexdigest()[:16]}..., want {r['sha256'][:16]}... (path {p})")
        return data

    def _download(self, r):
        sub = "train2014" if "_train2014_" in r["coco_name"] else "val2014"
        cached = os.path.join(self.cache_dir, r["coco_name"]) if self.cache_dir else None
        if cached and os.path.isfile(cached):
            return open(cached, "rb").read()
        last = None
        for _ in range(4):
            try:
                data = urllib.request.urlopen(self.url_template.format(sub=sub, name=r["coco_name"]), timeout=60).read()
                if cached:
                    with open(cached + ".tmp", "wb") as f:
                        f.write(data)
                    os.replace(cached + ".tmp", cached)
                return data
            except Exception as e:                                               # retry a few times, then fail loudly
                last = e
        raise RuntimeError(f"could not download {r['coco_name']}: {last}")

    def open(self, coco_id):
        from PIL import Image
        im = Image.open(io.BytesIO(self.read_bytes(coco_id)))
        return im.convert("RGB")
