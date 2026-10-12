"""Extract ONLY the benchmark's COCO 2014 images from the official zip and verify every one against kaggle/s1_images_sha256.tsv.gz (stdlib only).
  python extract_coco.py --sha-list s1_images_sha256.tsv.gz --zip train2014.zip --sub train2014 --out <dir>/coco
  python extract_coco.py --zip train2014.zip --check          (exit 0 iff the zip's central directory is readable = the download is complete)
Exit codes: 0 ok, 2 sha256 mismatch or a wanted image missing from the zip (ABORT), 3 unreadable zip."""
import argparse
import csv
import gzip
import hashlib
import os
import sys
import zipfile


def open_sha_list(path):
    """.tsv.gz or .tsv (Kaggle auto-decompresses uploaded .gz files); gzip detected by the magic bytes"""
    for p in (path, path[:-3] if path.endswith(".gz") else path + ".gz"):
        if os.path.isfile(p):
            with open(p, "rb") as f:
                gz = f.read(2) == b"\x1f\x8b"
            return gzip.open(p, "rt", encoding="utf-8") if gz else open(p, "rt", encoding="utf-8")
    raise FileNotFoundError(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--sha-list")
    ap.add_argument("--sub", choices=["train2014", "val2014"])
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    try:
        zf = zipfile.ZipFile(a.zip)
    except (zipfile.BadZipFile, OSError) as e:
        print(f"unreadable zip {a.zip}: {e}", file=sys.stderr)
        return 3
    if a.check:
        print(f"{a.zip}: {len(zf.infolist())} entries")
        return 0
    with open_sha_list(a.sha_list) as f:
        want = {r["coco_name"]: r["sha256"] for r in csv.DictReader(f, delimiter="\t") if f"_{a.sub}_" in r["coco_name"]}
    members = {os.path.basename(i.filename): i for i in zf.infolist() if not i.is_dir()}
    missing = [n for n in want if n not in members]
    if missing:
        print(f"ABORT: {len(missing)} wanted images are not in {a.zip} (first: {missing[0]})", file=sys.stderr)
        return 2
    dst_dir = os.path.join(a.out, a.sub)
    os.makedirs(dst_dir, exist_ok=True)
    done = 0
    for n, sha in want.items():
        dst = os.path.join(dst_dir, n)
        if os.path.isfile(dst) and hashlib.sha256(open(dst, "rb").read()).hexdigest() == sha:
            done += 1
            continue
        data = zf.read(members[n])                                      # also checks the zip CRC
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            print(f"ABORT: sha256 mismatch for {n}: got {got[:16]}..., want {sha[:16]}...", file=sys.stderr)
            return 2
        with open(dst + ".tmp", "wb") as f:
            f.write(data)
        os.replace(dst + ".tmp", dst)
        done += 1
        if done % 3000 == 0:
            print(f"  {a.sub}: {done}/{len(want)} verified", flush=True)
    print(f"{a.sub}: {done}/{len(want)} images extracted and sha256-verified in {dst_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
