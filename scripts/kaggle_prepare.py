"""Build the three Kaggle dataset folders (nothing is uploaded; no credentials are read or stored).
    python scripts/kaggle_prepare.py [--out ~/data/kaggle_upload] [--packed ~/data/vivqa_packed]
  vivqa-packed/  regions_fp16.bin index.npz meta.json tags.jsonl verify.json (hard links when on the same filesystem)
  vivqa-s1/      data/{train,calib,dev,test}.jsonl, s1bench/, evaluate.py, README.md          (+ SHA256SUMS.txt)
  viso-code/     ViVQAv2-main/ without output/, saved_models/, analysis_results/, evaluation_results/, plans/, tests/,
                 docs/openVqa.pdf, caches and any .env                                                      (+ SHA256SUMS.txt)
Each folder gets dataset-metadata.json with the placeholder id "<KAGGLE_USERNAME>/<slug>" (edit it before uploading).
Refuses to package a file named .env* or a text file containing a key-like pattern (an OpenAI-style secret token or an
HTTP bearer-authorization header; the patterns are assembled from pieces below so this file does not match itself).
On Kaggle the code finds s1bench through VIVQA_S1_ROOT=/kaggle/input/vivqa-s1 (utils/viso_s1.py honours it)."""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT = os.path.abspath(os.path.join(REPO, ".."))
SLUGS = {"vivqa-packed": "ViVQA-S1 packed VinVL region features (fp16)", "vivqa-s1": "ViVQA-S1 benchmark (frozen)",
         "viso-code": "ViSO code (ViVQAv2-main + Kaggle runner)"}
PACKED_FILES = ("regions_fp16.bin", "index.npz", "meta.json", "tags.jsonl", "verify.json")
# Phase 2c: the SigLIP 2 cache (scripts/siglip2_extract.py) has shared boxes instead of tags; the dataset is called vivqa-siglip2-packed
SIGLIP2_SLUG, SIGLIP2_TITLE = "vivqa-siglip2-packed", "ViVQA-S1 frozen SigLIP 2 B/16 256 px cached features (65 x 768 per image, fp16)"
SIGLIP2_FILES = ("regions_fp16.bin", "index.npz", "boxes.npy", "meta.json", "verify.json", "vision_tower_efficiency.json")
S1_DATA = ("train.jsonl", "calib.jsonl", "dev.jsonl", "test.jsonl")
S1_OPTIONAL = ("eval_flags.jsonl",)          # the Core / Image-unseen views of v1.1+ need it next to test.jsonl


def packed_kind(packed):
    """'siglip2' when the packed directory was written by siglip2_extract.py (meta.json model_id), else 'vinvl'"""
    try:
        with open(os.path.join(packed, "meta.json"), encoding="utf-8") as f:
            return "siglip2" if str(json.load(f).get("model_id", "")).startswith("google/siglip2") else "vinvl"
    except (OSError, ValueError):
        return "vinvl"
CODE_EXCLUDE_DIRS = {"output", "saved_models", "analysis_results", "evaluation_results", "plans", "tests", "__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints"}
CODE_EXCLUDE_FILES = {"docs/openVqa.pdf", "test_out.txt"}
TEXT_EXT = (".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".sh", ".cfg", ".ini", ".toml", ".csv", ".tsv", ".ipynb", ".rst")
_SK, _BEARER = b"s" + b"k-", b"Bear" + b"er "           # assembled so that this very file is not flagged
KEY_PATTERNS = [re.compile(rb"(?<![A-Za-z0-9])" + re.escape(_SK) + rb"[A-Za-z0-9_\-]{8,}"), re.compile(re.escape(_BEARER))]


class RefusedFile(Exception):
    pass


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def check_file(path, rel):
    """Raise RefusedFile for a .env file or a text file that looks like it holds a key."""
    name = os.path.basename(path)
    if name == ".env" or name.startswith(".env."):
        raise RefusedFile(f"refusing to package an environment file: {rel}")
    if name.lower().endswith(TEXT_EXT):
        with open(path, "rb") as f:
            data = f.read()
        for pat in KEY_PATTERNS:
            if pat.search(data):
                raise RefusedFile(f"refusing to package {rel}: it contains a key-like pattern ({pat.pattern.decode()})")


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tree(src, dst, exclude_dirs=(), exclude_files=(), root=None):
    root = root or src
    n = 0
    for d, dirs, files in os.walk(src):
        dirs[:] = sorted(x for x in dirs if x not in exclude_dirs)
        for f in sorted(files):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root)
            if rel in exclude_files or f.endswith((".pyc", ".pyo")):
                continue
            check_file(p, rel)
            target = os.path.join(dst, os.path.relpath(p, src))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(p, target)          # real copies for code / benchmark (only the 5 GB pack is hard-linked)
            n += 1
    return n


def write_sums(folder):
    lines = []
    for d, dirs, files in os.walk(folder):
        dirs.sort()
        for f in sorted(files):
            if f in ("SHA256SUMS.txt", "dataset-metadata.json"):          # Kaggle strips dataset-metadata.json from downloads: the list must hold what a download contains
                continue
            p = os.path.join(d, f)
            lines.append(f"{sha256_file(p)}  {os.path.relpath(p, folder)}")
    with open(os.path.join(folder, "SHA256SUMS.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def verify_sums(folder):
    bad, n = [], 0
    for line in open(os.path.join(folder, "SHA256SUMS.txt")):
        h, rel = line.rstrip("\n").split("  ", 1)
        n += 1
        if sha256_file(os.path.join(folder, rel)) != h:
            bad.append(rel)
    return n, bad


def write_metadata(folder, slug, title):
    meta = {"title": title, "id": f"<KAGGLE_USERNAME>/{slug}", "licenses": [{"name": "other"}]}
    with open(os.path.join(folder, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)


def dir_size(folder):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(folder) for f in fs)


def build(out, packed, s1_root=None, code_root=REPO, log=print):
    out = os.path.abspath(os.path.expanduser(out))
    packed = os.path.abspath(os.path.expanduser(packed))
    s1_root = os.path.abspath(s1_root or os.path.join(PROJECT, "vivqa_s1_v2"))
    if os.path.commonpath([out, PROJECT]) == PROJECT:
        raise SystemExit(f"refusing to write the upload folders inside the project: {out}")
    shutil.rmtree(out, ignore_errors=True)
    result = {}
    # 1. packed features
    d = os.path.join(out, "vivqa-packed")
    kind = packed_kind(packed)
    slugs = dict(SLUGS)
    if kind == "siglip2":
        slugs.pop("vivqa-packed")
        slugs = {SIGLIP2_SLUG: SIGLIP2_TITLE, **slugs}
        d = os.path.join(out, SIGLIP2_SLUG)
    for f in (SIGLIP2_FILES if kind == "siglip2" else PACKED_FILES):
        src = os.path.join(packed, f)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"missing {src}")
        link_or_copy(src, os.path.join(d, f))
    # 2. benchmark
    d2 = os.path.join(out, "vivqa-s1")
    for f in S1_DATA:
        check_file(os.path.join(s1_root, "data", f), f)
        os.makedirs(os.path.join(d2, "data"), exist_ok=True)
        shutil.copy2(os.path.join(s1_root, "data", f), os.path.join(d2, "data", f))
    for f in S1_OPTIONAL:
        if os.path.isfile(os.path.join(s1_root, "data", f)):
            check_file(os.path.join(s1_root, "data", f), f)
            shutil.copy2(os.path.join(s1_root, "data", f), os.path.join(d2, "data", f))
    copy_tree(os.path.join(s1_root, "s1bench"), os.path.join(d2, "s1bench"), exclude_dirs=CODE_EXCLUDE_DIRS, root=s1_root)
    for f in ("evaluate.py", "README.md"):
        check_file(os.path.join(s1_root, f), f)
        shutil.copy2(os.path.join(s1_root, f), os.path.join(d2, f))
    # 3. code
    d3 = os.path.join(out, "viso-code", "ViVQAv2-main")
    copy_tree(code_root, d3, exclude_dirs=CODE_EXCLUDE_DIRS, exclude_files=CODE_EXCLUDE_FILES)
    for slug, title in slugs.items():
        folder = os.path.join(out, slug)
        write_metadata(folder, slug, title)
        n = write_sums(folder)
        n_ok, bad = verify_sums(folder)
        result[slug] = {"files": n, "bytes": dir_size(folder), "sha256_ok": not bad and n_ok == n}
        if bad:
            raise RuntimeError(f"SHA256SUMS mismatch in {slug}: {bad}")
        log(f"{slug}: {n} files, {dir_size(folder) / 2 ** 20:.1f} MB, SHA256SUMS verified ({n_ok} entries)")
    readme = "README_KAGGLE_SIGLIP2.md" if kind == "siglip2" else "README_KAGGLE.md"
    shutil.copyfile(os.path.join(code_root, "kaggle", readme), os.path.join(out, "README_KAGGLE.md"))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/kaggle_upload")
    ap.add_argument("--packed", default="~/data/vivqa_packed", help="packed feature dir: the VinVL cache, or ~/data/siglip2_packed (detected from meta.json)")
    ap.add_argument("--s1-root", default=None)
    a = ap.parse_args(argv)
    try:
        res = build(a.out, a.packed, a.s1_root)
    except RefusedFile as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
