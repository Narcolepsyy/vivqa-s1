"""Refresh the Kaggle dataset folders of the probe / matrix kernels WITHOUT touching vivqa-siglip2-packed (which may be uploading).
    python scripts/kaggle_refresh.py [--out ~/data/kaggle_upload_v2] [--owner narcolepsyy]
Rebuilds ONLY: viso-code/ (current ViVQAv2-main), vivqa-s1/ (the frozen v2.0 benchmark: train / calib / dev hashes checked against the manifest), bamibert-cache/ (frozen BamiBERT question /
candidate features of ~/data/bamibert_cache). Each gets dataset-metadata.json (id <owner>/<slug>, title 6-50 chars, licence other) and SHA256SUMS.txt (verified). No credential is read,
copied or written; kaggle_prepare.check_file refuses .env files and key-like patterns."""
import argparse
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import kaggle_prepare as K  # noqa: E402

TITLES = {"viso-code": "ViSO code ViVQAv2-main runner", "vivqa-s1": "ViVQA-S1 benchmark v2.0", "vivqa-bamibert-cache": "ViVQA-S1 BamiBERT frozen cache", "vivqa-siglip2-text-cache": "ViVQA-S1 SigLIP2 text-tower cache", "vivqa-xyn-partial": "ViVQA-S1 partial X-YN extra train", "vivqa-teacher-kd": "ViVQA-S1 teacher targets for KD"}
BAMI_FILES = ("q_states.bin", "q_index.npz", "c_vecs.npy", "c_index.npz", "meta.json", "verify.json")
FROZEN = ("train.jsonl", "calib.jsonl", "dev.jsonl")         # test.jsonl is not frozen yet (Sheet A2 / S1-Verified); its hash is only recorded


def write_meta(folder, owner, slug):
    assert 6 <= len(TITLES[slug]) <= 50, TITLES[slug]
    json.dump({"title": TITLES[slug], "id": f"{owner}/{slug}", "licenses": [{"name": "other"}]}, open(os.path.join(folder, "dataset-metadata.json"), "w"), indent=1)


def seal(folder, log):
    n = K.write_sums(folder)
    n_ok, bad = K.verify_sums(folder)
    if bad or n_ok != n:
        raise RuntimeError(f"SHA256SUMS mismatch in {folder}: {bad}")
    log(f"{os.path.basename(folder)}: {n} files, {K.dir_size(folder) / 2 ** 20:.1f} MB, SHA256SUMS verified")
    return {"files": n, "bytes": K.dir_size(folder)}


SLIM_DROP_DIRS = ("data_utils/vncorenlp",)              # the Vietnamese word segmenter (a 143 MB Java jar): never used by ViSO / the probes / the VLM harness
SLIM_MAX_BYTES = 4 << 20                                 # no file above 4 MB is code (METEOR / Stanford jars of evaluation/, old fixtures): the kernels never read them; the 1.7 MB image sha list is kept


def slim_code(root, log=print):
    """keep ONLY code and small files: the upload of the code dataset must take seconds (the planner's upload speed is 2-4 MB/s)"""
    dropped = []
    for rel in SLIM_DROP_DIRS:
        p = os.path.join(root, rel)
        if os.path.isdir(p):
            dropped.append((rel + "/", K.dir_size(p)))
            shutil.rmtree(p)
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            if os.path.getsize(p) > SLIM_MAX_BYTES and not p.endswith("s1_images_sha256.tsv"):
                dropped.append((os.path.relpath(p, root), os.path.getsize(p)))
                os.remove(p)
    gz = os.path.join(root, "kaggle", "s1_images_sha256.tsv.gz")
    if os.path.isfile(gz):                                           # Kaggle auto-decompresses uploaded .gz files: ship the plain .tsv so the checksum list matches a download
        import gzip
        with gzip.open(gz, "rb") as fi, open(gz[:-3], "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(gz)
        log(f"viso-code: kaggle/s1_images_sha256.tsv.gz shipped as plain .tsv ({os.path.getsize(gz[:-3]) / 2 ** 20:.1f} MB)")
    log("slim viso-code: dropped " + ", ".join(f"{n} ({b / 2 ** 20:.1f} MB)" for n, b in dropped))
    return dropped


def refresh(out, owner="narcolepsyy", s1_root=None, code_root=REPO, bami="~/data/bamibert_cache", manifest=None, log=print, siglip_text="~/data/siglip2text_cache", xyn_partial=None, teacher_kd="~/data/teacher_kd_pkg"):
    out = os.path.abspath(os.path.expanduser(out))
    if os.path.commonpath([out, K.PROJECT]) == K.PROJECT:
        raise SystemExit(f"refusing to write the upload folders inside the project: {out}")
    s1_root = os.path.abspath(s1_root or os.path.join(K.PROJECT, "vivqa_s1_v2"))
    pins = json.load(open(manifest or os.path.join(code_root, "kaggle", "experiments_siglip2.json")))["s1_sha256"]
    for f in FROZEN:                                                   # the frozen data must be exactly the pinned one
        got = K.sha256_file(os.path.join(s1_root, "data", f))
        if got != pins[f]:
            raise SystemExit(f"{f} differs from the frozen v2.0 hash: {got} != {pins[f]}")
    res = {}
    # vivqa-s1
    d = os.path.join(out, "vivqa-s1")
    shutil.rmtree(d, ignore_errors=True)
    for f in K.S1_DATA + K.S1_OPTIONAL:
        src = os.path.join(s1_root, "data", f)
        if os.path.isfile(src):
            K.check_file(src, f)
            os.makedirs(os.path.join(d, "data"), exist_ok=True)
            shutil.copy2(src, os.path.join(d, "data", f))
    K.copy_tree(os.path.join(s1_root, "s1bench"), os.path.join(d, "s1bench"), exclude_dirs=K.CODE_EXCLUDE_DIRS, root=s1_root)
    for f in ("evaluate.py", "README.md"):
        K.check_file(os.path.join(s1_root, f), f)
        shutil.copy2(os.path.join(s1_root, f), os.path.join(d, f))
    write_meta(d, owner, "vivqa-s1")
    res["vivqa-s1"] = seal(d, log)
    # viso-code
    d = os.path.join(out, "viso-code")
    shutil.rmtree(d, ignore_errors=True)
    K.copy_tree(code_root, os.path.join(d, "ViVQAv2-main"), exclude_dirs=K.CODE_EXCLUDE_DIRS, exclude_files=K.CODE_EXCLUDE_FILES)
    slim_code(os.path.join(d, "ViVQAv2-main"), log)
    write_meta(d, owner, "viso-code")
    res["viso-code"] = seal(d, log)
    # bamibert-cache
    d = os.path.join(out, "bamibert-cache")
    shutil.rmtree(d, ignore_errors=True)
    bami = os.path.abspath(os.path.expanduser(bami))
    for f in BAMI_FILES:
        if not os.path.isfile(os.path.join(bami, f)):
            raise FileNotFoundError(os.path.join(bami, f))
        K.link_or_copy(os.path.join(bami, f), os.path.join(d, f))
    write_meta(d, owner, "vivqa-bamibert-cache")
    res["bamibert-cache"] = seal(d, log)
    # siglip2-text-cache (T-SIG): the frozen SigLIP 2 text tower features, same layout as the BamiBERT cache
    st = os.path.abspath(os.path.expanduser(siglip_text))
    if os.path.isfile(os.path.join(st, "meta.json")):
        d = os.path.join(out, "siglip2-text-cache")
        shutil.rmtree(d, ignore_errors=True)
        for f in BAMI_FILES:
            if not os.path.isfile(os.path.join(st, f)):
                raise FileNotFoundError(os.path.join(st, f))
            K.link_or_copy(os.path.join(st, f), os.path.join(d, f))
        write_meta(d, owner, "vivqa-siglip2-text-cache")
        res["siglip2-text-cache"] = seal(d, log)
    # xyn-partial: the translated real VQA v2 yes/no questions (extra TRAIN data, separate from the benchmark files)
    xp = os.path.abspath(os.path.expanduser(xyn_partial or os.path.join(K.PROJECT, "vivqa_s1_v2", "extra")))
    if os.path.isfile(os.path.join(xp, "xyn_partial_train.jsonl")):
        d = os.path.join(out, "xyn-partial")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for f in ("xyn_partial_train.jsonl", "extra_partial_manifest.json"):
            K.check_file(os.path.join(xp, f), f)
            shutil.copy2(os.path.join(xp, f), os.path.join(d, f))
        open(os.path.join(d, "README.txt"), "w", encoding="utf-8").write("Extra TRAIN items: real VQA v2 yes/no questions on train-only images of ViVQA-S1 v2.0.1, QUESTION translated to Vietnamese (DeepSeek), soft labels from the 10 human answers. "
                                                                          "Partial run (the first translated questions of the qid-sorted pool). The benchmark files are untouched; meta.extra_vqa = true.\n")
        write_meta(d, owner, "vivqa-xyn-partial")
        res["xyn-partial"] = seal(d, log)
    # teacher-kd: the teacher's distributions on the TRAIN split (knowledge-distillation targets; ids + probability lists, no model name)
    tk = os.path.abspath(os.path.expanduser(teacher_kd))
    if os.path.isfile(os.path.join(tk, "teacher_train.jsonl")):
        d = os.path.join(out, "teacher-kd")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for f in ("teacher_train.jsonl", "manifest.json"):
            K.check_file(os.path.join(tk, f), f)
            shutil.copy2(os.path.join(tk, f), os.path.join(d, f))
        write_meta(d, owner, "vivqa-teacher-kd")
        res["teacher-kd"] = seal(d, log)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/kaggle_upload_v2")
    ap.add_argument("--owner", default="narcolepsyy")
    ap.add_argument("--s1-root", default=None)
    ap.add_argument("--bami", default="~/data/bamibert_cache", help="BamiBERT cache dir (use the one extended with the extra question texts)")
    a = ap.parse_args(argv)
    try:
        print(json.dumps(refresh(a.out, a.owner, a.s1_root, bami=a.bami), indent=1))
    except K.RefusedFile as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
