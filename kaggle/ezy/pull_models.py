"""Pre-download the pinned Hugging Face models (the MODELS table of vlm_kernel.py: full 40-char revisions) into $HF_HOME.
  python pull_models.py <kernel.py> [key ...]      download (all models when no key is given)
  python pull_models.py <kernel.py> --check [key ...]   only ask the Hub whether each pinned revision exists (no download)"""
import ast
import sys


def models_table(path):
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "MODELS":
            return ast.literal_eval(node.value)
    raise SystemExit(f"no MODELS table in {path}")


def main():
    path, rest = sys.argv[1], sys.argv[2:]
    if rest and rest[0] == "--print":                                  # "<repo> <revision> <licence>" of one key (licence without spaces)
        repo, rev, lic = models_table(path)[rest[1]][:3]
        print(repo, rev, lic.split()[0])
        return
    from huggingface_hub import HfApi, snapshot_download
    check = "--check" in rest
    keys = [k for k in rest if k != "--check"]
    table = models_table(path)
    keys = keys or list(table)
    bad = [k for k in keys if k not in table]
    if bad:
        raise SystemExit(f"unknown model keys {bad}; known: {list(table)}")
    for k in keys:
        repo, rev, lic = table[k][:3]
        assert len(rev) == 40, f"{k}: revision must be the full sha, got {rev}"
        if check:
            info = HfApi().model_info(repo, revision=rev)
            assert info.sha == rev, (k, info.sha, rev)
            print(f"ok   {k:14s} {repo}@{rev[:8]}  ({lic})", flush=True)
            continue
        print(f"pull {k:14s} {repo}@{rev[:8]} ...", flush=True)
        p = snapshot_download(repo, revision=rev, ignore_patterns=["*.gguf", "*.onnx", "original/*", "*.msgpack", "*.h5"])
        print(f"     -> {p}", flush=True)


if __name__ == "__main__":
    main()
