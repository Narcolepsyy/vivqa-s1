"""Atomic file writes (temp file + os.replace): a reader or an rsync copying a result directory while a job runs never sees a half-written file under its final name.
Temp files are named '<final>.tmp' (and the harness's '<shard>.tmp.npz'): exclude '*.tmp*' when copying a running job's directory."""
import json
import os
import shutil


def _tmp(path):
    return os.fspath(path) + ".tmp"


def atomic_text(path, text, encoding="utf-8"):
    with open(_tmp(path), "w", encoding=encoding, newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(_tmp(path), path)


def atomic_json(path, obj, **kw):
    kw.setdefault("indent", 1)
    atomic_text(path, json.dumps(obj, **kw))


def atomic_savez(path, **arrays):
    import numpy as np
    with open(_tmp(path), "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(_tmp(path), path)


def atomic_save_npy(path, array):
    import numpy as np
    with open(_tmp(path), "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(_tmp(path), path)


def atomic_torch_save(obj, path):
    import torch
    torch.save(obj, _tmp(path))
    os.replace(_tmp(path), path)


def atomic_dir(final, writer):
    """writer(tmp_dir) fills a temp directory which then replaces `final` (the previous one is removed only after the new one is complete)"""
    final = os.fspath(final)
    tmp, old = final + ".tmp", final + ".old"
    for p in (tmp, old):
        shutil.rmtree(p, ignore_errors=True)
    os.makedirs(tmp)
    writer(tmp)
    if os.path.isdir(final):
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)
