"""Reader of the frozen BamiBERT cache written by scripts/bamibert_extract.py (token states of every question, one pooled vector per candidate string).
Lookup is by the sha1 of the exact text; a text that is not cached raises KeyError (re-run the extractor after a split's text changed)."""
import hashlib
import json
import os

import numpy as np


def key_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


MIN_SD = 0.05                                              # floor of the per-dimension std used by `standardize` (near-constant dims must not be amplified)


class TextCache:
    def __init__(self, path: str, pool: str = "mean", standardize: bool = False):
        if pool not in ("mean", "cls"):
            raise ValueError(f"pool must be mean or cls, got {pool!r}")
        self.path = os.path.abspath(os.path.expanduser(path))
        self.pool = 0 if pool == "mean" else 1
        meta = json.load(open(os.path.join(self.path, "meta.json")))
        self.dim = int(meta["dim"])
        qi, ci = np.load(os.path.join(self.path, "q_index.npz")), np.load(os.path.join(self.path, "c_index.npz"))
        self.q_row = {k: i for i, k in enumerate(qi["key"].tolist())}
        self.q_off, self.q_len = qi["offset"], qi["length"]
        self.c_row = {k: i for i, k in enumerate(ci["key"].tolist())}
        self.c_vec = np.load(os.path.join(self.path, "c_vecs.npy"), mmap_mode="r")
        self._mm = None
        # standardize: subtract the per-dimension mean and divide by the (floored) std of the cache. BERT-style last-layer vectors share a large common component (about 57% of the squared
        # norm of the BamiBERT candidate vectors, within-item cosine 0.59) that hides the differences between candidates after the learned projection.
        self.standardize = standardize
        if standardize:
            c = np.asarray(self.c_vec[:, self.pool], dtype=np.float32)
            self.c_mu, self.c_sd = c.mean(0), np.maximum(c.std(0), MIN_SD)
            n_tok = int((self.q_off + self.q_len).max())
            rows = np.linspace(0, n_tok - 1, num=min(n_tok, 60000)).astype(np.int64)
            q = np.asarray(self._memmap()[rows], dtype=np.float32)
            self.q_mu, self.q_sd = q.mean(0), np.maximum(q.std(0), MIN_SD)

    def __getstate__(self):                                   # DataLoader workers re-open the memmap lazily
        d = dict(self.__dict__); d["_mm"] = None
        return d

    def _memmap(self):
        if self._mm is None:
            self._mm = np.memmap(os.path.join(self.path, "q_states.bin"), dtype="<f2", mode="r").reshape(-1, self.dim)
        return self._mm

    def question(self, text: str) -> np.ndarray:
        """(L, dim) float32 token states of the question (<s> ... </s>)"""
        try:
            i = self.q_row[key_of(text)]
        except KeyError:
            raise KeyError(f"question not in the BamiBERT cache {self.path}: {text!r}") from None
        x = np.asarray(self._memmap()[self.q_off[i]:self.q_off[i] + self.q_len[i]], dtype=np.float32)
        return (x - self.q_mu) / self.q_sd if self.standardize else x

    def candidate(self, text: str) -> np.ndarray:
        """(dim,) float32 pooled vector (mean or cls) of a candidate string"""
        try:
            i = self.c_row[key_of(text)]
        except KeyError:
            raise KeyError(f"candidate not in the BamiBERT cache {self.path}: {text!r}") from None
        x = np.asarray(self.c_vec[i, self.pool], dtype=np.float32)
        return (x - self.c_mu) / self.c_sd if self.standardize else x
