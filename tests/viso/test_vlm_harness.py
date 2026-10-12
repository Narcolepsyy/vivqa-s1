"""The VLM System One harness (scripts/s1_vlm_eval.py) without a model: prompts, option orders, letter-logit -> label-space mapping, head-only training on synthetic hidden states, and the verified image store."""
import gzip
import hashlib
import io
import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import s1_vlm_eval as V  # noqa: E402

from utils.coco_images import ImageStore  # noqa: E402
from utils.viso_s1 import label_space  # noqa: E402

S1_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "vivqa_s1_v2", "data"))
needs_s1 = pytest.mark.skipif(not os.path.isfile(os.path.join(S1_DATA, "dev.jsonl")), reason="needs ../vivqa_s1_v2/data")


def jpeg(color, size=(40, 30)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "JPEG")
    return b.getvalue()


def write_sha_list(path, files):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("coco_id\tour_name\tcoco_name\tbytes\tsha256\n")
        for cid, (ours, coco, data) in files.items():
            f.write(f"{cid}\t{ours}\t{coco}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}\n")


def test_image_store_finds_by_coco_name_anywhere_verifies_and_fails_loudly(tmp_path):
    a, b = jpeg("red"), jpeg("blue")
    files = {7: ("000007.jpg", "COCO_train2014_000000000007.jpg", a), 8: ("000008.jpg", "COCO_val2014_000000000008.jpg", b)}
    root = tmp_path / "input" / "datasets" / "someone" / "coco" / "train2014"
    root.mkdir(parents=True)
    (root / files[7][1]).write_bytes(a)                                       # image 7: under its COCO name, nested deep; image 8: not mounted at all
    write_sha_list(tmp_path / "sha.tsv.gz", files)
    st = ImageStore(str(tmp_path / "sha.tsv.gz"), roots=[str(tmp_path / "input")])
    assert len(st) == 2 and st.open(7).size == (40, 30)
    with pytest.raises(FileNotFoundError, match="not found"):
        st.open(8)
    (root / files[7][1]).write_bytes(jpeg("green"))                           # a different file under the right name -> sha mismatch
    with pytest.raises(RuntimeError, match="does not match"):
        ImageStore(str(tmp_path / "sha.tsv.gz"), roots=[str(tmp_path / "input")]).open(7)
    # download fallback (a file:// URL template stands in for images.cocodataset.org); the downloaded bytes are verified too
    web = tmp_path / "web" / "val2014"
    web.mkdir(parents=True)
    (web / files[8][1]).write_bytes(b)
    st2 = ImageStore(str(tmp_path / "sha.tsv.gz"), roots=[], cache_dir=str(tmp_path / "cache"), allow_download=True, url_template=(tmp_path / "web").as_uri() + "/{sub}/{name}")
    assert st2.open(8).size == (40, 30) and (tmp_path / "cache" / files[8][1]).is_file()
    (web / files[8][1]).write_bytes(jpeg("black"))
    (tmp_path / "cache" / files[8][1]).unlink()
    with pytest.raises(RuntimeError, match="does not match"):
        st2.open(8)


@needs_s1
def test_prompts_orders_and_letter_mapping():
    items = V.load_items(S1_DATA, "dev")
    by = {}
    for it in items:
        by.setdefault(it["questions"]["q"]["type"], it)
    yn, sc, ch = by["noul"], by["score"], by["choice"]
    p = V.build_prompt(yn, 1)
    assert "A. không" in p and "B. có" in p and yn["questions"]["q"]["instructions"] in p
    assert "K. 10 trở lên" in V.build_prompt(sc, 1) and "A. 0" in V.build_prompt(sc, 1)
    n = len(label_space(ch["questions"]["q"]))
    p1, p2 = V.build_prompt(ch, 1), V.build_prompt(ch, 2)
    texts = V.option_texts(ch["questions"]["q"])
    assert f"A. {texts[0]}" in p1 and f"A. {texts[-1]}" in p2 and f"{V.LETTERS[n - 1]}. {texts[0]}" in p2
    assert V.make_units([yn, ch], "train") == [(0, 1), (1, 1)] and V.make_units([yn, ch], "dev") == [(0, 1), (1, 1), (1, 2)]
    # the model "answers" the gold candidate in BOTH orders: letters are mapped back to label space and averaged
    gold = label_space(ch["questions"]["q"]).index(ch["gold_label"])
    ll = {}
    for o in (1, 2):
        perm = V.order_perm(n, o)
        z = np.full(16, -5.0)
        z[perm.index(gold)] = 5.0
        ll[o] = z
    z = V.to_label_logits(ch, ll)
    assert int(np.argmax(z[:n])) == gold and np.all(z[n:] == 0)


@needs_s1
def test_head_only_starts_at_zero_shot_and_trains(tmp_path):
    items = V.load_items(S1_DATA, "train", 300)
    dev = V.load_items(S1_DATA, "dev", 150)
    d = 12
    rng = np.random.default_rng(0)
    W0 = rng.standard_normal((16, d)).astype(np.float32)
    mdir = tmp_path / "m"
    (mdir / "shards").mkdir(parents=True)
    np.save(mdir / "letter_rows.npy", W0)

    def shard(split, its):
        ids, orders, hs, lls = [], [], [], []
        for it in its:
            for o in (1, 2) if (split != "train" and it["questions"]["q"]["type"] == "choice") else (1,):
                gold_letter = V.order_perm(len(label_space(it["questions"]["q"])), o).index(label_space(it["questions"]["q"]).index(it["gold_label"]))
                h = W0[gold_letter] * 1.0 + 0.3 * rng.standard_normal(d).astype(np.float32)           # a hidden state that points at the gold letter row
                ids.append(it["id"]); orders.append(o); hs.append(h.astype(np.float16)); lls.append(h @ W0.T)
        np.savez(mdir / "shards" / f"{split}_00000.npz", ids=np.array(ids), order=np.array(orders), h=np.stack(hs), ll=np.stack(lls))
    shard("train", items)
    shard("dev", dev)
    a = type("A", (), dict(model_dir=str(mdir), s1_data=S1_DATA, limit=300, head_train_limit=0, subset_seed=0, lr=1e-2, wd=0.0, dropout=0.0, epochs=4, patience=3, batch_head=64))()
    a.limit = 0
    # the items are the first 300 / 150 of the files: limit them like the shards
    import s1_vlm_eval as VV
    orig = VV.load_items
    VV.load_items = lambda s1, sp, limit=0: orig(s1, sp, 300 if sp == "train" else 150)
    try:
        (W, b), hist = VV.train_head(a, 0)
    finally:
        VV.load_items = orig
    assert hist[0]["epoch"] == 0 and len(hist) >= 2 and W.shape == (16, d)
    assert min(h["dev_nll"] for h in hist) <= hist[0]["dev_nll"]                                         # epoch 0 = zero-shot; training never ends worse than it (selection by dev NLL)


def test_image_store_reads_the_sha_list_as_tsv_gz_or_plain_tsv(tmp_path):
    """Kaggle auto-decompresses uploaded *.gz: a mounted dataset has the plain .tsv although the code asks for .tsv.gz"""
    a = jpeg("green")
    files = {5: ("000005.jpg", "COCO_train2014_000000000005.jpg", a)}
    root = tmp_path / "in"
    root.mkdir()
    (root / files[5][1]).write_bytes(a)
    write_sha_list(tmp_path / "s.tsv.gz", files)
    (tmp_path / "plain.tsv").write_bytes(gzip.decompress((tmp_path / "s.tsv.gz").read_bytes()))
    for given in (tmp_path / "s.tsv.gz", tmp_path / "plain.tsv", tmp_path / "plain.tsv.gz"):            # the last one: asked for .gz, only the plain file exists
        assert ImageStore(str(given), roots=[str(root)]).open(5).size == (40, 30)
    (tmp_path / "s.tsv.gz").rename(tmp_path / "other.tsv")                                                # a gzip file under a plain name: detected by the magic bytes
    assert ImageStore(str(tmp_path / "other.tsv"), roots=[str(root)]).open(5).size == (40, 30)
    with pytest.raises(FileNotFoundError):
        ImageStore(str(tmp_path / "missing.tsv.gz"), roots=[str(root)])
