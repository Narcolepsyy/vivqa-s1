"""VLM adapters: trust_remote_code is allowed only for the pinned Vintern revision with the reviewed remote files; InternVL dynamic tiling."""
import os
import sys
import types

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import vlm_adapters as A  # noqa: E402


def ns(**kw):
    return types.SimpleNamespace(**kw)


def test_remote_code_is_refused_for_any_other_model_or_revision():
    with pytest.raises(SystemExit):
        A.make_adapter(ns(trust_remote_code=True, model="Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17"), torch.float16)
    with pytest.raises(SystemExit):
        A.make_adapter(ns(trust_remote_code=False, model=A.VINTERN["repo"], revision="0" * 40), torch.float16)             # the repo name alone selects the remote-code path: other revisions are refused
    with pytest.raises(SystemExit):
        A.require_pinned_vintern(A.VINTERN["repo"], "main")
    A.require_pinned_vintern(A.VINTERN["repo"], A.VINTERN["revision"])


def test_reviewed_hashes_cover_the_five_remote_files():
    assert sorted(A.VINTERN["remote_files_sha256"]) == ["configuration_intern_vit.py", "configuration_internvl_chat.py", "conversation.py", "modeling_intern_vit.py", "modeling_internvl_chat.py"]
    assert all(len(h) == 64 for h in A.VINTERN["remote_files_sha256"].values()) and len(A.VINTERN["revision"]) == 40


def test_verify_remote_code_refuses_a_changed_file(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    fake = tmp_path / "x.py"
    fake.write_text("print('not the reviewed file')")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(fake))
    with pytest.raises(SystemExit):
        A.verify_remote_code(A.VINTERN["repo"], A.VINTERN["revision"])


@pytest.mark.parametrize("size,max_num,n_expected", [((448, 448), 4, 1), ((640, 480), 4, 1), ((1000, 300), 4, 5), ((300, 1000), 6, 7), ((640, 480), 1, 1)])
def test_dynamic_tiling_counts_and_shapes(size, max_num, n_expected):
    tiles = A.dynamic_preprocess(Image.new("RGB", size, (10, 20, 30)), 1, max_num, 448, True)
    assert len(tiles) <= max_num + 1 and all(t.size == (448, 448) for t in tiles)
    if n_expected == 1:
        assert len(tiles) == 1
    else:
        assert len(tiles) >= 2                                                                    # several tiles + the thumbnail
