import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import kaggle_prepare as K  # noqa: E402
import viso_pack_features as P  # noqa: E402

from tests.viso.conftest import FEATURES, S1_DIR  # noqa: E402

REPO = K.REPO
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
BASE = os.path.expanduser("~/data/viso_test_tmp")


@pytest.fixture(scope="module")
def upload():
    os.makedirs(BASE, exist_ok=True)
    root = tempfile.mkdtemp(prefix="k_", dir=BASE)
    packed = os.path.join(root, "pack")
    P.pack(FEATURES, S1_DIR, packed, limit=10, log=lambda *_: None)
    open(os.path.join(packed, "verify.json"), "w").write("{}")
    out = os.path.join(root, "upload")
    res = K.build(out, packed, log=lambda *_: None)
    yield out, res, packed
    shutil.rmtree(root, ignore_errors=True)


def test_three_dataset_folders_with_metadata_and_verified_sums(upload):
    out, res, _ = upload
    assert set(res) == {"vivqa-packed", "vivqa-s1", "viso-code"} and all(r["sha256_ok"] for r in res.values())
    for slug in res:
        meta = json.load(open(os.path.join(out, slug, "dataset-metadata.json")))
        assert meta["id"] == f"<KAGGLE_USERNAME>/{slug}" and meta["title"] and meta["licenses"]
        n, bad = K.verify_sums(os.path.join(out, slug))
        assert n == res[slug]["files"] and bad == []
    assert os.path.isfile(os.path.join(out, "README_KAGGLE.md"))
    assert sorted(os.listdir(os.path.join(out, "vivqa-packed"))) == sorted(K.PACKED_FILES + ("SHA256SUMS.txt", "dataset-metadata.json"))
    assert os.path.isfile(os.path.join(out, "vivqa-s1", "data", "test.jsonl")) and os.path.isdir(os.path.join(out, "vivqa-s1", "s1bench"))


def test_code_folder_has_what_kaggle_needs_and_nothing_else(upload):
    out, _, _ = upload
    code = os.path.join(out, "viso-code", "ViVQAv2-main")
    for f in ("train_viso.py", "scripts/viso_kaggle_run.py", "scripts/viso_bootstrap.py", "scripts/viso_report.py", "scripts/viso_calibrate.py",
              "kaggle/experiments.json", "kaggle/README_KAGGLE.md", "tasks/typed_decision_task.py", "models/viso/viso_model.py",
              "configs/viso_vivqav2.yaml", "utils/viso_s1.py", "data_utils/datasets/s1_dataset.py"):
        assert os.path.isfile(os.path.join(code, f)), f
    names = [os.path.relpath(os.path.join(d, f), code) for d, _, fs in os.walk(code) for f in fs]
    assert not [n for n in names if n.split(os.sep)[0] in ("output", "saved_models", "analysis_results", "evaluation_results", "plans", "tests")]
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc") or os.path.basename(n).startswith(".env")]
    assert "docs/openVqa.pdf" not in names
    assert K.dir_size(os.path.join(out, "viso-code")) < 400 * 2 ** 20


def test_import_works_in_the_unpacked_kaggle_layout(upload):
    out, _, _ = upload
    code = os.path.join(out, "viso-code", "ViVQAv2-main")
    s1 = os.path.join(out, "vivqa-s1")
    snippet = ("import utils.viso_s1 as s, s1bench, os, warnings; warnings.filterwarnings('ignore'); "
               "from s1bench.metrics import evaluate; "
               "assert os.path.realpath(s.S1_ROOT) == os.path.realpath(os.environ['VIVQA_S1_ROOT']), s.S1_ROOT; "
               "import data_utils.vocabs.s1_vocab, models.viso; from train_viso import load_config; "
               "from builders.vocab_builder import build_vocab; "
               "cfg = load_config('configs/viso_vivqav2.yaml', ['DATASET.VOCAB.S1_DIR', os.environ['VIVQA_S1_ROOT'] + '/data']); "
               "v = build_vocab(cfg.DATASET.VOCAB); print('vocab', len(v), v.total_answers)")
    env = {**ENV, "VIVQA_S1_ROOT": s1}
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-c", snippet], cwd=code, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "vocab 3084 3074" in r.stdout            # same vocabulary as the original layout
    # the packaged runner accepts the packaged benchmark (checksums in the manifest)
    r2 = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'scripts'); import viso_kaggle_run as R, os; "
                         "R.check_s1(os.environ['VIVQA_S1_ROOT'], R.load_manifest('kaggle/experiments.json')); print('s1 ok')"],
                        cwd=code, env=env, capture_output=True, text=True)
    assert r2.returncode == 0 and "s1 ok" in r2.stdout, r2.stderr[-1500:]


def make_code_root(tmp_path, extra_name, content):
    root = tmp_path / "code"
    (root / "kaggle").mkdir(parents=True)
    (root / "kaggle" / "README_KAGGLE.md").write_text("x")
    (root / "a.py").write_text("print('hi')\n")
    p = root / extra_name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return str(root)


@pytest.mark.parametrize("name,content", [(".env", "KEY=1\n"), ("cfg.json", '{"k": "sk-abcdefghijklmnop123"}'),
                                          ("net.py", "h = {'Authorization': 'Bearer abc'}\n")])
def test_refuses_env_files_and_key_patterns(tmp_path, upload, name, content):
    _, _, packed = upload
    root = make_code_root(tmp_path, name, content)
    with pytest.raises(K.RefusedFile):
        K.build(str(tmp_path / "up"), packed, code_root=root, log=lambda *_: None)


def test_harmless_task_dash_text_is_not_a_key(tmp_path, upload):
    _, _, packed = upload
    root = make_code_root(tmp_path, "notes.md", "the mask-based task-specific disk-cache risk-score\n")
    K.build(str(tmp_path / "up"), packed, code_root=root, log=lambda *_: None)


def test_refuses_to_write_inside_the_project(upload):
    _, _, packed = upload
    with pytest.raises(SystemExit, match="refusing"):
        K.build(os.path.join(K.PROJECT, "kaggle_upload_inside"), packed, log=lambda *_: None)
