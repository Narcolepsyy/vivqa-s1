import json
import os
import sys
import warnings

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)                       # configs use paths relative to the repo root
warnings.filterwarnings("ignore")

from yacs.config import CfgNode  # noqa: E402

S1_DIR = "../vivqa_s1_v2/data"
FEATURES = "../vivqav2_dataset/vinvl_vinvl"


def vocab_cfg(**kw):
    d = dict(TYPE="S1Vocab", TOKENIZER=None, MIN_FREQ=1, BOS_TOKEN="<bos>", EOS_TOKEN="<eos>", PAD_TOKEN="<pad>",
             UNK_TOKEN="<unk>", S1_DIR=S1_DIR)
    d.update(kw)
    return CfgNode(init_dict=d)


def feature_cfg(cache=0):
    return CfgNode(init_dict=dict(TYPE="S1FeatureDataset", FEATURE_PATH=dict(FEATURES=FEATURES), FEATURE_CACHE=cache))


@pytest.fixture(scope="session")
def vocab():
    import data_utils.vocabs.s1_vocab  # noqa: F401  (registers S1Vocab)
    from builders.vocab_builder import build_vocab
    return build_vocab(vocab_cfg())


def _subset(tmp_path_factory, name, n):
    """First n dev items of each type, mixed (so every type is covered), written to a tmp jsonl."""
    items = [json.loads(l) for l in open(os.path.join(REPO, S1_DIR, "dev.jsonl"), encoding="utf-8")]
    by = {"noul": [], "score": [], "choice": []}
    for it in items:
        t = it["questions"]["q"]["type"]
        if len(by[t]) < n:
            by[t].append(it)
    sub = by["noul"] + by["score"] + by["choice"]
    p = tmp_path_factory.mktemp("s1") / f"{name}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for it in sub:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return str(p), sub


@pytest.fixture(scope="session")
def subset32(tmp_path_factory):
    return _subset(tmp_path_factory, "sub32", 11)      # 11 + 11 + 10... -> at least 32 items in total


@pytest.fixture(scope="session")
def dataset32(vocab, subset32):
    import data_utils.datasets.s1_dataset as ds
    return ds.S1FeatureDataset(subset32[0], vocab, feature_cfg())


@pytest.fixture
def tmp_path(request):
    """Per-test scratch dir on the big disk (~/data/viso_test_tmp, outside the project), removed after the test:
    runs write 100+ MB checkpoints, which would fill the 8 GB tmpfs at /tmp."""
    import shutil
    import tempfile
    base = os.environ.get("VISO_TEST_TMP") or os.path.expanduser("~/data/viso_test_tmp")
    os.makedirs(base, exist_ok=True)
    d = tempfile.mkdtemp(prefix="t_", dir=base)
    yield __import__("pathlib").Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- legacy VinVL data
# The raw VinVL features (../vivqav2_dataset/vinvl_vinvl/*.npy) and the packed VinVL copy no longer exist on this machine (the paper's legacy setup;
# the configs viso_vivqav2.yaml / viso_flat_vivqav2.yaml are kept unchanged). The tests below read them, so they are SKIPPED (not deleted) when the data is missing:
# they run again wherever the VinVL features are present. Everything SigLIP 2 / synthetic stays active.
VINVL_TESTS = {
    "test_dataset.py": {"test_candidate_tokens_follow_candidate_order", "test_collate_pads_regions_and_masks", "test_feature_cache_returns_same_values", "test_flat_index_maps_into_flat_vocab", "test_gold_idx_points_at_gold_label", "test_sample_fields_and_masks", "test_sizes_and_types", "test_target_sums_to_one_and_is_zero_outside_mask"},
    "test_kaggle_prepare.py": {"test_code_folder_has_what_kaggle_needs_and_nothing_else", "test_harmless_task_dash_text_is_not_a_key", "test_import_works_in_the_unpacked_kaggle_layout", "test_refuses_env_files_and_key_patterns", "test_refuses_to_write_inside_the_project", "test_three_dataset_folders_with_metadata_and_verified_sums"},
    "test_kaggle_runner.py": {"test_preflight_runs_on_cpu_and_writes_the_projection", "test_two_runners_dry_run_end_to_end_with_a3_from_a2_calibration_and_bundle"},
    "test_models.py": {"test_choice_head_is_permutation_equivariant_and_ignores_padding", "test_flat_model_restricted_layout", "test_forward_backward_is_finite", "test_head_layout_and_masking", "test_padding_changes_nothing_even_if_padding_rows_are_garbage", "test_zero_row_padding_does_not_change_h", "test_zero_vision_runs_without_nan"},
    "test_packed_features.py": {"test_bad_feature_format_is_rejected", "test_dataset_packed_equals_npy_shapes_and_masks", "test_missing_id_raises_keyerror_with_the_id", "test_offsets_and_counts_are_consistent", "test_reader_matches_originals_within_fp16_tolerance", "test_repack_is_byte_identical", "test_tags_are_written_for_every_image", "test_workers_can_read_the_packed_file"},
    "test_resume.py": {"test_deadline_exit_code_and_resumable_state", "test_deadline_far_away_does_not_stop", "test_done_json_skip_when_done_and_fresh_run_without_resume", "test_old_default_behaviour_has_no_resume_state_when_resume_is_off", "test_resume_across_the_rl_stage_boundary", "test_resume_equals_straight_run"},
    "test_scripts.py": {"test_predict_script_writes_logits_and_preds"},
    "test_task.py": {"test_logits_are_deterministic_for_the_same_seed", "test_overfit_tiny_subset_soft_ce_loss_halves", "test_predictions_pass_through_s1_evaluate_with_no_missing", "test_rlcd_stage_runs_without_nan_and_logs_reward"},
}


def _vinvl_data_present():
    d = os.path.join(REPO, FEATURES)
    return os.path.isdir(d) and any(n.endswith(".npy") for n in os.listdir(d))


def pytest_collection_modifyitems(config, items):
    if _vinvl_data_present():
        return
    skip = pytest.mark.skip(reason="legacy VinVL features (../vivqav2_dataset/vinvl_vinvl/*.npy) are not on this machine; SigLIP 2 tests cover the new setup")
    for it in items:
        f = os.path.basename(str(it.fspath))
        name = it.originalname or it.name
        if name in VINVL_TESTS.get(f, ()):
            it.add_marker(skip)
