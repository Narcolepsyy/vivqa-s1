import hashlib
import json
import os
import time

import numpy as np
import pytest

import tasks.typed_decision_task as T
from tests.viso.test_task import make_task


def epoch_rows(task):
    """The training log without bookkeeping events: what 'identical dev logs' means."""
    return [r for r in (json.loads(l) for l in open(task.log_file, encoding="utf-8")) if "event" not in r]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def task_in(tmp_path, name="ViSO", loss="soft_ce", epochs=3, **extra):
    extra = {"TRAINING.RESUME": "True", **extra}
    return make_task(tmp_path, name, loss, limit=32, batch=16, epochs=epochs, **extra)


def test_resume_equals_straight_run(tmp_path):
    a = task_in(tmp_path / "straight")
    a.start()
    a.get_predictions()
    rows_a = epoch_rows(a)
    assert len(rows_a) == 3 and os.path.isfile(os.path.join(a.checkpoint_path, "DONE.json"))
    assert not os.path.exists(os.path.join(a.checkpoint_path, "last.pth"))        # deleted when finished

    b = task_in(tmp_path / "resumed", **{"TRAINING.STOP_AFTER_EPOCHS": "1"})
    with pytest.raises(SystemExit) as e:
        b.start()                                                                    # 1 epoch, then "killed"
    assert e.value.code == 3 and os.path.isfile(os.path.join(b.checkpoint_path, "last.pth"))
    assert not os.path.exists(os.path.join(b.checkpoint_path, "DONE.json")) and len(epoch_rows(b)) == 1
    c = task_in(tmp_path / "resumed")                                                # a NEW process / task object
    c.start()
    c.get_predictions()
    assert epoch_rows(c) == rows_a                                                   # identical dev logs
    for split in ("dev", "test", "calib"):
        assert sha(os.path.join(a.output_path, f"logits_{split}.npz")) == sha(os.path.join(c.output_path, f"logits_{split}.npz"))
    done_a = json.load(open(os.path.join(a.checkpoint_path, "DONE.json")))
    done_c = json.load(open(os.path.join(c.checkpoint_path, "DONE.json")))
    assert done_a["best_epoch"] == done_c["best_epoch"] and done_a["best_dev_nll"] == done_c["best_dev_nll"]


def test_resume_across_the_rl_stage_boundary(tmp_path):
    """rlcd 'all': stop after the single stage-1 epoch (stage boundary) and after the first RL epoch."""
    ref = task_in(tmp_path / "ref", loss="rlcd", epochs=1, **{"LOSS.STAGE": "all"})
    ref.start()
    ref.get_predictions()
    out = tmp_path / "cut"
    for k in range(3):                                   # stop after every epoch: 1 (sup) + 2 (rl) = 3 epochs in 3 sessions
        t = task_in(out, loss="rlcd", epochs=1, **{"LOSS.STAGE": "all", "TRAINING.STOP_AFTER_EPOCHS": "1"})
        try:
            t.start()
            break
        except SystemExit as e:
            assert e.code == 3
    else:
        t = task_in(out, loss="rlcd", epochs=1, **{"LOSS.STAGE": "all"})
        t.start()
    t.get_predictions()
    assert epoch_rows(t) == epoch_rows(ref)
    assert sha(os.path.join(ref.output_path, "logits_dev.npz")) == sha(os.path.join(t.output_path, "logits_dev.npz"))


def test_deadline_exit_code_and_resumable_state(tmp_path):
    t = task_in(tmp_path, epochs=4, **{"TRAINING.DEADLINE_UNIX": str(time.time() + 1.0)})   # passes after epoch 1
    with pytest.raises(SystemExit) as e:
        t.start()
    assert e.value.code == 3 and isinstance(e.value, T.IncompleteRun) and e.value.message == "deadline"
    assert os.path.isfile(os.path.join(t.checkpoint_path, "last.pth"))
    assert not os.path.exists(os.path.join(t.checkpoint_path, "DONE.json"))
    # without a deadline the same dirs finish the run
    u = task_in(tmp_path, epochs=4)
    done = u.start()
    assert len(epoch_rows(u)) == 4 and done["stage_results"][0]["epochs_run"] == 4


def test_deadline_far_away_does_not_stop(tmp_path):
    t = task_in(tmp_path, epochs=2, **{"TRAINING.DEADLINE_UNIX": str(time.time() + 3600.0)})
    assert t.start()["stage_results"][0]["epochs_run"] == 2


def test_done_json_skip_when_done_and_fresh_run_without_resume(tmp_path):
    t = task_in(tmp_path, epochs=1)
    d1 = t.start()
    n_rows = len(epoch_rows(t))
    again = task_in(tmp_path, epochs=1)
    d2 = again.start()                                                               # RESUME true + DONE.json: skipped
    assert d2 == d1 and len(epoch_rows(again)) == n_rows
    fresh = make_task(tmp_path, "ViSO", "soft_ce", limit=32, batch=16, epochs=1)    # RESUME false (default): retrains
    fresh.start()
    assert len(epoch_rows(fresh)) == n_rows + 1


def test_old_default_behaviour_has_no_resume_state_when_resume_is_off(tmp_path):
    t = make_task(tmp_path, "ViSO", "soft_ce", limit=16, batch=16, epochs=2)
    t.start()
    assert os.path.isfile(os.path.join(t.checkpoint_path, "DONE.json"))
    assert not os.path.exists(os.path.join(t.checkpoint_path, "last.pth"))
