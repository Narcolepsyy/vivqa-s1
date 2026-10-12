# Phase 2b plan: run the ViSO experiment matrix on Kaggle (planner: ie307-final-ec; implementor: ie307-final-7f)

Decision (user): full training runs happen on Kaggle. Phase 2a (code) and step 0 (packed features) are accepted.
Phase 2b has two parts: **2b-prep** (this plan: make everything Kaggle-ready, test it locally on CPU, package it) and
**2b-run** (the user runs the notebook on Kaggle; the planner reviews the downloaded results). The implementor cannot run Kaggle.

## 0. Kaggle facts the design must respect (checked 2026-10-08; re-check in the notebook preflight)
- Free GPU: T4 (x2 option) or P100, 16 GB VRAM each, 30 GPU-hours/week (wall clock), 9-hour session limit, 4 CPU cores,
  ~29 GB RAM (T4 x2 image), `/kaggle/working` <= 20 GB, `/kaggle/input` datasets read-only.
- `/kaggle/working` is lost when the session ends unless the notebook version is saved; a later session can read an
  earlier version's output only if the user attaches it as an input dataset.
- Internet may be off by default. Plan for `pip install` with Internet ON (needs the user's phone verification), and
  do not rely on it for data.
- With the "T4 x2" accelerator the quota counts wall-clock time, so running TWO experiments in parallel (one per GPU) is
  effectively double throughput. Use one process per GPU; no multi-GPU training of a single run.

## 1. Hard rules
- Pre-existing repo files, `vivqa_s1/`, `vivqav2_dataset/`: READ-ONLY. NEW files only, except your own Phase 2a / step-0
  files (`tasks/typed_decision_task.py`, `data_utils/datasets/s1_dataset.py`, `utils/viso_*.py`, configs, docs, tests), which you may
  extend. Defaults must keep the 57 existing tests and the tiny runs unchanged.
- No Kaggle API calls, no network, no API keys, do not read `.env`. The user uploads the datasets and runs the notebook.
- No AMP / fp16 in this phase. (`-1e9` masks and the encoder's `-1e5` additive mask overflow in fp16 and CPU tests cannot
  catch it.) T4 fp32 (~8 TFLOPS) is enough: expected epoch ~1-2 min with packed features. Revisit only if the preflight says epochs > 3 min.
- Deterministic where possible: seeds as before; set `torch.backends.cudnn.deterministic = True` and `benchmark = False`; GPU runs are
  not bit-exact and the docs must say so (the CPU determinism test stays the reference).
- Never write anything inside the project folder except source files; large outputs of local tests go to `/tmp` or a scratch dir.

## 2. What to build (all new unless noted)
```
scripts/viso_kaggle_run.py        runner: job queue, one process per GPU, resume, deadline, manifest, post-processing
kaggle/experiments.json           the experiment manifest (section 3)
kaggle/README_KAGGLE.md           click-by-click steps for the user + the notebook cells (section 6)
scripts/kaggle_prepare.py         builds ~/data/kaggle_upload/ with the three dataset folders (section 5)
scripts/viso_bootstrap.py         paired bootstrap + seed aggregation (section 4)
tests/viso/test_kaggle_runner.py, test_bootstrap.py, test_resume.py
```
Own-file edits: `tasks/typed_decision_task.py` (resume + deadline + device index + cudnn flags), `docs/viso_README.md`.

### 2.1 Resume and deadline (typed_decision_task.py)
- After EVERY epoch save `last.pth` = model, optimiser, scheduler, epoch, stage, best_dev_nll, bad_epochs, global_step, CPU RNG
  states (`torch`, `numpy`, `random`, train DataLoader generator, `rl_gen`). On start, if `last.pth` exists and `RESUME: true` (default
  false in the old configs, true in the Kaggle ones), continue exactly where it stopped (same stage, next epoch). A run that
  finished writes `DONE.json` (stage results, best epoch, dev NLL, wall time) and deletes `last.pth` to save space.
- `TRAINING.DEADLINE_UNIX` (optional): after an epoch, if now + (mean epoch time * 1.2) exceeds the deadline, save `last.pth`
  and exit with code 3 ("incomplete, resume me"). The runner turns that into a clean stop.
- Tests (CPU, tiny): train 3 epochs straight vs 1 epoch + kill + resume 2 epochs: identical dev logs and identical final logits sha256.
  Also a test for the deadline exit code and for `DONE.json` / skip-when-done.

### 2.2 Runner (scripts/viso_kaggle_run.py)
Arguments: `--experiments kaggle/experiments.json --out /kaggle/working/viso_out --code-root --s1-dir --packed --gpu {0,1}
--deadline-hours 8.5 --prior-output <dir with earlier results> --only <ids> --preflight --dry-run`.
- Builds each run's YAML by loading the base config (`configs/viso_vivqav2.yaml` or `viso_flat_vivqav2.yaml`) and applying the job's overrides
  (yacs types: floats must be floats), with Kaggle paths, `FEATURE_FORMAT: packed`, `DEVICE: cuda`, `WORKERS: 2`, `BATCH_SIZE: 64`,
  `RESUME: true`, `CHECKPOINT_PATH`/`OUTPUT_PATH` under `--out`. Writes the YAML into the run dir.
- Queue: jobs are claimed by creating `<out>/.locks/<id>` atomically (`os.mkdir`); two runner processes (gpu 0 and 1) share it. A job runs only
  when its `depends_on` have `DONE.json`. Dependencies on stage-1 checkpoints: for A3/A3c, copy A2's `best_model.pth` (same seed) to
  `<A3 dir>/best_model_stage1.pth` and the vocab, then run `LOSS.STAGE rl`. This is valid because A3's stage 1 is exactly A2's run.
- `--prior-output`: before starting, copy completed run dirs (DONE.json + best_model.pth + logits) from an earlier session's output.
- `--preflight`: 200 training steps of A2 and A0 (batch 64, real packed features) on the chosen GPU; prints and saves `preflight.json`
  (GPU name, torch/CUDA versions, seconds/step, peak memory, projected seconds/epoch and projected total matrix hours, data-loader wait share).
- `env_info.json` per session: python, torch, cuda, GPU names, git-less code hash (sha256 of all source files), dataset checksums
  (from `meta.json` of the packed pack and a sha256 of the four S1 jsonl files). Abort early if the S1 files' sha256 differ from the ones in
  `experiments.json` (stops silent use of the wrong benchmark version).
- After each completed training job the runner writes logits (existing `get_predictions`), then runs the CPU post-processing for that run
  (`viso_calibrate.py` for A0 -> A0T and A3 -> A4, plain predictions elsewhere). Failures are logged and do not stop other jobs.
- At the end (or on `--finalize`): `viso_report.py` + `viso_bootstrap.py` + `results_bundle.zip` (section 4).
- Exit codes: 0 all done, 3 stopped by deadline with resumable state, 1 error. All logs go to `<out>/runner_<gpu>.log` (no secrets exist here, but never print env).
- Tests (CPU): `--dry-run` with a 2-job tiny manifest and the local tiny paths executes end to end (device cpu, LIMIT 64); lock claiming under two
  simultaneous processes never runs a job twice; dependency order; skip-when-done; refusal when the S1 checksum mismatches.

## 3. Experiment manifest (kaggle/experiments.json)
IDs, models, objectives as in the matrix (A0, A0T, A1, A2, A3, A3c, A4, A5, A6). Concretely:
| job | model / config | seeds | notes |
|---|---|---|---|
| A0_s | FlatS1, `soft_ce` | 0,1,2 | A0T = per-type temperature on A0 (post-processing) |
| A1 | ViSO `hard_ce` | 0 | |
| A2_s | ViSO `soft_ce` | 0,1,2 | stage-1 donor for A3 and A3c |
| A3_s | A2_s checkpoint + `rlcd` stage 2 (5 epochs, PG_WEIGHT 1.0) | 0,1,2 | `LOSS.STAGE rl` |
| A3c_s | A2_s checkpoint + stage 2 with `LOSS.PG_WEIGHT 0.0` (float) | 0,1,2 | control: same extra epochs, no RLCD term |
| A4_s | post-processing of A3_s: per-type temperature + escalation head | 0,1,2 | no training |
| A5 | ViSO `ZERO_VISION` true, `soft_ce` | 0 | |
| A6 | ViSO `SET_ATTENTION` false, `soft_ce` | 0 | |
Hyper-parameters = the plan defaults (AdamW, encoder lr 2e-4, heads lr 1e-3, cosine, 500 warmup, batch 64, up to 25 sup epochs, patience 5,
dev-NLL model selection; RLCD K=4, sigma 0.4->0.1, spherical 0.75, RPS 1.0). Jobs: 9 supervised + 6 RL-stage runs. Order by priority so a partial session is still
useful: A2_0, A0_0, A3_0, A3c_0, A1, A5, A6, then seeds 1 and 2. The manifest carries an `est_minutes` per job (filled from the preflight when available).

## 4. Aggregation, statistics, results bundle
- `viso_bootstrap.py`: from the saved logits and the gold in `vivqa_s1/data/test.jsonl`. For every comparison pair (A3 vs A2, A3 vs A3c, A2 vs A0, A4 vs A3, A0T vs A0, A5 vs A2, A6 vs A2) and every seed: paired bootstrap over test items (1,000 resamples, seed 0) of the
  difference in primary accuracy, NLL, ECE, AURC; 95% percentile CI; also the S1-Human metrics. Seed aggregation: mean +- std over seeds of each metric, and of
  each paired difference. Must reuse `s1bench.metrics` item-level scores (no re-implementation). Deterministic; unit-tested on a synthetic case with a known difference.
- `viso_report.py` (existing): extend (own file) to write `results.md` with tables: overall, per type, S1-Human, calibration (per-type T, ECE before/after), escalation
  (AUROC and AURC vs max-prob, selective accuracy at 0.5/0.7/0.9), efficiency (params, FLOPs; latency is NOT measured on Kaggle; it will be measured locally on the user's CPU from a downloaded checkpoint).
- `results_bundle.zip` (< 500 MB, NO checkpoints): per-run `train_log.jsonl`, `config.yaml`, `DONE.json`, `logits_*.npz`, `preds_*.jsonl`, calibration/escalation jsons,
  `preflight.json`, `env_info.json`, `results.md`, `bootstrap.json`, runner logs. Checkpoints (`best_model.pth`, ~160 MB each, 15 runs ~ 2.4 GB) stay in `/kaggle/working/viso_out`
  (<= 20 GB total) and are kept by saving the notebook version.

## 5. Upload packaging (scripts/kaggle_prepare.py)
Creates `~/data/kaggle_upload/` (outside the project) with three dataset folders, each with a `dataset-metadata.json` whose `id` is `<KAGGLE_USERNAME>/<slug>` (a placeholder the
user edits; do not ask for or store credentials):
1. `vivqa-packed` (5.0 GB): `regions_fp16.bin`, `index.npz`, `meta.json`, `tags.jsonl`, `verify.json` (hard links if on the same filesystem, else copies).
2. `vivqa-s1` (~40 MB): `data/{train,calib,dev,test}.jsonl`, `s1bench/`, `evaluate.py`, `README.md`, plus `SHA256SUMS.txt`.
3. `viso-code` (~250 MB): `ViVQAv2-main` WITHOUT `output/`, `saved_models/`, `analysis_results/`, `docs/openVqa.pdf`, `__pycache__`, test artifacts, and without any `.env`. Include `vivqa_s1/s1bench` as a sibling folder
   so the existing `utils/viso_s1.py` bridge finds it, OR add an `S1BENCH_PATH` env/config switch (own file) and document it. Test that `import utils.viso_s1` resolves in the unpacked layout.
The script prints sizes and verifies SHA256SUMS; it does not upload anything. It must refuse to include a file named `.env` or containing the key patterns `sk-`, `Bearer `.

## 6. README_KAGGLE.md (for the user, in plain steps)
1. Kaggle account, phone verification (needed for GPU and Internet), create the three datasets from the folders (web upload or `kaggle datasets create`, user's choice); keep them PRIVATE.
2. New notebook: Accelerator `GPU T4 x2` (or P100), Internet ON, add the three datasets as inputs.
3. Cells (give the exact text): (a) `pip install yacs termcolor tabulate pytorch_transformers` (list what the import probe needed; add `transformers` only if missing), copy code to `/kaggle/working/code`, print GPU info;
   (b) `python scripts/viso_kaggle_run.py --preflight ...` and send `preflight.json` to the planner BEFORE the full run; (c) two background runner processes, one per GPU, `--deadline-hours 8.5`; (d) `--finalize`, then download `results_bundle.zip`.
4. If the session stops early: "Save Version", then in the next session add that version's output as an input and pass `--prior-output`.
5. Budget table: expected GPU hours (from the preflight), weekly quota 30 h, what to do if over.

## 7. Acceptance for 2b-prep
1. All tests pass (`pytest tests/viso -p no:cacheprovider`), old 57 + new ones (resume equality, deadline, queue, dry-run, bootstrap, packaging).
2. Local CPU `--dry-run` of the runner over a 2-job tiny manifest works end to end, including the A3-from-A2 dependency, calibration, report and `results_bundle.zip`.
3. `kaggle_prepare.py` run for real into `~/data/kaggle_upload/` (not inside the project); show `du` sizes, SHA256SUMS verification, and the unpacked-layout import test.
4. No pre-existing repo file modified (sha256 manifest before/after, as before); nothing in `vivqa_s1/`, `vivqav2_dataset/`, `.env` touched or read.
5. Report to the planner: file list, test summary, the dry-run log, the upload folder sizes, `README_KAGGLE.md` text, deviations. Quote generated files only.
Then stop. The user runs preflight on Kaggle and sends `preflight.json` to the planner; the full run follows only after the planner has reviewed the projection.
