# ViSO on ViVQA-S1 (Phase 2a)

ViSO is the proposed baseline for the ViVQA-S1 benchmark (`../vivqa_s1`): ViCompreVQA's `ConstituentMCAN` encoder with
**typed decision heads**, an **RLCD** objective, **per-type temperature calibration** and a post-hoc **escalation head**.
Everything here is NEW files; no existing file of this repository (including every `__init__.py`) was edited, and
`../vivqa_s1` and `../vivqav2_dataset` are read-only. Phase 2a = code, unit tests and tiny plumbing runs (CPU only).
Full training is Phase 2b.

**The question type is part of the interface.** At inference the item says whether it is a `noul` (boolean), `score`
(count 0..10+) or `choice` (16 candidate texts) question; there is no type predictor. Comparing ViSO with the flat
baseline A0 is a comparison of output *parametrisations* under the same typed interface, not of type prediction.

## Layout
```
train_viso.py                         entry point (explicit imports register the new modules)
configs/viso_vivqav2.yaml             ViSO base config (DEVICE cuda for 2b; override with --opts MODEL.DEVICE cpu)
configs/viso_flat_vivqav2.yaml        A0 FlatS1 (ViCompreVQA flat classifier)
data_utils/vocabs/s1_vocab.py         S1Vocab        data_utils/datasets/s1_dataset.py  S1FeatureDataset, s1_collate
models/viso/{encoder,heads,viso_model}.py   ViSOEncoder, Noul/Score/ChoiceHead (+SetAttention), ViSO, FlatS1
tasks/typed_decision_task.py          TypedDecisionTask (hard_ce / soft_ce / rlcd, dev selection, checkpoints)
utils/viso_s1.py                      bridge to ../vivqa_s1 (s1bench on sys.path)       utils/viso_rlcd.py
utils/viso_calibration.py             temperature + escalation + IO                     utils/viso_efficiency.py
scripts/viso_predict.py  scripts/viso_calibrate.py  scripts/viso_report.py  scripts/viso_tiny_runs.sh
tests/viso/                           pytest (CPU)
```

## Environment
`uv venv ~/.venvs/viso --python 3.12`; CPU torch via
`uv pip install --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match torch`;
then `numpy pyyaml yacs tqdm tabulate termcolor requests pillow transformers pytorch_transformers pytest`.
(`transformers` and `pytorch_transformers` are needed only because `models/__init__.py` and `models/modules/text_embeddings.py`
import them eagerly.) Always run with `PYTHONDONTWRITEBYTECODE=1` and `pytest -p no:cacheprovider`. Full `uv pip freeze`:
see the end of this file.

## Commands
```bash
export PYTHONDONTWRITEBYTECODE=1; PY=~/.venvs/viso/bin/python
$PY -m pytest tests/viso -p no:cacheprovider -q                    # unit tests (~3 min on 8 CPU cores)
$PY train_viso.py --config-file configs/viso_vivqav2.yaml --opts MODEL.DEVICE cpu MODEL.NAME my_run LOSS.NAME rlcd
$PY scripts/viso_predict.py --config-file configs/viso_vivqav2.yaml --opts MODEL.NAME my_run     # logits + preds jsonl
$PY scripts/viso_calibrate.py --run-dir output/viso/my_run          # temperatures, calibrated preds, escalation head
$PY scripts/viso_report.py --runs A2=output/viso/a2:pertype --split test --out results.md
$PY -m utils.viso_efficiency --config-file configs/viso_vivqav2.yaml configs/viso_flat_vivqav2.yaml --out eff.json
bash scripts/viso_tiny_runs.sh                                     # A0..A6 plumbing runs + determinism check
```
Checkpoints go to `saved_models/<MODEL.NAME>/` (`vocab.bin`, `best_model.pth`, `best_model_stage1.pth`), logs/outputs to
`output/viso/<MODEL.NAME>/` (`config.yaml`, `train_log.jsonl`, `logits_<split>.npz`, `preds_<split>.jsonl`, `calibration/`).

## Config flags (all are `--opts` overrides)
| flag | meaning |
|---|---|
| `MODEL.ARCHITECTURE` ViSO / FlatS1 | typed heads / A0 flat classifier (use the matching yaml) |
| `MODEL.ZERO_VISION` | A5: all-zero region features, explicit mask kept (question-only ablation) |
| `MODEL.SET_ATTENTION` | A6: `False` removes the set-attention layers of the choice head |
| `MODEL.SHARE_EMBEDDING` | candidate encoder shares the question word-embedding table |
| `LOSS.NAME` hard_ce / soft_ce / rlcd, `LOSS.STAGE` all / sup / rl | objective; rlcd = stage 1 soft_ce then stage 2 rlcd |
| `LOSS.TYPE_BALANCE` | average of per-type means instead of the item mean |
| `LOSS.PG_WEIGHT NUM_SAMPLES SIGMA_START SIGMA_END SPHERICAL_WEIGHT RPS_WEIGHT` | RLCD constants (1.0, 4, 0.4, 0.1, 0.75, 1.0) |
| `TRAINING.EPOCHS_SUP PATIENCE EPOCHS_RL LR_ENCODER LR_HEADS WEIGHT_DECAY WARMUP_STEPS GRAD_CLIP SEED` | optimiser schedule (25, 5, 5, 2e-4, 1e-3, 0.01, 500, 1.0, 0) |
| `DATASET.FEATURE_DATASET.LIMIT` | tiny runs: type-balanced deterministic subset of every split (first items of each type in file order) |
| `DATASET.FEATURE_DATASET.BATCH_SIZE WORKERS FEATURE_CACHE` | loader; `FEATURE_CACHE` = LRU size of loaded feature files (0 = off) |

Note on `--opts`: yacs type-checks overrides, so float keys need float literals (`LOSS.PG_WEIGHT 0.0`, not `0`; `0` is
rejected with a type-mismatch error). The A3c control (rlcd with the policy-gradient term switched off, same stage-2
epochs, soft-CE only) is `LOSS.NAME rlcd LOSS.PG_WEIGHT 0.0`: the loss is then exactly `soft_ce + 0.0 * pg`
(checked: identical to `soft_ce` for a batch; the pg term stays finite and only consumes the separate RLCD noise generator).

## Packed region features (Phase 2b step 0)
`scripts/viso_pack_features.py` packs `region_features` of the 34,893 images used by ViVQA-S1 into one float16 array
(`regions_fp16.bin`, exact lengths, ascending coco_id) + `index.npz` + `tags.jsonl` (objects / attributes, for a later
`state_text` ablation) + `meta.json`, **outside the project folder** (default `~/data/vivqa_packed/`; the script refuses to
write inside the project). Use it with `DATASET.FEATURE_DATASET.FEATURE_FORMAT packed` and `... .PACKED_PATH <dir>`; the default
stays `npy`. The reader (`utils/viso_packed_features.py`) opens a lazy memmap per process, so DataLoader workers are safe.
```bash
nice -n 10 ionice -c3 $PY scripts/viso_pack_features.py --out ~/data/vivqa_packed          # ~100 s, < 300 MB RAM
$PY scripts/viso_pack_features.py --out ~/data/vivqa_packed --verify                        # writes verify.json next to the pack
```
The pack is deterministic (sha256 of the three data files equal on rerun; `meta.json` holds the creation time).

## Resumable runs and the Kaggle runner (Phase 2b-prep)
* `tasks/typed_decision_task.py` saves `last.pth` after EVERY epoch (model, optimiser, scheduler, epoch, stage, best NLL, early-stopping counter, global
  step, CPU/CUDA RNG states, the DataLoader generator and the RLCD noise generator). With `TRAINING.RESUME True` a restart continues exactly where the run
  stopped (the CPU test `tests/viso/test_resume.py` checks that 3 epochs straight == 1 epoch + resume 2: identical dev logs and logits sha256, also across the
  stage-1 / RL boundary). A finished run writes `DONE.json` (stage results, best epoch, dev NLL, wall time) and deletes `last.pth`; a finished run is
  skipped on restart. `TRAINING.DEADLINE_UNIX` (> 0): after an epoch, if now + 1.2 x the mean epoch time would pass the deadline, the run exits with code 3
  (`IncompleteRun`) with `last.pth` saved. `TRAINING.STOP_AFTER_EPOCHS` simulates an interruption (tests). Defaults (`RESUME False`, no deadline) keep the old behaviour.
* `scripts/viso_kaggle_run.py` runs `kaggle/experiments.json` (A0, A1, A2, A3, A3c, A4 = A3 + calibration, A5, A6; seeds 0-2 for A0/A2/A3/A3c; 9 supervised + 6 RL-stage
  jobs). One process per GPU shares the queue (atomic `os.mkdir` locks in `<out>/.locks`, stale locks of dead pids are stolen); A3 / A3c copy A2's best checkpoint of
  the same seed as their stage-1 checkpoint and train `LOSS.STAGE rl` (A3c has `LOSS.PG_WEIGHT 0.0`, a float). `--preflight` times 200 steps of A2 and A0;
  `--finalize` writes `results.md`, `bootstrap.json` (`scripts/viso_bootstrap.py`: paired bootstrap, 1,000 resamples, seed 0, via the s1bench item scores, mean ± std
  over seeds) and `results_bundle.zip` (no checkpoints). `--dry-run` executes everything on CPU with the manifest's `dry_run_overrides`.
* `scripts/kaggle_prepare.py` builds `~/data/kaggle_upload/` (`vivqa-packed`, `vivqa-s1`, `viso-code`, each with `dataset-metadata.json` and `SHA256SUMS.txt`) and refuses
  `.env` files and key-like patterns; `kaggle/README_KAGGLE.md` has the click-by-click steps. On Kaggle `s1bench` is found through `VIVQA_S1_ROOT=/kaggle/input/vivqa-s1`.
* fp32 only: `TRAINING.AMP` must stay False (the -1e9 / -1e5 masks overflow in fp16). `torch.backends.cudnn.deterministic=True`, `benchmark=False`; GPU runs are not
  bit-exact (the CPU determinism tests are the reference).
* Tests write checkpoints of 100+ MB, so `tests/viso/conftest.py` overrides `tmp_path` to `~/data/viso_test_tmp/` (removed after every test; `VISO_TEST_TMP` overrides).

## Method
* **Encoder** (`ViSOEncoder`): copy of `ConstituentMCAN.forward` returning the 512-d fused vector `h`. Two deliberate
  differences: (1) the vision padding mask is built from the explicit `region_mask` of the batch (the existing
  `FeatureEmbedding` masks rows that *sum to zero*, which would mask everything under `ZERO_VISION`) and the vision attention
  pooling is masked with it, so zero-padded regions cannot change `h` (tested to 1e-5; the original pools over padded
  rows too); (2) text pooling is unchanged (questions have a fixed padded length). A0 uses the same encoder body.
* **Heads** (shared 16-wide logit layout, masked columns = -1e9): `NoulHead` MLP 512-512-2; `ScoreHead` MLP 512-512-11 (level
  10 means >= 10); `ChoiceHead`: candidate text -> shared word embedding -> masked mean -> Linear(300,512) -> LayerNorm ->
  `SetAttention` (2 Transformer encoder layers, 8 heads, ff 1024, no positional encoding, key-padding mask: permutation
  equivariant) ; `score_i = <W_q h, c_i> / sqrt(512)`.
* **A0 (FlatS1)**: flat linear classifier over `<yes>, <no>, #0..#10` + every train choice gold (`vocab.total_answers`);
  evaluated on the SAME restricted label space (noul [<no>,<yes>], score #0..#10, choice = the candidates; a candidate
  absent from the flat vocab gets `min(row logits) - 5`). Its soft/hard CE runs over the full flat vocabulary with the
  item's labels scattered into it (labels absent from the flat vocab dropped, rest renormalised).
* **Objectives**: `hard_ce` (one-hot gold), `soft_ce` (`-sum g_i log softmax`). **RLCD** = soft_ce + `PG_WEIGHT` x policy-gradient:
  K = 4 samples `z_k = z.detach() + sigma eps_k` (own seeded generator, valid columns only), `p_k = softmax(z_k)`,
  reward `r_k = 0.75 * spherical(p_k, g) [+ 1.0 * (1 - RPS(p_k, g)) for score items]`, with
  `spherical(p, g) = sum g_i p_i / ||p||_2` and RPS over the 11 ordered levels normalised by K-1 (same as
  `s1bench.metrics.rps`; unordered types do not use RPS), advantage `A_k = (r_k - mean)/(std + 1e-6)`, loss
  `-mean_k A_k.detach() * ( -||z_k - z||^2 / (2 sigma^2) )`; sigma is annealed 0.4 -> 0.1 linearly over the RL steps.
* **Selection**: for every method the checkpoint with the best DEV primary NLL (s1bench, temperature-free probabilities).
  For rlcd the final model is the best DEV-NLL epoch of stage 2 (the stage-1 best is kept as `best_model_stage1.pth`).
* **Calibration** (`scripts/viso_calibrate.py`, fitted on CALIB only): per type, LBFGS on `log T` (clamped to [0.1, 10])
  minimising NLL vs `gold_label`; ablations: one global T, and per-type T on the soft gold (cross-entropy vs the soft
  distribution). Applied as `softmax(logits / T_type)`.
* **Escalation head**: logistic regression (torch LBFGS, L2 1e-3, standardised features) predicting `error = gold_label not in the
  argmax set` from `[max_prob, entropy/log|space|, top1-top2 margin, one-hot type, max_prob x one-hot type]` of the
  calibrated probabilities; 5-fold out-of-fold on calib, then fitted on all calib and scored on dev/test; reports AUROC,
  AURC and selective accuracy at coverage 0.5/0.7/0.9 against the baseline score `max_prob`. It is "adopted" only if it
  beats `max_prob` by a stated margin; numbers are reported either way.
* **Efficiency**: parameters (encoder / each head / total), fp32/fp16 checkpoint size, FLOPs of a batch-1 forward
  (`FlopCounterMode`, 44 regions, 20 question tokens, 16 candidates), CPU latency (1 and 4 threads, 20 warm-up + 200 runs,
  median / p95), GPU latency if CUDA is present. The VinVL region extractor is NOT included.

## Deviations from the plan (and why)
1. `FlatS1` / ViSO share `ViSOEncoder` (masked vision pooling) instead of the unmodified `ConstituentMCAN`, so A0 and ViSO
   differ only in the output parametrisation; the original model's pooling over padded rows is not reproduced.
2. `LIMIT` selects a type-balanced subset (first items of each type) rather than the first N items: the first 64 train
   items are 63 `noul` + 1 `choice` and would not exercise score/choice.
3. The overfit smoke tests use the budgets found by probing (ViSO: 16 items / 50 steps, FlatS1: 32 items / 20 steps,
   dropout off) instead of 64 items / 40 steps: with the default learning rates the 64-item / 40-step budget only reached
   a loss ratio of about 0.5-0.6 for ViSO (the halving check is deterministic with these settings).
4. `tests/viso` takes about 3 minutes on this machine (the plan said < 60 s): the two overfit tests and the RLCD stage
   dominate.
5. Added files not in the plan: `utils/viso_s1.py` (bridge), `scripts/viso_tiny_runs.sh`, `tests/__init__.py`.
6. `TypedDecisionTask` does not inherit `BaseTask` (its constructor builds an Adam/NLLLoss setup and its checkpointing uses
   cuda RNG states); it follows the same conventions (registry, `vocab.bin`, `saved_models/`).

## Known risks
* Random access to the 27 GB feature folder: `FEATURE_CACHE` and `WORKERS` are the levers; features are never preloaded.
* Candidate words of dev/test that are not in the train+calib vocabulary become `<unk>` (the share is reported by the tiny-run
  report).
* `transformers` 5.x and `pytorch_transformers` are imported only for their side effect on `models/__init__.py`; they are not
  used by ViSO.
* RLCD's policy-gradient term is a Gaussian-exploration approximation of Laya's recipe; its weight and sigma schedule are
  untuned (Phase 2b).

## Full `uv pip freeze` of ~/.venvs/viso (Python 3.12.14)

```
annotated-doc==0.0.5
anyio==4.15.1
boto3==1.43.109
botocore==1.43.109
certifi==2026.7.22
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
filelock==4.0.12
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.7.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.33.0
idna==3.20
iniconfig==2.3.1
jinja2==3.1.6
jmespath==1.1.0
joblib==1.6.0
markdown-it-py==4.2.0
markupsafe==3.0.4
mdurl==0.1.2
mpmath==1.3.0
networkx==3.7
numpy==2.5.3
packaging==26.3
pillow==12.3.0
pluggy==1.6.0
pygments==2.21.0
pytest==9.1.1
python-dateutil==2.9.0.post0
pytorch-transformers==1.2.0
pyyaml==6.0.3
regex==2026.9.29
requests==2.34.2
rich==15.0.0
s3transfer==0.19.2
sacremoses==0.2.0
safetensors==0.8.0
sentencepiece==0.2.2
setuptools==84.0.0
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
tabulate==0.10.0
termcolor==3.3.0
tokenizers==0.23.2
torch==2.14.1+cpu
tqdm==4.70.1
transformers==5.19.0
typer==0.27.3
typing-extensions==4.16.0
urllib3==2.8.0
yacs==0.1.8
```
