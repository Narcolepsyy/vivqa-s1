# ViSO Phase 2a plan (planner: ie307-final-ec; implementor: ie307-final-7f)

Phase 1 (ViVQA-S1 benchmark in /home/khaitran/project/IE307-Final/vivqa_s1) is ACCEPTED and frozen.
Phase 2 builds ViSO, the proposed baseline, on top of it. Phase 2a = code + unit tests + tiny smoke runs
(CPU only, no full training). Phase 2b = full training runs; its plan comes after 2a is accepted (compute venue
is undecided).

## 0. Definitions
- S1 item: see vivqa_s1/README.md. Types: `noul` (2 keys "false","true"), `score` (11 keys "0".."10", level 10 = ">=10"),
  `choice` (16 candidate texts). Gold = probability distribution; `gold_label` = one-hot reference; `human_subset`
  (= S1-Human) marks strictly human-averaged items. Splits: train / calib / dev / test (data/*.jsonl).
- ViSO = ViCompreVQA encoder (models/constituent_mcan, ConstituentMCAN, VinVL region features) + typed heads
  (noul / score / choice) + RLCD loss + per-type temperature calibration + post-hoc escalation head.
- RLCD (after Laya docs/finetune.md): soft cross-entropy at full weight + a GRPO-style policy-gradient term, 4 samples
  per item, exploration noise on the logits annealed 0.4 -> 0.1, rewards = proper scoring rules (spherical weight 0.75,
  ranked probability score weight 1.0). Temperatures: one per type, LBFGS on log T, clamped to [0.1, 10], fitted on a
  held-out calibration slice.

## 1. Hard constraints
- NEW FILES ONLY under /home/khaitran/project/IE307-Final/ViVQAv2-main. Never edit an existing file (including every
  `__init__.py`, configs, tasks, models). New modules register themselves in the existing registries
  (META_ARCHITECTURE, META_TASK, META_DATASET, META_VOCAB) when imported; `train_viso.py` imports them explicitly
  (e.g. `import models.viso`), so no `__init__` edit is needed. If a registration really cannot work without editing
  an existing file, STOP and ask me.
- vivqa_s1/ and ../vivqav2_dataset are READ-ONLY. Import `s1bench` via `sys.path` (sibling dir) for schema/metrics;
  do not copy it.
- Features: ../vivqav2_dataset/vinvl_vinvl/<coco_id>.npy (dict with region_features (~44,2048), grid_features,
  attr_features, objects, boxes, attributes). Every coco_id of S1 has a file (verified). Use region_features only.
  `objects` (Vietnamese tags) are reserved for a later `state_text` ablation, not used now.
- Env: the user approved a CPU-only venv OUTSIDE the repo: `uv venv ~/.venvs/viso` + CPU torch + numpy, pyyaml, yacs,
  tqdm, tabulate, pytest (add other imports the probe in step 1 needs). Nothing system-wide. No GPU is needed in 2a.
  Do not touch the 27 GB feature folder except reading a few files (smoke tests use ~64 items).
- Determinism: everything seeded (`--seed`, default 0): python/numpy/torch, DataLoader generator, RLCD noise
  (own `torch.Generator`). Unit tests must be deterministic.
- Do not use the word "Jev" in any name.

## 2. Files to create (all new)
```
ViVQAv2-main/
  train_viso.py                         entry: import registrations, get_config, build_task, task.start()
  configs/viso_vivqav2.yaml             ViSO typed heads (base config; every variant is a CLI/yaml flag, see 4)
  configs/viso_flat_vivqav2.yaml        A0 ViCompreVQA flat head (baseline)
  data_utils/vocabs/s1_vocab.py         S1Vocab (META_VOCAB): question words + candidate words + flat answer vocab
  data_utils/datasets/s1_dataset.py     S1FeatureDataset (META_DATASET) + s1_collate
  models/viso/__init__.py
  models/viso/encoder.py                ViSOEncoder(ConstituentMCAN): returns fused vector h and break_probs
  models/viso/heads.py                  NoulHead, ScoreHead, ChoiceHead (+SetAttention), EscalationHead helper
  models/viso/viso_model.py             ViSO (META_ARCHITECTURE) and FlatS1 (A0, META_ARCHITECTURE)
  tasks/typed_decision_task.py          TypedDecisionTask (META_TASK): soft-CE / hard-CE / RLCD, eval, checkpoints
  utils/viso_rlcd.py                    pure functions: spherical, rps, rewards, GRPO policy-gradient loss
  utils/viso_calibration.py             per-type temperature fit (LBFGS on log T), apply, ECE helpers
  utils/viso_efficiency.py              params / FLOPs / latency proxies
  scripts/viso_predict.py               dump raw logits + predictions for calib/dev/test (jsonl for evaluate.py)
  scripts/viso_calibrate.py             temperatures (+ ablations) and escalation head; writes calibrated preds
  scripts/viso_report.py                aggregate runs into results markdown (calls vivqa_s1 metrics)
  tests/viso/test_*.py                  pytest (CPU)
  docs/viso_README.md                   how to run, config flags, method description, deviations
```
Checkpoints and outputs go to `saved_models/<MODEL.NAME>/` (existing convention) and `output/viso/<run>/`.

## 3. Components

### 3.1 S1Vocab / S1FeatureDataset
- Mirror the attributes the existing text embedding needs (inspect `Vocab` base, `build_text_embedding`,
  `LSTMTextEmbedding`, `encode_question`); reuse `preprocess_sentence` with the same tokenizer setting as the
  original vocab (TOKENIZER: null).
- Vocabulary: words of all train+calib questions and all candidate texts in train+calib (+ specials). Dev/test words
  not in the vocab -> `<unk>`. Save with the checkpoint (pickle, existing convention).
- Flat answer vocab (for A0 only): `<yes>`, `<no>`, `#0`..`#10`, plus every distinct choice gold_label seen in
  train. `total_answers` = its size.
- Item -> Instance: `id`, `region_features` (float32, as stored), `question_tokens`, `type_id` (0 noul, 1 score,
  2 choice), `space_mask` (bool[16]; noul 2, score 11, choice = #candidates), `target` (float[16], soft gold padded),
  `gold_idx` (index of gold_label in the space), `human_flag`, `cand_tokens` (int[16, Lc], Lc <= 6 tokens, pad 0) and
  `cand_mask` for choice (zeros otherwise), `flat_target` (float[n_flat], A0 only).
- Label space order: noul ["false","true"], score ["0".."10"], choice = the item's `criteria` order (already shuffled
  by Phase 1). No sorting.
- Collate -> padded tensors; one `s1_collate`. Avoid the O(N*M) image lookup used by the old datasets (index by id).
- Optional small LRU cache of loaded feature dicts (flag `FEATURE_CACHE`, default 0). Default `WORKERS: 0` (Windows
  habit of the repo); allow >0.
- Unit tests: shapes, mask correctness, target sums to 1 over the mask, gold_idx points at gold_label, label space order.

### 3.2 Encoder
`ViSOEncoder(ConstituentMCAN)`: call `super().__init__(config, vocab)` then `del self.classify`; override `forward` to
run exactly the ConstituentMCAN computation (vision embedding, text embedding, constituent encoder, self/guided
encoders, attention pooling, `layer_norm(vision_proj + text_proj)`) and return `(h, break_probs)` with `h` of size
D_MODEL (512). Copy the logic rather than editing the original. Flag `MODEL.ZERO_VISION: true` multiplies
region_features by 0 (question-only ablation, A5).

### 3.3 Heads (all in the shared 16-wide logit layout)
Output of every head: logits `(B,16)` with `-inf` (use a large negative, e.g. -1e9, in fp32) outside `space_mask`.
- NoulHead: MLP(512->512->2) -> columns 0..1.
- ScoreHead: MLP(512->512->11) -> columns 0..10.
- ChoiceHead:
  - candidate encoder: `nn.Embedding` initialised from the question word embedding weights (shared table, config flag
    `SHARE_EMBEDDING: true`), mean-pool over non-pad tokens, `Linear(300->512)`, LayerNorm.
  - `SetAttention`: 2 layers of a Transformer encoder over the <=16 candidates, d=512, 8 heads, ff 1024, dropout 0.1,
    NO positional encoding, key padding mask from `cand_mask` (permutation equivariant, so shuffling candidates cannot
    change scores).
  - score_i = <W_q h, c_i> / sqrt(512). Flag `SET_ATTENTION: false` skips the set-attention layers (ablation).
- `ViSO.forward(batch)` -> dict(logits (B,16), h (B,512), break_probs). Route by `type_id` (run each head on its
  rows). There is NO type predictor: the question type is part of the interface.
- `FlatS1.forward` (A0): ConstituentMCAN with a Linear over the flat vocab -> logits (B, n_flat); `restrict(batch)`
  maps flat logits into the same 16-wide layout (noul: [<no>,<yes>], score: #0..#10, choice: the candidate texts;
  a candidate absent from the flat vocab gets `min(row logits) - 5`, documented). Evaluation of A0 always uses the
  restricted space (otherwise it would be unfair).

### 3.4 Training objectives (config `LOSS.NAME`)
All use mean-reduced per-item loss; `LOSS.TYPE_BALANCE: false` default (flag: average the per-type means).
- `hard_ce`: CE against one-hot `gold_idx` (masked log-softmax).
- `soft_ce`: `-sum_i target_i * log softmax(logits)_i` over the masked space. (A0 uses the same objective over the
  full flat vocab with `flat_target`.)
- `rlcd` = soft_ce (weight 1.0) + `LOSS.PG_WEIGHT` (1.0) * policy-gradient term:
  - for each item draw K=4 noisy logits `z_k = z.detach() + sigma * eps_k` (eps ~ N(0,I), masked columns excluded),
    sigma annealed linearly 0.4 -> 0.1 over the RL epochs;
  - decision distribution `p_k = softmax(z_k)` over the masked space;
  - reward: `r_k = 0.75 * spherical(p_k, g)`; for `score` items add `1.0 * (1 - RPS(p_k, g))`
    (g = soft target; RPS over the 11 ordered levels, normalised by K-1, same definition as s1bench.metrics.rps).
    spherical(p,g) = sum_i g_i p_i / ||p||_2. `noul`/`choice` are unordered, so RPS is not used for them (document);
  - group advantage `A_k = (r_k - mean_k r) / (std_k r + 1e-6)`;
  - `loss_pg = - mean_k [ A_k.detach() * log N(z_k ; z, sigma^2 I) ]`, with `z_k` treated as a constant and
    `log N = - ||z_k - z||^2 / (2 sigma^2)` over the masked columns (z carries the gradient).
  - Training schedule: Stage 1 = soft_ce for up to `EPOCHS_SUP` with early stopping; Stage 2 = start from the best
    stage-1 checkpoint and train `EPOCHS_RL` (default 5) with `rlcd`. Config key `LOSS.STAGE`.
- Optimiser: AdamW, encoder lr 2e-4 and heads lr 1e-3 (param groups), weight decay 0.01, cosine decay with 500 warmup
  steps, grad-clip 1.0, batch 64, fp32 on CPU smoke runs (`AMP` flag for GPU later). Max 25 supervised epochs,
  patience 5. These are defaults for 2b; 2a must only prove they run.
- Model selection (all methods, for fairness): best DEV primary NLL vs gold_label (computed with s1bench.metrics on
  the temperature-free probabilities). Log dev accuracy / Brier / ECE every epoch.

### 3.5 Calibration and escalation (scripts/viso_calibrate.py)
- `viso_predict.py` saves, for calib/dev/test, the raw masked logits (npz: ids, type, logits[N,16], space_mask) so that
  calibration never needs another forward pass, and writes `preds_<split>.jsonl` ({"id","probabilities"} keyed by
  the item's label keys) for evaluate.py.
- Temperature: for each type, LBFGS on log T (clamp [0.1, 10]) minimising NLL vs `gold_label` on calib (primary
  target). Also report two ablations: one global T for all types; per-type T fitted on the SOFT gold (cross-entropy vs
  `gold` distribution). Apply as `softmax(logits / T_type)`.
- Escalation head (System-1 -> System-2 deferral signal, no new encoder): a logistic regression (torch, LBFGS, L2 1e-3)
  predicting `error = (argmax != gold_label)` from features `[max_prob, entropy / log(|space|), top1-top2 margin,
  one-hot type, max_prob * one-hot type]` computed from the CALIBRATED probabilities. Evaluate it out-of-fold on calib
  (5 folds, seed 0) and then fit on all calib and score dev/test. Report AUROC of error detection and AURC /
  selective accuracy at coverage 0.5 / 0.7 / 0.9 against the baseline score `max_prob`. Report the numbers either
  way; the head is only "adopted" in the paper if it beats max_prob by a margin we state.

### 3.6 Efficiency proxies (utils/viso_efficiency.py)
Report for ViSO and FlatS1: parameter counts (encoder, each head, total), checkpoint size fp32/fp16 (MB),
FLOPs of one forward at batch 1 (44 regions, 20 question tokens, 16 candidates; `torch.utils.flop_counter.FlopCounterMode`),
CPU latency (batch 1, 1 and 4 threads, 20 warmup + 200 timed runs, median and p95; seeded inputs) and GPU latency if CUDA is
available. State explicitly that the VinVL region extractor is NOT included (it is not on-device feasible; this
is the cost of the features, not of ViSO).

## 4. Experiment matrix (to be run in Phase 2b; 2a only builds the switches and proves each runs)
| id | model | loss | calib | notes |
|---|---|---|---|---|
| A0 | FlatS1 (ViCompreVQA, restricted at eval) | soft_ce over flat vocab | none | the baseline |
| A0T | A0 | same | per-type T | fairness: calibrated baseline |
| A1 | ViSO | hard_ce | none | typed heads, hard labels |
| A2 | ViSO | soft_ce | none | typed heads + soft labels |
| A3 | ViSO | rlcd | none | + RLCD |
| A4 | A3 | - | per-type T (+ escalation head) | full ViSO |
| A5 | ViSO, `ZERO_VISION` | soft_ce | none | question-only language-bias check |
| A6 | ViSO, `SET_ATTENTION: false` | soft_ce | none | choice-head ablation |
2b protocol (for the record, not for 2a): seeds 0,1,2 for A0, A2, A3, A4; 1 seed for the others; test only touched for
final reports; paired bootstrap (1000 resamples, seed 0) of A3-A2, A2-A0, A4-A3 on test primary accuracy / NLL / ECE.
Main tables: s1 `evaluate.py` primary (all items) and S1-Human secondary, overall and per type.

## 5. Tests (pytest, CPU, `tests/viso/`, no GPU, <60 s total, features faked where possible)
1. Dataset/collate (3.1 list) on a 32-item subset using real npy files for those ids.
2. Heads: output layout and masking; ChoiceHead is permutation EQUIVARIANT (shuffle candidates -> same scores,
   shuffled) and handles 2..16 candidates; padding candidates never change real scores.
3. RLCD functions: spherical/RPS hand cases (RPS equals s1bench.metrics.rps; perfect prediction = best reward);
   advantages have mean ~0 per group; with a fixed seed `loss_pg` is deterministic; the gradient w.r.t. `z` is non-zero
   and finite; `sigma` schedule endpoints 0.4 and 0.1.
4. Calibration: synthetic logits generated with a known temperature (e.g. true T=2.0 over 5000 samples) are recovered
   within 10%; clamp works; per-type fit is independent.
5. Escalation: on synthetic data where error is a function of max_prob, AUROC > 0.9 and the 5-fold split is deterministic.
6. Model smoke: ViSO and FlatS1 forward/backward on a 4-item batch from real data (fp32 CPU): finite loss.
7. Overfit smoke: 64 train items, 40 steps, soft_ce loss at least halves for ViSO and for FlatS1 (seeded). RLCD stage
   runs 2 steps without NaN.
8. `scripts/viso_predict.py` output passes through `vivqa_s1/evaluate.py` (use the 64-item subset as a tiny gold file)
   with no missing predictions.

## 6. Acceptance for Phase 2a
1. All files exist; no existing file in ViVQAv2-main or anything in vivqa_s1/vivqav2_dataset was modified (list the
   files you created; show a before/after listing diff of the two trees).
2. `pytest tests/viso -p no:cacheprovider` passes in the venv.
3. One tiny end-to-end per matrix row A0..A6 (64 train items, 1-2 epochs, dev = 64 items) runs through
   train -> viso_predict -> viso_calibrate -> evaluate.py without error (A4 includes the escalation head). Send the
   logged losses, the parameter counts and CPU latency from `viso_efficiency.py` (ViSO vs FlatS1). These numbers are
   plumbing checks only; do NOT quote them as results.
4. Deterministic: the same tiny run twice gives identical logits (sha256 of the npz) on CPU.
5. `docs/viso_README.md`: commands, config flags, formulas for RLCD/calibration/escalation, deviations, known risks.
6. Report back: file tree, test summary, the 7 tiny-run summaries (quote generated logs), deviations. Do not hand-type numbers.

## 7. Known risks to watch (report if they bite)
- Importing `models`/`tasks` packages may pull heavy or missing dependencies (their `__init__` files import many
  models); do an import probe first (step 1) and report the minimal dependency set. If an `__init__` import cannot
  be avoided, tell me which dependency fails.
- 27 GB of features, random-access disk reads: `FEATURE_CACHE` and `WORKERS` are the levers; do not preload.
- Candidate words missing from the vocab -> `<unk>`; check the share of `<unk>` candidate tokens on train/dev/test and
  report it.
- A0 vs ViSO fairness: same encoder config, optimiser family, model selection rule. Any deviation must be reported.
- The question type is given at inference (typed interface); this must be stated in the docs so the comparison to
  flat classification is not misread as type prediction.

## 8. Step order
1. venv + import probe (dependency set report to me, one message, before writing code).
2. S1Vocab + dataset + tests. 3. Encoder + heads + ViSO/FlatS1 + tests. 4. Losses/RLCD/task + tests.
5. predict / calibrate / escalation / efficiency / report scripts + tests. 6. Tiny runs A0..A6. 7. Docs + report.
Send me the step-1 message first and the final report last; ask early if anything is ambiguous.
