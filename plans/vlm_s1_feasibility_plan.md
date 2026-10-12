# VLM-S1 feasibility (planner: ie307-final-ec; implementor: ie307-final-7f)

Goal of this step ONLY: decide which small VLM can be fine-tuned "System One style" (typed heads, soft-CE then RLCD, per-type
temperature) on a Kaggle T4, and measure what it costs. No real training. Output = a Kaggle feasibility notebook + packaged images +
a decision table the planner reviews. The full VLM-S1 plan follows after the numbers.

## 0. Why
Baseline table = zero-shot VLM (reference), VLM-S1 (fine-tuned System One style; the strong baseline), ViSO (ours, lightweight).
Planner estimates (to be replaced by measurements): 1B ~1-1.5 h/epoch, 2B ~3-4 h/epoch on one T4, 32k items/epoch.

## 1. Hard rules
- Same as before: pre-existing repo files, vivqa_s1/, vivqav2_dataset/ READ-ONLY; NEW files only (plus your own Phase 2 files if really needed;
  keep the 88 tests green). No keys, do not read .env, no network from this machine, no Kaggle API. Outputs of heavy steps outside the project
  (~/data/...). Do NOT start training. The user runs the notebook.
- No bf16 on T4: fp16 autocast + GradScaler, LoRA weights and heads in fp32. All masks must use a dtype-safe fill (e.g. `finfo(dtype).min`
  after casting scores to fp32 BEFORE softmax); add a CPU test that feeds fp16 logits through the head/loss code and checks finite results.
- Do not use a model that helped label or check the benchmark as the baseline (circularity): not gpt-*, not deepseek-*; open weights only.

## 2. Candidates (the notebook has internet ON; confirm exact HF ids with `huggingface_hub` at run time; report ids + licences + revision hashes)
1. SmolVLM2-500M (HF SmolVLM2 family, ~0.5B)
2. InternVL3-1B
3. Vintern-1B (Vietnamese fine-tune of InternVL-class; check the current best version on the Hub)
4. Qwen3-VL-2B (strongest; included to see how bad 2B really is)
For each: parameter count, vocabulary/tokenizer coverage of Vietnamese (tokens per Vietnamese question: mean over 500 S1 questions), image-token count at the
chosen resolution, whether `trust_remote_code` is needed, transformers version required (pin it; Kaggle's preinstalled version may differ).

## 3. Interface to implement (scripts + library, new files)
```
vlm/                               new package dir at ViVQAv2-main/vlm/
  vlm_s1_model.py                  wrapper: backbone (frozen + LoRA) -> last-token / pooled hidden state h -> the SAME NoulHead/ScoreHead as ViSO,
                                   ChoiceHead variant: candidate = mean-pooled input-embedding (or hidden) of its text tokens -> proj -> pointer score
                                   (no text generation; one forward per item)
  vlm_s1_data.py                   dataset: image (resized so the longest side <= --max-side, default 336) + prompt "Câu hỏi: ... (loại: noul|score|choice)";
                                   returns the S1 fields (type_id, space_mask, target, gold_idx, cand tokens) like S1FeatureDataset
scripts/vlm_feasibility.py         the notebook script
scripts/vlm_prepare_images.py      downscale + pack the 34,893 S1 images into ~/data/kaggle_upload/vivqa-images/ (JPEG q=90, longest side 448 px, keep names)
kaggle/README_VLM_FEASIBILITY.md   click-by-click for the user + the notebook cells
tests/vlm/...                      CPU tests with a TINY random-weight VLM (or a stub backbone with the same interface), no downloads
```
Reuse `utils/viso_rlcd.py`, `utils/viso_calibration.py`, `tasks/typed_decision_task.py` losses and s1bench metrics; do not duplicate them. If the task class cannot be reused
without edits to it, write a thin `VLMTypedDecisionTask` subclass in a new file.

## 4. What the feasibility script measures (per candidate, on a T4, seeds fixed, real S1 items and real images)
1. Load in fp16; LoRA r=16 on attention (+MLP if memory allows), gradient checkpointing on; trainable parameter count; peak GPU memory for batch 4/8/16 at the chosen image size.
2. 50 training steps (after 5 warmup) of soft-CE through the typed heads: seconds/step, items/s, projected hours per epoch of the 31,993-item train split, tokens/item
   (mean, p95), image tokens/item. Also eval throughput on 200 dev items (items/s).
3. Numerical health: loss finite, grad-scaler scale, any NaN/inf event; compare fp16 loss of a fixed batch against an fp32 forward (CPU or GPU) -> relative difference.
4. Zero-shot sanity (no training): the letter-prompt accuracy on 200 dev items is NOT required; skip unless trivial.
5. Image-resolution sweep for the best 1-2 candidates: max side 224 / 336 / 448 -> time/step and image tokens.
6. Writes `vlm_feasibility.json` (one object per candidate and per setting) + a markdown table. Fails gracefully per candidate (record the error, continue).
Summaries must be produced by script from the json; no hand-typed numbers.

## 5. Packaging
`vlm_prepare_images.py` runs locally on CPU under nice/ionice, deterministic, outputs a private Kaggle dataset folder `vivqa-images` (+ dataset-metadata.json with the `<KAGGLE_USERNAME>` placeholder, SHA256SUMS.txt). Report its size (target < 2 GB).
Re-run `kaggle_prepare.py`-style packaging for the new code folder (`viso-code` must include `vlm/` and the scripts) without breaking the existing three datasets: either a new `vlm-code` folder or an updated `viso-code` (state which). Same secret checks as before.

## 6. Tests (CPU, no downloads)
- Tiny random VLM stub: forward/backward through typed heads, finite fp16-safe losses, choice-head permutation equivariance, masks.
- Data: image resizing keeps aspect ratio and cap; prompt format; S1 fields identical to `S1FeatureDataset` for the same item (target, space_mask, gold_idx).
- Prepare-images script on 20 images; refusal of `.env`.
- Existing 88 tests still pass.

## 7. Acceptance
1. All tests (88 old + new) pass. 2. `vlm_prepare_images.py` ran for real; sizes and checksums quoted. 3. The notebook cells in README_VLM_FEASIBILITY.md are exact and
include the pip pins; a dry run of `vlm_feasibility.py --stub` on CPU works end to end and writes `vlm_feasibility.json`. 4. No pre-existing file modified (manifest diff).
5. Report: file list, test summary, dry-run json, upload folder sizes, README text, deviations, and the list of risks that cannot be tested locally. Then stop.
The user runs the Kaggle notebook and sends `vlm_feasibility.json` to the planner.
