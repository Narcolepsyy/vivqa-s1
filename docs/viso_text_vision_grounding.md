# Question-to-vision grounding in ViSOEncoder

This is one config-gated change to the existing ViSO encoder. It is not a new system, not a router, and not a measured accuracy gain. No training run was started for it. The unit tests use tiny random tensors and check shapes, masks, gradients, the config, and checkpoint compatibility. They are not accuracy evidence.

## What the encoder does today

`ViSOEncoder` (the `ConstituentMCAN` body) updates the two streams in one direction only:

1. The question tokens go through the constituent block and the self-attention stack. That path does not read the image.
2. The guided-attention stack then lets every visual token read the question.
3. Each stream is attention-pooled on its own. The fused vector is

```text
h = LayerNorm( vision_proj(pooled vision) + text_proj(pooled text) )
```

with an optional Hadamard term when `MODEL.FUSION_PRODUCT` is true.

`pooled text` does not depend on the image. Every typed head (yes/no, count, choice) reads `h`, so an image-invariant text vector is added into all three decisions. `CHOICE_CROSS_ATTENTION` and `SCORE_COUNT_POOL` ground candidates or count statistics against visual tokens after `h` is built. They do not change the text half of the fusion, and both stay off in the baseline config.

## The change

`MODEL.TEXT_VISION_GROUNDING` (default `false`) inserts one cross-attention block after the guided encoder and before text pooling. Question tokens are the queries. The visual tokens, already conditioned on the question by the guided encoder, are the keys and values. A residual is added to the question tokens:

```text
text <- text + tanh(gate) * LayerNorm(Attention(text, vision))
```

`gate` is a single parameter initialized at 0, so `tanh(gate)` is 0 and the residual is exactly 0. The block is constructed inside `torch.random.fork_rng()`, so its initialization does not advance the RNG that initializes the typed heads. A same-seed flag-on model therefore has the same weights everywhere else as the flag-off model, and `h` and the logits match until training moves the gate. The block has no dropout, so it does not draw random numbers and does not shift the dropout masks of the heads.

Other properties, all checked by `tests/viso/test_text_vision_grounding.py`:

- Padded visual tokens are removed from the softmax. Replacing them, or appending masked visual rows, does not change the output. A row with no real visual token gets a zero residual (a softmax over an empty key set is undefined).
- Padded question positions keep their incoming features. Appending masked question positions does not change `h`.
- No new positional encoding. Positions already live in the two streams (the guided encoder adds its own sinusoid to vision).
- The unpooled `vision_features` returned to the heads, and the constituent `break_probs`, are unchanged by the block. Choice cross-attention, when enabled separately, still sees the same visual tokens.
- Gold labels, soft targets, and image ids are not inputs. Changing them does not change the logits. The block cannot route on the test split or on the Image-Unseen view, because it never receives a split name, an image id, or a label.
- `MODEL.ZERO_VISION` still zeros the region features and the boxes and keeps the explicit mask. With the block on, two different images that share a mask produce the same `h`.
- The flag is implemented only on `ViSOEncoder`. `ViSO` and `FlatS1` raise if it is set with any other `MODEL.ENCODER_TYPE`, so a baseline encoder cannot silently ignore it. `FlatS1` uses the same encoder body, so the flag applies to A0 when the encoder type is `ViSO`.

The gate is the reason the block can be trained without changing step-0 behavior. At gate 0 the attention weights receive no gradient; the gate does, because its gradient is the upstream gradient times the residual. Once the gate leaves 0, the projections receive gradient. Whether the gate actually opens on ViVQA-S1 is an empirical question. It is not answered here.

## Why this, and not a second router

The escalation head already predicts "the argmax is wrong" from calibrated max-probability, entropy, margin, and question type. It is fit on the calib split only and does not run inside the forward pass. A second confidence head would either repeat those features or, if it were trained to detect errors, would be a place where gold could leak into an inference decision. The user constraint is that gold never enters inference features or routing.

The text half of `h` being image-invariant is a property of the current forward, visible in the code, and it affects every head. Making that half depend on the visual tokens is the change that matches the architecture. It does not by itself detect a confidently wrong answer. After a real training run, the existing calib-only escalation head and `scripts/viso_vision_check.py` (image shuffle on dev) are the measurements for that. This patch does not modify them.

## Parameters and compute

The block is four linear maps (query, key, value, output), one LayerNorm, and the gate. With `D_MODEL` 512 and the existing self-attention head count 8 (`D_MODEL` must be divisible by the head count):

| piece | parameters |
|---|---|
| four `Linear(512, 512)` (weight and bias) | 4 × (512×512 + 512) = 1,050,624 |
| LayerNorm | 1,024 |
| gate | 1 |
| **total** | **1,051,649** |

They live on `encoder.tv_ground.*` and use `TRAINING.LR_ENCODER`. `VISION_PATH_AS_HEADS` does not move them. `utils/viso_efficiency.py` reports them as `text_vision_grounding` when the flag is on, and omits that key when it is off. The flag-off state dict has no `tv_ground` keys.

Operation count for one question of length 32 against the 65 SigLIP-2 tokens, counting multiply-adds of the four projections and the two attention matmuls: about 5.3×10^7. There is no feed-forward inside the block. The encoder already runs a 3-layer self-attention stack and a 3-layer guided-attention stack with feed-forward width 2048. This count is an operation tally for those shapes, not a device benchmark and not a latency measurement.

## Checkpoint compatibility

This is the limitation.

- **Flag off** (every existing config, including `configs/viso_siglip2_bami.yaml` after this change). The module is not constructed. Parameter names and the eval/train forward match the previous encoder. `TypedDecisionTask` loads checkpoints with the default `strict=True`. An existing baseline checkpoint still loads.
- **Flag on, from scratch.** New keys `encoder.tv_ground.q_proj`, `k_proj`, `v_proj`, `out_proj`, `norm`, and `gate`. Because that module is initialized under `fork_rng`, every other parameter matches a same-seed flag-off model, and step 0 matches until the gate moves. A baseline checkpoint is still missing the new keys, so the stock strict load fails. A grounding checkpoint loaded into a flag-off model fails the other way (unexpected keys). Do not mix the two with the training entry point.
- **Non-strict load, eval only.** Loading a baseline `state_dict` with `strict=False` leaves `tv_ground` at its initialization. The gate is 0, so the eval forward matches the baseline. The stock trainer does not do this. It is a manual step if someone later wants to fine-tune a baseline checkpoint with the block, and it still has to be trained before the gate moves. Train-mode dropout masks of the rest of the network stay aligned with the baseline only while the gate is 0, because the block itself draws no random numbers.

`configs/viso_vivqav2.yaml` and `configs/viso_flat_vivqav2.yaml` are unchanged. The code default is false when the key is absent, so those published configs keep the previous behavior. `--opts MODEL.TEXT_VISION_GROUNDING True` works only on a yaml that already contains the key (yacs rejects new keys). The SigLIP-2 ViSO and FlatS1 configs, and `configs/viso_ft_bami.yaml`, contain the key set to false. The BUTD config does not: that encoder is not `ViSOEncoder`.

## Commands

Same data, same loss, same learning rates, same early stopping. The only architectural difference is the flag. Run names differ so the outputs do not overwrite each other. `OUTPUT_PATH` stays `output/viso`, and the task writes to `output/viso/<MODEL.NAME>`.

Baseline:

```bash
export PYTHONDONTWRITEBYTECODE=1
python train_viso.py --config-file configs/viso_siglip2_bami.yaml \
  --opts MODEL.NAME bami_baseline MODEL.DEVICE cuda
```

Grounded question stream, one override:

```bash
python train_viso.py --config-file configs/viso_siglip2_bami.yaml \
  --opts MODEL.NAME bami_tvground MODEL.TEXT_VISION_GROUNDING True MODEL.DEVICE cuda
```

The same grounded run from the example file (`configs/viso_siglip2_bami_tvground.yaml` is the baseline file with `MODEL.NAME` `bami_tvground` and `MODEL.TEXT_VISION_GROUNDING` true):

```bash
python train_viso.py --config-file configs/viso_siglip2_bami_tvground.yaml \
  --opts MODEL.DEVICE cuda
```

CPU smoke of the wiring only, not a result (limit is a type-balanced subset; this does not measure accuracy):

```bash
python train_viso.py --config-file configs/viso_siglip2_bami_tvground.yaml \
  --opts MODEL.DEVICE cpu TRAINING.EPOCHS_SUP 1 TRAINING.PATIENCE 0 \
  DATASET.FEATURE_DATASET.LIMIT 12
```

Do not launch the full run until the packed SigLIP-2 features and the BamiBERT cache are the ones the config points at (`DATASET.FEATURE_DATASET.PACKED_PATH`, `TEXT_CACHE`). This repository checkout does not ship those files.

## Controlled ablation

Do this once, on the frozen splits. Do not tune on test. Do not pick a checkpoint by the Image-Unseen slice.

1. Two runs, seeds 0, 1, and 2 if a mean is reported. Same yaml. The grounded run adds only `MODEL.TEXT_VISION_GROUNDING True`. Leave `FUSION_PRODUCT`, `CHOICE_CROSS_ATTENTION`, `SCORE_COUNT_POOL`, and `PRETRAINED_RESIDUAL` as they are in `configs/viso_siglip2_bami.yaml` (all false). Turning them on in the same comparison confounds the flag.
2. The task already selects the checkpoint with the best dev primary NLL. Keep that rule. Dev is the full dev split, not the Image-Unseen subset. Image-Unseen is an image-level holdout defined by the benchmark flags. This model has no path that reads image ids, so it cannot move an image across that cut. The training code is unchanged and still reads train for updates, calib only inside `scripts/viso_calibrate.py`, and dev for selection.
3. Fit temperatures and the escalation head on calib only, after training:

```bash
python scripts/viso_calibrate.py --run-dir output/viso/bami_baseline
python scripts/viso_calibrate.py --run-dir output/viso/bami_tvground
```

4. Report dev before looking at test. Then report test once, all items and the Image-Unseen view separately:

```bash
python scripts/viso_report.py --split dev \
  --runs baseline=output/viso/bami_baseline:pertype grounded=output/viso/bami_tvground:pertype \
  --out output/viso/tvground_dev.md
python scripts/viso_report.py --split test --view all \
  --runs baseline=output/viso/bami_baseline:pertype grounded=output/viso/bami_tvground:pertype \
  --out output/viso/tvground_test_all.md
python scripts/viso_report.py --split test --view image_unseen \
  --runs baseline=output/viso/bami_baseline:pertype grounded=output/viso/bami_tvground:pertype \
  --out output/viso/tvground_test_image_unseen.md
```

5. On dev, not on test, check that the grounded model still uses the image. `scripts/viso_vision_check.py` shuffles region features within a batch. A useful outcome is a larger image-shuffle drop than the baseline, with the paired interval the script already prints. A zero drop means the gate did not make `h` use the image. The `ZERO_VISION` control (same flag, `MODEL.ZERO_VISION True`) must ignore image content; the unit test covers that on synthetic inputs, and a real run would confirm it on dev.
6. Record parameter count and batch-1 FLOPs with `python -m utils.viso_efficiency` on the two configs. Quote those numbers from that run. Do not quote the operation tally above as a measured speed.

Adopt the flag only if dev NLL improves under this protocol and the image-shuffle check agrees. Report test numbers either way, once. A gain on test that was not visible on dev is not a result of this ablation.

## What still needs real data

- Whether the zero-init gate leaves 0, and whether dev NLL of yes/no, count, and choice moves. All three heads see `h`. The unit tests do not measure that.
- The image-shuffle drop on dev, including the yes/no slice. Report N4 in `docs/REPORT_NOTES_viso.md` says small models sit near chance on yes/no in Vietnamese and in English. This block does not override that finding.
- Wall-clock per step at batch 64. The operation tally is not a latency measurement.
- A strict-load error is expected if a baseline `best_model.pth` is pointed at a grounding run. That is the compatibility limitation above, not a training bug.
- Image-Unseen metrics, from the report command, after dev selection. They are not a tuning target.
