# REPORT NOTES - ViSO (Phase 2c)

## N1. Padding-mask defect in the published constituent module (found 2026-10-09)
File: `models/constituent_mcan/constituent_module.py` (the ViCompreVQA constituent block, shared by `ConstituentMCAN`).
Two defects:
1. `GroupAttention.forward` receives the ADDITIVE float text mask (0 = real token, -1e5 = padding) and uses it as a boolean in `torch.logical_and(mask, neighbour_matrix)`. Non-zero (padding) counts as True and 0 (real) as False, so the neighbour attention is inverted: the columns of real tokens are filled with -1e4 and the padding columns are kept. The group probabilities therefore do not depend on the question content.
2. `ScaledDotProductAttention.forward` calls `att.masked_fill(attention_mask == 0, -1e4)` without assigning the result (not in-place), so the key mask has no effect and every token attends to the padding positions (about 20 of the 32 positions of a question).
The self-attention and guided-attention encoders (`models/modules/attentions.py`) add the additive mask correctly; only this block is affected.
Fix (opt-in): `MODEL.STRICT_TEXT_MASK: True` passes a BOOL mask (True = real) to the block, applies it in both places and masks the text pooling. Default False in the code = the published behaviour (kept reproducible; `tests/viso/test_bamibert.py::test_default_constituent_masks_keep_the_published_behaviour_and_strict_masks_fix_them` documents the leak and checks the fix). ON in all SigLIP 2 configs.
Effect (3-epoch probes, dev): with the same low learning rate the strict mask lifts score 0.23-0.26 -> 0.35 (balanced loss) and keeps choice at 0.30-0.34; yes/no is unchanged (still ~chance).

## N2. The default learning rate collapses ViSO on frozen SigLIP 2 features
LR_ENCODER 2e-4 / LR_HEADS 1e-3 (warmup 500) leaves every type at the prior (train loss = prior, fused vector nearly constant, pairwise cosine 0.98). LR_ENCODER 5e-5 / LR_HEADS 5e-4 (warmup 200) learns score / choice (probes in `~/data/viso_diag`).

## N3. Benchmark facts that matter for reading the numbers
- Question-only naive Bayes on the yes/no items (`scripts/s1_nb_question_only.py`): 0.53 on dev / 0.53 on test in Vietnamese AND 0.53 / 0.53 in English (v1 and v2 alike; 'true' share 0.50-0.51). Yes/no cannot be answered from the question text.
- B-MLP (question word bag x pooled SigLIP 2 image vector, `scripts/s1_bmlp_baseline.py`, 3 seeds, epoch chosen by best DEV soft-CE, T = 1) is the honest simple-model FLOOR. Test, All view, mean of 3 seeds:
  all 0.355 | yes/no 0.510 | score 0.281 | choice 0.251 (Core 0.355 / 0.510 / 0.280 / 0.250; Image-unseen 0.356 / 0.515 / 0.258 / 0.250). Files: `~/data/s1_sanity/bmlp_results.{json,md}`.
- CAVEAT: figures quoted earlier in the session for the simple model (yes/no 0.54-0.58, score 0.33, choice 0.38-0.41) were dev ACCURACY at late epochs (7-12) of a fixed 12-epoch run, past the NLL minimum (accuracy kept creeping up while dev NLL got worse). They are NOT comparable to a model selected by dev NLL and must not be quoted as the floor.
- The 0.56 / 0.33 / 0.38 "bar" used in the collapse probes is a diagnostic target only (the late-epoch accuracies above). The mechanistic success criteria for yes/no are: (a) yes/no train loss leaves ln 2 (< 0.68); (b) dev yes/no NLL < 0.690; (c) the image-shuffle drop on yes/no is > 0 with a paired-bootstrap 95% CI excluding 0 (`scripts/viso_vision_check.py`, 5 shuffles per item, 2000 resamples).

## N4. Yes/no is at chance for small models in ENGLISH too (benchmark property, not a translation effect)
Same items, English source questions (`meta.en_q`): question-only naive Bayes dev / test 0.527 / 0.533 (Vietnamese 0.530 / 0.527); English B-MLP (`scripts/s1_bmlp_baseline.py --lang en`, 3 seeds, epoch by dev soft-CE) test yes/no 0.502 / 0.515 / 0.496 (mean 0.504; Vietnamese 0.510). Hence the near-chance yes/no of every small model (ViSO and baselines) is a property of the benchmark (VQA v2 yes/no balancing - complementary pairs - plus coarse frozen features), not of the Vietnamese translation. State it as a finding; do not chase it.

## N5. BamiBERT cached features need standardisation
The raw BamiBERT last-layer candidate vectors have a large common component (57% of the squared norm is the per-dimension mean; within-question candidate cosine 0.59). With the plain Linear + LayerNorm projection the choice head collapsed (B1b choice 0.142 while score improved). `DATASET.FEATURE_DATASET.TEXT_CACHE_STANDARDIZE: True` (per-dimension z-score, std floored at 0.05) is on in the *_bami configs.

## N6. Order-permuted candidate lists (benchmark v2.0.2)
Candidate sets contained order-permuted equivalents ('đỏ và trắng' / 'trắng và đỏ'): train 6.4% of the choice items (1,005; 144 involve the gold), calib 5.7% (98), dev 7.7% (166; 23 involve the gold), test 6.3% (139; 16 involve the gold). Test (not yet frozen) was fixed at build time (merged mass, no permuted distractors, v2.0.2). The FROZEN train / calib / dev keep their sets: report them with the eval-time equivalence (`evaluate.py --equivalence`, `s1bench/equivalence.py`: the masses of permuted candidates are summed and a prediction in the gold's class is correct). Effect on a real prediction file (B-MLP seed 0, dev, choice): accuracy 0.2684 -> 0.2712, NLL 2.4229 -> 2.4123 (small, and it can only help, since it removes spurious misses). Models trained on train (which still contains the permuted pairs) are unaffected; the test numbers use the merged sets.
Near-synonym pairs (not merged; `annotation/near_synonym_pairs_top30.md`): the top pairs are hypernym / hyponym or wording pairs ('nâu' / 'vàng nâu', 'lướt sóng' / 'ván lướt sóng', 'mũ' / 'mũ bảo hiểm', 'diều' / 'thả diều'); 'đất' / 'mặt đất' co-occur in 12 items.
Or-questions with a yes/no gold ('có' / 'không') exist only among UNJOINED items (no English original to repair them); flag `or_question_yes_no_gold` (test 2, dev 4, calib 1, train 6); the two test ones are in Sheet A2.

## N7. Extra yes/no data did not help (learning curve and X7K, round 3)
Yes/no train subsets (25% / 50% / 100% of the 12.6k real yes/no items, same steps, best-dev-NLL selection): best dev yes/no NLL 0.693 / 0.689 / 0.689, i.e. the learning curve is flat from 50% to 100%. Adding the 7,376 already-translated real VQA v2 yes/no questions (X7K, +58% yes/no data) did not improve the overall result either (best overall dev NLL 1.409 vs 1.346 for the same config without them, accuracy 0.438 vs 0.462). More translated yes/no data is therefore NOT the lever; the X-YN / X-NUM translation runs were stopped (the 7,380 translations and the pilot outputs are kept in vivqa_s1_verify/v2/raw/xyn, DeepSeek spend $0.135 in total).
