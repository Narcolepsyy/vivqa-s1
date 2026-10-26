# ViVQA-S1: System-One Typed Decision Architecture & Benchmarking for Vietnamese Visual Question Answering

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Narcolepsyy%2Fvivqa--s1-black.svg?logo=github)](https://github.com/Narcolepsyy/vivqa-s1)

**ViVQA-S1** is an efficient **System-One ($O(1)$ single forward pass)** visual question answering framework and benchmark tailored for Vietnamese. Unlike standard Vision-Language Models (VLMs) that perform slow autoregressive decoding ($O(L)$ steps), ViVQA-S1 parametrizes VQA as a **typed decision task**—achieving high accuracy, low latency, and formal confidence calibration with selective fallback.


---

## ⚡ Key Highlights

- **$O(1)$ Single Forward Pass**: Decides the answer in a single forward pass without autoregressive token generation.
- **Typed Decision Heads**: Specialized decision interfaces conditioned on the question category:
  - 🔘 `noul` (**Boolean / Verification**): Binary cross-entropy with learned prior.
  - 🔢 `score` (**Counting / Numeric**): Ordinal regression and distribution matching ($0 \dots 10+$).
  - 🔠 `choice` (**Candidate Selection**): 16-way candidate matching powered by `SetAttention` and token-level `Candidate-to-Visual Cross-Attention`.
- **Confidence Calibration & Selective Prediction**: Post-hoc temperature scaling ($T_{\text{noul}}, T_{\text{score}}, T_{\text{choice}}$) and an **escalation head** that selectively defers uncertain questions to a heavy System-Two generative VLM.
- **Reinforcement Learning from Candidate Distributions (RLCD)**: Dual-stage training with soft cross-entropy and policy-gradient fine-tuning.
- **Multi-Modal Vision Backbones**: Built-in support for **SigLIP-2**, **BaMI (Vietnamese BERT)**, **VinVL**, and **ViT-mBERT**.

---

## 📁 Repository Structure

```text
ViVQA_S1/
├── configs/                  # YAML configurations for all ViSO variants
│   ├── viso_siglip2_bami.yaml            # SigLIP-2 vision + BaMI text encoder
│   ├── viso_siglip2_bami_crossattn.yaml  # With Candidate Cross-Attention
│   ├── viso_flat_siglip2.yaml            # Flat classifier baseline (A0)
│   └── viso_vivqav2.yaml                 # Base ViVQA v2 benchmark config
├── models/                   # Neural network architectures
│   ├── viso/                             # ViSO core modules
│   │   ├── viso_model.py                 # Full ViSO model & FlatS1 wrapper
│   │   ├── encoder.py                    # Multimodal fusion & token unpooling
│   │   ├── heads.py                      # Noul, Score, Choice heads (+CrossAttn)
│   │   └── viso_ft.py                    # End-to-end fine-tuning module
│   ├── mcan.py                           # Modular Co-Attention baseline
│   ├── m4c.py                            # Multimodal Multi-Copy baseline
│   └── modules/                          # Attention, embeddings & fusion layers
├── tasks/                    # Task runners & training loops
│   ├── typed_decision_task.py            # TypedDecisionTask with RLCD & soft-CE
│   └── base_task.py                      # Abstract training task
├── scripts/                  # Training, evaluation & utility scripts
│   ├── train_viso_ft.py                  # End-to-end ViSO training script
│   ├── s1_vlm_eval.py                    # VLM System-One zero-shot & head-only evaluator
│   ├── s1_vlm_lora.py                    # VLM System-One LoRA + 16-option head trainer
│   ├── viso_probe.py                     # Controlled learning probes (400 steps)
│   ├── viso_calibrate.py                 # Post-hoc temperature calibration
│   ├── viso_predict.py                   # Model inference & prediction dumper
│   ├── viso_report.py                    # Multi-view evaluation reporter
│   ├── viso_pack_features.py             # High-speed float16 feature packer
│   └── kaggle_prepare.py                 # Kaggle dataset packaging utility
├── kaggle/                   # Automated Kaggle/Colab GPU experiment pipelines
│   ├── experiments.json                  # Experiment matrix (A0 - A6)
│   ├── experiments_siglip2.json          # SigLIP-2 experiment manifests
│   ├── viso_matrix_kernel.py             # Multi-GPU ViSO matrix runner
│   └── vlm_kernel.py                     # Multi-VLM System-One runner (Qwen, MiniCPM, Vintern)
├── data_utils/               # Tokenizers, vocabs, and feature collators
│   ├── datasets/s1_dataset.py            # PyTorch S1FeatureDataset & collator
│   ├── vocabs/s1_vocab.py                # S1 vocabulary and candidate mapper
│   └── vncorenlp/                        # VnCoreNLP Vietnamese toolkit
├── utils/                    # Calibration, packed features & RLCD helpers
│   ├── viso_calibration.py               # ECE, NLL, calibration optimization
│   ├── viso_packed_features.py           # Memory-mapped binary feature reader
│   └── viso_rlcd.py                      # RLCD loss computation
└── tests/viso/               # Pytest unit & regression test suite
```

---

## 📊 ViVQA-S1 Benchmark Dataset

The model is evaluated on **ViVQA-S1**, a clean, audited Vietnamese Visual Question Answering dataset with full type annotations:

| Split | Total Items | `choice` | `noul` | `score` | Images |
|---|---|---|---|---|---|
| **Train** | 31,647 | 15,392 | 12,603 | 3,652 | 24,964 |
| **Calib** | 3,455 | 1,694 | 1,376 | 385 | 2,755 |
| **Dev** | 4,392 | 2,117 | 1,770 | 505 | 3,466 |
| **Test** | 4,409 | 2,157 | 1,723 | 529 | 3,438 |
| **Total** | **43,903** | **21,360** | **17,472** | **5,071** | **34,623** |

*Evaluation Views*:
- **All**: Complete test split (comparable with standard ViVQA lineages).
- **Core**: Evaluated on verified, clean question-answer pairs.
- **Image-Unseen**: Strict generalization benchmark on images never seen during training or calibration.

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Narcolepsyy/vivqa-s1.git
cd vivqa-s1

python3 -m venv .venv
source .venv/bin/activate
pip install -r <(echo "
torch>=2.0.0
transformers>=4.30.0
yacs>=0.1.8
pytest>=7.0.0
numpy
pyyaml
tqdm
tabulate
termcolor
pillow
")
```

### 2. Run Unit Tests

Verify that all modules, loss functions, and dataset pipelines pass:

```bash
pytest tests/viso -v
```

### 3. Training ViSO

To train a ViSO model with SigLIP-2 visual features and BaMI text representations (F2 recipe):

```bash
python train_viso.py \
    --config-file configs/viso_siglip2_bami.yaml \
    --opts MODEL.DEVICE cuda \
           MODEL.NAME viso_siglip2_run \
           LOSS.NAME soft_ce \
           TRAINING.EPOCHS_SUP 12 \
           TRAINING.LR_ENCODER 5e-5 \
           TRAINING.LR_HEADS 5e-4 \
           TRAINING.WARMUP_STEPS 200 \
           MODEL.FUSION_PRODUCT True \
           TRAINING.VISION_PATH_AS_HEADS True \
           TRAINING.TEXT_PROJ_AS_HEADS True \
           MODEL.STRICT_TEXT_MASK True
```

### 4. Calibration & Selective Prediction

Optimize post-hoc temperatures on the calibration set (`calib`) to minimize Expected Calibration Error (ECE):

```bash
# 1. Generate predictions and raw logits
python scripts/viso_predict.py \
    --config-file configs/viso_siglip2_bami.yaml \
    --opts MODEL.NAME viso_siglip2_run

# 2. Fit temperature scaling per question type
python scripts/viso_calibrate.py \
    --run-dir output/viso/viso_siglip2_run
```

### 5. Multi-View Benchmark Evaluation

Generate paired bootstrap confidence intervals and comprehensive metric reports:

```bash
python scripts/viso_report.py \
    --runs ViSO=output/viso/viso_siglip2_run:pertype \
    --split test \
    --out results_test.md
```

### 6. System-One VLM Zero-Shot & LoRA Fine-Tuning

Evaluate external VLMs (`Qwen/Qwen3.5-0.8B`, `vllm-sr/d3-lite`, `openbmb/MiniCPM-V-4.7-1B`, `Qwen/Qwen3.5-2B`, `vllm-sr/d3-nano`) in single-forward-pass System-One mode and fine-tune with Vision+Language LoRA (`--mode img`) + 16-option decision head:

```bash
python scripts/s1_vlm_lora.py \
    --mode img \
    --model vllm-sr/d3-nano \
    --revision 80820624a8170104679aa99897a6e9026eae0790 \
    --licence apache-2.0 \
    --model-dir output/vlm_d3_nano \
    --s1-data ../vivqa_s1_v2/data \
    --root ../vivqa_s1_v2 \
    --save-zeroshot \
    --batch 8 --accum 4 --epochs 1.0
```

---

## 🔬 Model Architectures & ViSO Ablations

| Variant | Vision Backbone | Text Backbone | Decision Head | Loss / Training | Test Acc | Core Acc | ECE | Latency |
|---|---|---|---|---|---|---|---|---|
| **A0 (Flat Baseline)** | SigLIP-2 (768d) | BaMI / WordEmb | Flat 3,500-way Softmax | Cross-Entropy | 43.2% | 43.2% | 0.106 | ~3 ms |
| **A1** | SigLIP-2 (768d) | BaMI | Typed Heads (`noul`/`score`/`choice`) | Hard-CE | 43.8% | 43.9% | 0.052 | ~3 ms |
| **A2 (ViSO Base)** | SigLIP-2 (768d) | WordEmb | Typed Heads + SetAttention | Soft-CE | 44.6% | 44.7% | 0.020 | ~3 ms |
| **A2bami (ViSO + BaMI, 3-seed)**| SigLIP-2 (768d) | BaMI BERT (768d) | Typed Heads + SetAttention | Soft-CE + Standardize | **45.6%** (`46.4%` s2) | **45.7%** (`46.6%` s2) | **0.023** | ~3.5 ms |
| **A3 (ViSO + RLCD)** | SigLIP-2 (768d) | BaMI BERT | Typed Heads + SetAttention | Dual-stage RLCD | 45.1% | 45.3% | 0.027 | ~3.5 ms |
| **Cross-Attn Head (v4)** | SigLIP-2 (768d) | BaMI BERT | Question-Cond Cross-Attn + FILIP Max-Patch | Soft-CE (Ep 5 NLL) | **46.3%** | **46.5%** | **0.017** | ~4 ms |
| **Cross-Attn Spliced (v5)** | SigLIP-2 (768d) | BaMI BERT | Per-Type Checkpoint Splicing (`SAVE_BEST_PER_TYPE`) | Soft-CE | **47.7%** (`47.9%` vec) | **47.9%** (`48.1%` vec) | **0.021** | ~4 ms |
| **Cross-Attn Ep8 (v5_ep8)** | SigLIP-2 (768d) | BaMI BERT | Question-Cond Cross-Attn + FILIP (Ep 8 + Calib T) | Soft-CE | **48.7%** | **49.0%** | **0.032** | ~4 ms |
| **ViSO-Hybrid v5 (Single-Pass)** | SigLIP-2 (768d) | BaMI BERT | Type-Routed (`v5_ep8` `noul`/`choice` + `A2bami_s2` `score`) | Soft-CE + Calib T | **49.3%** | **49.6%** | **0.026** | ~3.8 ms |
| **10-ViSO S1 Ensemble** | SigLIP-2 (768d) | BaMI BERT (10-run pool) | Cross-Attn (`v5`/`v4`) + SetAttention + RLCD | Calib Log/Lin Pool + Soft-T | **49.9%** (`49.7%` log) | **50.1%** | **0.017** (dev) | ~16 ms |
| **10-ViSO Cascade (50% Cov)** | SigLIP-2 (768d) | BaMI + d3-nano | Disagreement Escalation + PAVA DP | Selective Routing | **66.4%** | **66.7%** | **0.019** | ~13 ms |
| **10-ViSO Cascade (20% Cov)** | SigLIP-2 (768d) | BaMI + d3-nano | Disagreement Escalation + PAVA DP | Selective Routing | **73.9%** | **74.1%** | **0.024** | ~18 ms |
| **10-ViSO Cascade (10% Cov)** | SigLIP-2 (768d) | BaMI + d3-nano | Disagreement Escalation + PAVA DP | Selective Routing | **75.7%** | **75.9%** | **0.024** | ~19 ms |
| **3-Tier S1 Cascade (10/40/50%)** | SigLIP-2 + 0.85B + 2.2B | 10-ViSO $\rightarrow$ d3-lite $\rightarrow$ d3-nano | 2-Stage Calibrated Escalation | Hierarchical Routing | **75.5%** | **75.7%** | **0.021** | ~16 ms |

---

## 🏆 Benchmark Results on ViVQA-S1 (Test Split)

The table below presents the evaluation results on the **ViVQA-S1 Test Split** (4,409 items), comparing **ViSO Architectures**, **System-One $\rightarrow$ System-Two Cascades**, and **External VLMs in System-One Mode** (evaluated with standardized letter prompt and single-forward-pass letter-token logits).

> **Evaluation Views**:  
> - **All Acc**: Overall accuracy on the full test set.  
> - **Core Acc**: Accuracy on clean, verified items.  
> - **Image-Unseen Acc**: Strict out-of-distribution evaluation on images never appearing in train or calibration splits.  
> - **Per-type breakdown**: `noul` (Boolean yes/no verification), `score` (Counting $0 \dots 10+$), `choice` (16-way candidate selection).

| Model / Architecture | Parameters | Paradigm / Method | All Acc | Core Acc | Image-Unseen | `noul` | `score` | `choice` | ECE | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Multi-VLM S1 Ensemble (9-Model)** | 9-Model Pool | Canonical-Calib Pre-T + Lin/Log Pool | **78.0%** | **78.2%** | **78.7%** | **83.7%** | **73.7%** | **74.6%** | **0.018** | ~0.22s |
| **Multi-VLM S1 Ensemble (14-Model)** | 14-Model Pool | All Canonical-4409 LoRA + Heads + Ref ZS | **77.9%** | **78.1%** | **78.7%** | **83.5%** | **73.7%** | **74.5%** | **0.019** | ~0.28s |
| **10-ViSO $\rightarrow$ 9-VLM Cascade (10% Cov)** | ~106 M + VLM Pool | System-1 (`10-ViSO`) $\rightarrow$ `9-VLM Pool` | **77.7%** | **77.9%** | **78.3%** | **83.5%** | **73.3%** | **73.8%** | **0.018** | ~0.19s |
| **Multi-VLM S1 Ensemble (5-Model)** | 5-Model Pool | Calib-Weighted Lin/Log Pool | **77.1%** | **77.3%** | **77.6%** | **82.2%** | **72.0%** | **74.3%** | **0.021** | ~0.16s |
| **d3-nano (Decision 3.0 VLM)** | 2.21 B | LoRA (`img+text`) + Head | **76.1%** | **76.3%** | **76.8%** | **81.8%** | **70.3%** | **73.0%** | 0.024 | 0.021s |
| **Qwen3.5-2B (VLM, Canonical 4409)** | 2.27 B | LoRA (`img+text`) + Head | **75.9%** | **76.1%** | **76.4%** | **81.1%** | **70.0%** | **72.9%** | 0.025 | 0.044s |
| **10-ViSO Cascade (10% Cov, 2.2B)** | ~106 M + 2.2B | System-1 (`10-ViSO`) $\rightarrow$ `d3-nano` | **75.7%** | **75.9%** | **76.3%** | **81.9%** | **70.1%** | **71.8%** | 0.024 | ~0.019s |
| **10-ViSO $\rightarrow$ 9-VLM Cascade (20% Cov)** | ~106 M + VLM Pool | System-1 (`10-ViSO`) $\rightarrow$ `9-VLM Pool` | **75.7%** | **76.0%** | **76.3%** | **82.0%** | **71.1%** | **71.2%** | **0.019** | ~0.17s |
| **3-Tier Cascade (10% ViSO / 40% lite / 50% nano)** | ~106 M + 0.85B + 2.2B | `8-ViSO` $\rightarrow$ `d3-lite` $\rightarrow$ `d3-nano` | **75.5%** | **75.7%** | **76.1%** | **81.7%** | **69.9%** | **71.9%** | **0.021** | **~0.016s** |
| **ViSO-Hybrid v5 Cascade (10% Cov, 2.2B)** | ~105 M + 2.2B | System-1 (`ViSO-Hybrid v5`) $\rightarrow$ `d3-nano` | **75.3%** | **75.6%** | **76.0%** | 81.8% | 69.8% | 71.5% | 0.024 | ~0.019s |
| **10-ViSO Cascade (20% Cov, 2.2B)** | ~106 M + 2.2B | System-1 (`10-ViSO`) $\rightarrow$ `d3-nano` | **73.9%** | **74.1%** | **74.4%** | **80.9%** | **68.4%** | **69.0%** | 0.024 | ~0.018s |
| **ViSO-Hybrid v5 Cascade (20% Cov, 2.2B)** | ~105 M + 2.2B | System-1 (`ViSO-Hybrid v5`) $\rightarrow$ `d3-nano` | **73.7%** | **73.9%** | **74.1%** | 80.7% | 68.1% | 68.8% | 0.022 | ~0.018s |
| **MiniCPM-V-4.7-1B (VLM)** | 1.55 B | LoRA (`img+text`) + Head | **73.5%** | **73.6%** | **74.1%** | **80.7%** | **67.7%** | **69.1%** | **0.022** | 0.065s |
| **d3-lite (Decision 3.0 VLM)** | 0.85 B | LoRA (`img+text`) + Head | **72.4%** | **72.5%** | **72.6%** | **78.6%** | **63.9%** | **69.4%** | **0.021** | **0.014s** |
| **Qwen3.5-0.8B (VLM, Canonical 4409)** | 0.85 B | LoRA (`img+text`) + Head | **72.3%** | **72.5%** | **72.8%** | **78.4%** | **65.6%** | **68.9%** | 0.020 | 0.025s |
| **ViSO v4 Cascade (10% Cov, 0.85B)** | ~106 M + 0.85B | System-1 (`ViSO v4`) $\rightarrow$ `d3-lite` | **71.7%** | **71.9%** | **71.9%** | 78.6% | 63.3% | 68.2% | **0.022** | **~0.013s** |
| **10-ViSO Cascade (30% Cov, 2.2B)** | ~106 M + 2.2B | System-1 (`10-ViSO`) $\rightarrow$ `d3-nano` | **71.4%** | **71.6%** | **72.0%** | **75.5%** | **67.9%** | **68.8%** | 0.024 | ~0.016s |
| **ViSO-Hybrid v5 Cascade (30% Cov, 2.2B)** | ~105 M + 2.2B | System-1 (`ViSO-Hybrid v5`) $\rightarrow$ `d3-nano` | **71.4%** | **71.7%** | **71.6%** | 75.2% | 67.7% | 68.9% | 0.022 | ~0.016s |
| *Qwen3-VL-8B (VLM, Debiased / ResMLP OOF)* | 8.77 B | Zero-shot Debiased (`69.7%`) / OOF Head (`0.836 NLL`) | **69.7%** | **70.0%** | **70.4%** | **83.1%** | **70.5%** | **59.6%** | **0.024** | 0.122s |
| **ViSO v4 Cascade (20% Cov, 0.85B)** | ~106 M + 0.85B | System-1 (`ViSO v4`) $\rightarrow$ `d3-lite` | **69.7%** | **69.9%** | **70.0%** | 76.7% | 62.6% | 65.8% | 0.023 | **~0.012s** |
| **d3-nano (Decision 3.0 VLM, Vector Calib)** | 2.21 B | Zero-shot (`vectype`) | **69.2%** | **69.4%** | **69.5%** | **80.5%** | **66.7%** | **59.5%** | **0.025** | 0.021s |
| *Intern-Decision-4B (VLM, 5-Fold OOF Head)* | 4.54 B | $W_0$-Anchored OOF Head (`vectype`, **0.861 NLL**) | **69.1%** | **69.4%** | **70.0%** | **82.0%** | **67.3%** | **59.1%** | **0.029** | 0.082s |
| **d3-nano (Decision 3.0 VLM)** | 2.21 B | Zero-shot | **68.4%** | **68.7%** | **68.7%** | **80.5%** | **65.0%** | **59.5%** | 0.029 | 0.021s |
| *Intern-Decision-4B (VLM)* | 4.54 B | Zero-shot Ref | 68.1% | 68.5% | 69.1% | 81.8% | 63.3% | 58.5% | 0.044 | 0.082s |
| *Decision API (Teacher)* | *Cloud API* | *Reference Teacher* | *68.1%* | *68.5%* | *69.2%* | *81.5%* | *62.2%* | *59.1%* | *0.105* | *~0.800s* |
| *Intern-Decision-2B (VLM, 5-Fold OOF Head)* | 2.21 B | $W_0$-Anchored OOF Head (**0.987 NLL**) | **66.5%** | **66.8%** | **67.2%** | **78.4%** | **59.0%** | **58.0%** | **0.024** | 0.047s |
| **10-ViSO Cascade (50% Cov, 2.2B)** | ~106 M + 2.2B | System-1 (`10-ViSO`) $\rightarrow$ `d3-nano` | **66.4%** | **66.7%** | **67.3%** | **70.8%** | **64.1%** | **63.6%** | **0.019** | ~0.013s |
| **ViSO-Hybrid v5 Cascade (50% Cov, 2.2B)** | ~105 M + 2.2B | System-1 (`ViSO-Hybrid v5`) $\rightarrow$ `d3-nano` | **65.7%** | **66.0%** | **66.3%** | 69.8% | 63.8% | 62.9% | **0.019** | ~0.012s |
| *Intern-Decision-2B (VLM)* | 2.21 B | Zero-shot Ref | 64.9% | 65.3% | 65.8% | 77.6% | 57.2% | 56.9% | 0.025 | 0.047s |
| **Qwen3.5-2B (VLM)** | 2.27 B | Zero-shot | 64.4% | 64.6% | 64.9% | 69.2% | 61.4% | 62.6% | 0.049 | 0.044s |
| **OneJev-0.8B (VLM, Canonical 4409)** | 0.85 B | $W_0$-Anchored Head (`head_s1`, **1.069 NLL**) | **61.8%** | **62.1%** | **62.4%** | **74.8%** | **61.3%** | **51.3%** | 0.032 | 0.029s |
| **ViSO v4 Cascade (50% Cov, 0.85B)** | ~106 M + 0.85B | System-1 (`ViSO v4`) $\rightarrow$ `d3-lite` | **61.7%** | **61.8%** | **62.0%** | 65.9% | 58.6% | 59.1% | 0.023 | **~0.009s** |
| **d3-lite (Decision 3.0 VLM)** | 0.85 B | Zero-shot | **61.3%** | **61.6%** | **62.3%** | **73.1%** | **55.6%** | **53.2%** | 0.035 | **0.014s** |
| **MiniCPM-V-4.7-1B (VLM)** | 1.55 B | Zero-shot | **60.8%** | **61.2%** | **62.4%** | **72.4%** | **56.7%** | **52.6%** | **0.027** | 0.065s |
| **OneJev-0.8B (VLM, Debiased)** | 0.85 B | Zero-shot (`vectype`) | **60.7%** | **61.0%** | **61.3%** | 74.4% | 56.1% | 50.8% | 0.033 | 0.029s |
| **jpt-0.8B (VLM, Debiased)** | 0.85 B | Zero-shot (`vectype`) | **60.5%** | **60.9%** | **60.9%** | 72.1% | 56.4% | 50.2% | 0.037 | 0.029s |
| **jpt-0.8B (VLM, Canonical 4409)** | 0.85 B | $W_0$-Anchored Head (`head_s1`, **1.208 NLL**) | **59.9%** | **60.3%** | **60.5%** | 74.1% | 59.2% | 50.1% | 0.026 | 0.029s |
| **Intern-Decision-0.8B (VLM, Canonical 4409)** | 0.85 B | $W_0$-Anchored Head (`head_s1`, **1.215 NLL**) | **59.8%** | **60.1%** | **60.9%** | 73.5% | 57.2% | 50.3% | 0.030 | 0.029s |
| **Intern-Decision-0.8B (VLM, Debiased)** | 0.85 B | Zero-shot (`vectype`) | **59.1%** | **59.5%** | **60.0%** | 72.5% | 51.8% | 49.9% | 0.029 | 0.029s |
| **Qwen3.5-0.8B (VLM, Canonical 4409)** | 0.85 B | $W_0$-Anchored Head (`head_s2`, **1.251 NLL**) | **59.0%** | **59.4%** | **59.8%** | 72.5% | 56.8% | 50.4% | 0.028 | 0.029s |
| **Qwen3.5-0.8B (VLM, Debiased)** | 0.85 B | Zero-shot (`vectype`) | **57.6%** | **58.0%** | **58.5%** | 70.5% | 45.2% | 49.2% | 0.029 | 0.029s |
| **Vintern-1B-v3.5 (VLM, Canonical 4409)** | 0.94 B | $W_0$-Anchored Head (`head_s0`, **1.234 NLL**) | **56.8%** | **57.0%** | **57.6%** | 69.7% | 51.3% | 48.6% | 0.032 | 0.070s |
| **Vintern-1B-v3.5 (VLM, Letter-Debiased)** | 0.94 B | Zero-shot (`--debias-letters`) | **55.8%** | **56.0%** | **56.3%** | 69.1% | 48.0% | **47.2%** | 0.044 | 0.070s |
| **Vintern-1B-v3.5 (VLM)** | 0.94 B | Zero-shot | 53.3% | 53.5% | 53.8% | 69.1% | 48.0% | 42.3% | 0.046 | 0.070s |
| **ViSO System-One Ensemble (10-Model)** | ~106 M (x10) | `Cross-Attn v5` + `v4` + `A2bami` + `RLCD` + `FT` | **49.9%** (`49.7%` log) | **50.1%** | **50.2%** | **56.0%** | **35.2%** | **48.8%** | **0.017** (dev) | ~0.016s |
| **ViSO-Hybrid v5 (Single-Pass Routed)** | ~105 M | `v5_ep8` (`noul`/`choice`) + `A2bami_s2` (`score`) | **49.3%** | **49.6%** | **49.8%** | **56.7%** | **37.8%** | **46.2%** | **0.026** | ~0.004s |
| **ViSO (Cross-Attn v5_ep8)** | ~106 M | Question-Cond Cross-Attn + FILIP (Ep 8 + Calib T) | **48.7%** | **49.0%** | **49.4%** | **56.7%** | **33.1%** (`34.2%` vec) | **46.2%** | **0.032** | ~0.004s |
| **ViSO (Cross-Attn v5 Spliced)** | ~106 M | Per-Type Best Checkpoint Splicing (`SAVE_BEST_PER_TYPE`) | **47.7%** (`47.9%` vec) | **47.9%** (`48.1%` vec) | **48.3%** (`48.6%` vec) | **53.6%** | **34.4%** (`35.5%` vec) | **46.3%** | **0.021** | ~0.004s |
| **ViSO (A2bami_s2 Peak Seed)** | ~103 M | Typed System-1 (BaMI+SigLIP2) | **46.4%** | **46.6%** | **46.5%** | **55.3%** | **37.8%** | **41.4%** | **0.031** | ~0.003s |
| **ViSO (Cross-Attn v4)** | ~106 M | Question-Cond Cross-Attn + FILIP | **46.3%** | **46.5%** | **46.6%** | **53.2%** | **33.3%** (`34.2%` vec) | **43.9%** | **0.017** | ~0.004s |
| **ViSO (A2bami: BaMI+SigLIP2, 3-seed)**| ~103 M | Typed System-1 | **45.6%** | **45.7%** | **45.9%** | **54.5%** | **36.5%** | **40.9%** | **0.023** | ~0.003s |
| **ViSO (A2: SigLIP2 Base)** | ~103 M | Typed System-1 | **44.6%** | **44.7%** | **44.9%** | 52.9% | 33.5% | 40.8% | **0.020** | ~0.003s |
| **FlatS1 (A0 Baseline)** | ~100 M | Flat Softmax | 43.2% | 43.2% | 43.4% | 52.0% | 30.3% | 39.4% | 0.106 | ~0.003s |
| *ViSO Zero-Vision (A5)* | ~103 M | Text-only Control | 40.8% | 40.8% | 41.0% | 51.1% | 26.8% | 36.2% | 0.033 | ~0.002s |
| *B-MLP (Simple Floor)* | ~15 M | Question Bag x Image | 35.5% | 35.5% | 35.6% | 51.0% | 28.0% | 25.0% | - | <0.001s |

### 📌 Benchmark Insights
1. **Per-Type Checkpoint Dynamics (`ViSO v5`), Single-Pass `ViSO-Hybrid v5` (`49.3% All / 49.8% Image-Unseen`), & `10-ViSO` Ensemble (`49.9% All / 50.2% Image-Unseen`)**:
   - **Why `ViSO v4` stopped early (`46.25%` Test)**: Selecting `best_model.pth` by *uncalibrated* overall `dev_nll` stopped `ViSO v4` at Epoch 5 because logit sharpening ($T < 1$) increased uncalibrated NLL even as `choice` accuracy kept climbing from `42.56%` (Epoch 5) to **`46.24%` (Epoch 8)** and **`47.19%` (Epoch 10)** on `Dev`.
   - **Standalone `ViSO v5_ep8` (`48.72% All / 49.00% Core / 49.38% Image-Unseen`)**: Evaluating Epoch 8 with post-hoc per-type calibration (`viso_calibrate.py`) boosts single-model ViSO accuracy by **+2.47%** (`56.65% noul`, `33.08% score`, **`46.22% choice`** — a **+2.36% `choice` gain** and **+2.20% `noul` gain** over `ViSO v4`).
   - **Single-Pass Type-Routed `ViSO-Hybrid v5` (`49.29% All / 49.57% Core / 49.81% Image-Unseen`)**: Routing `noul` (`56.65%`) and `choice` (`46.22%`) to `v5_ep8` and `score` (`37.81%`) to `A2bami_s2` achieves **49.29% `Test All` (`+3.04%` over `ViSO v4`)** while executing only **1 ~105M ViSO model per question (`2.39 GFLOPs`, ~3.8 ms)**.
   - **10-ViSO Calibrated Ensemble (`50.93% Dev / 49.90% Test All / 50.13% Core / 50.19% Image-Unseen`)**: Combining `v5_spliced`, `v5_ep8`, `v4`, `A2bami_s0..s2`, `A3fix_s0..s2`, and `ViSOFT` crosses the **50% accuracy milestone** on `Dev All` (`50.93%`), `Test Core` (`50.13%`), and `Test Image-Unseen` (`50.19%`) (`55.8% noul`, `35.2% score`, **`48.8% choice`**, **1.168 NLL**, **0.0165 Dev ECE**).
2. **Selective Escalation (`10-ViSO Ensemble` & `ViSO-Hybrid v5` $\rightarrow$ `vllm-sr/d3-nano`, `d3-lite`, & `9-VLM`)**: Incorporating type-interaction terms (`oh * ent`, `oh * margin`), ensemble member disagreement (`std_top`, `vote_agree`), and **PAVA Least Concave Majorant smoothing** before dynamic-programming budget allocation ([scripts/viso_cascade.py](file:///home/khaitran/project/IE307-Final/ViVQA_S1/scripts/viso_cascade.py)):
   - **Upgraded 2-Tier Cascade (`10-ViSO` $\rightarrow$ `d3-nano`, 2.21B)**: Reaches **75.7% accuracy** (**76.3%** on `Image-Unseen`) at **10% coverage**, **73.9%** (**74.4%** `Image-Unseen`) at **20% coverage**, **71.4%** (**72.0%** `Image-Unseen`) at **30% coverage**, and **66.4%** (**67.3%** `Image-Unseen`, **0.019 ECE**) at **50% coverage** (+2.3% `All` / +2.5% `Image-Unseen` over single-model `ViSO v4` cascade at 50% budget), retaining $\ge 95\%$ of `d3-nano`'s accuracy at **27% System-One coverage**.
   - **Single-Pass `ViSO-Hybrid v5` $\rightarrow$ `d3-nano` Cascade**: Reaches **75.3% `All`** at 10% coverage, **73.7%** at 20% coverage (+1.1% over `ViSO v4` cascade), **71.4%** at 30% coverage (+1.4% over `ViSO v4`), and **65.7%** at 50% coverage (+1.6% over `ViSO v4`) with zero ensemble overhead.
3. **Option-Order Letter Debiasing, $W_0$-Anchored OOF Readout Adaptation, & Fixed-Space Class-Prior Calibration**:
   - Subtracting the empirical `calib` letter-position prior $b_{\text{pos}} = \frac{1}{2}(\bar{\ell}^{(1)} + \bar{\ell}^{(2)})$ *before* the non-linear `log_softmax` in [`to_label_logits`](file:///home/khaitran/project/IE307-Final/ViVQA_S1/scripts/s1_vlm_eval.py#L427-L438) and training 5-fold Out-of-Fold (OOF) $W_0$-anchored readout heads ([`kaggle/colab_heavy_boost.py`](file:///home/khaitran/project/IE307-Final/ViVQA_S1/kaggle/colab_heavy_boost.py)) boosts `Intern-Decision-4B` to **69.06% `Test All`** (`+3.15%` over raw zero-shot, cutting NLL to **0.8607**), `Intern-Decision-2B` to **66.45% `Test All`** (`+2.83%`, cutting NLL to **0.9871**), and cuts `Qwen3-VL-8B` Test NLL from `0.8847` to **0.8361** (`head_res_mlp_s0`).
   - Fitting an L2-regularized class-prior bias vector $z / T_t + b_t$ (`vectype` in [`utils/viso_calibration.py`](file:///home/khaitran/project/IE307-Final/ViVQA_S1/utils/viso_calibration.py#L99-L127)) on `calib` for fixed-space types (`noul` $K=2$, `score` $K=11$) boosts single-model `score` accuracy by **+3.78%** on `Intern-Decision-4B` (`63.52%` $\rightarrow$ **`67.30%`**), **+1.70%** on zero-shot `d3-nano` (`65.03%` $\rightarrow$ **`66.73%`**), **+1.32%** on LoRA `Qwen3.5-0.8B` (`65.41%` $\rightarrow$ **`66.73%`**), and **+1.14%** on `ViSO v5` (`34.40%` $\rightarrow$ **`35.54%`**).
4. **2-Scale Controlled Comparison: Causal VLMs vs. Bidirectional Decision 3.0 (`d3-lite` & `d3-nano`)**: Across two parameter scales (`0.85B` and `2.21B`), converting standard causal autoregressive VLMs (`Qwen3.5-0.8B`, `Qwen3.5-2B`) into Decision 3.0 models (`vllm-sr/d3-lite`, `vllm-sr/d3-nano`) with bidirectional `noncausal_full_attention` and dedicated `readout.safetensors` consistently improves accuracy, calibration, NLL, and single-pass latency:
   - **0.85B Scale (`Qwen3.5-0.8B` vs. `vllm-sr/d3-lite`)**: Zero-Shot jumps by **+5.3%** (`56.0%` $\rightarrow$ **61.3%**). After Vision+Language LoRA (`--mode img`) + Decision Head tuning, `d3-lite` reaches **72.4% `All` / 72.5% `Core` / 72.6% `Image-Unseen`**, lowering NLL (`0.736` $\rightarrow$ **0.720** on `All`), cutting calibration error (**0.021 ECE**), and running **1.78$\times$ faster** (**0.0138 s/unit** vs. `0.0245 s/unit`).
   - **2.21B Scale (`Qwen3.5-2B` vs. `vllm-sr/d3-nano`)**: Zero-Shot jumps by **+4.0%** (`64.4%` $\rightarrow$ **68.4%** `All` / **69.2%** with `vectype`). After Vision+Language LoRA + Decision Head tuning, `d3-nano` sets the single-model SOTA at **76.1% `All` / 76.3% `Core` / 76.8% `Image-Unseen`** (**0.606 NLL** on `Image-Unseen`) while running **2.1$\times$ faster** (**0.0210 s/unit** vs. `0.0444 s/unit`).
5. **Multi-Family Calibrated System-One VLM Ensembles (`78.0% All / 78.2% Core / 78.7% Image-Unseen`)**: Pre-calibrating constituent models on the full canonical `3,455`-item `calib` split and combining System-One VLM runs in [`scripts/s1_vlm_ensemble.py`](file:///home/khaitran/project/IE307-Final/ViVQA_S1/scripts/s1_vlm_ensemble.py) pushes accuracy to **78.44% on `Dev`**, **78.02% on `Test All`**, **78.22% on `Core`**, and **78.73% on `Image-Unseen`** (`noul = 83.69%`, `score = 73.72%`, `choice = 74.55%`, **0.5881 NLL**, **0.0182 ECE** on `9-Model`, and **0.0147 Dev ECE** on `14-Model`).
6. **Reference Teacher & "Student Beats Teacher"**: The Decision API Teacher achieves **68.1% accuracy** with an API latency of ~800 ms and higher calibration error (ECE 0.105). Every fine-tuned on-device System-One VLM (`Qwen3.5-0.8B` at 72.3%, `d3-lite` at 72.4%, `MiniCPM-V-4.7-1B` at 73.5%, `Qwen3.5-2B` at 75.9%, `d3-nano` at **76.1%**, 9-VLM Ensemble at **78.0% / 78.7% Image-Unseen**) as well as our `10-ViSO` and `3-Tier` Cascades (**75.7%** at 10% coverage) surpass the cloud teacher by up to **+9.9%**.




