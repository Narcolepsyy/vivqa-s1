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
│   ├── viso_probe.py                     # Controlled learning probes (400 steps)
│   ├── viso_calibrate.py                 # Post-hoc temperature calibration
│   ├── viso_predict.py                   # Model inference & prediction dumper
│   ├── viso_report.py                    # Multi-view evaluation reporter
│   ├── viso_pack_features.py             # High-speed float16 feature packer
│   └── kaggle_prepare.py                 # Kaggle dataset packaging utility
├── kaggle/                   # Automated Kaggle GPU experiment pipelines
│   ├── experiments.json                  # Experiment matrix (A0 - A6)
│   ├── experiments_siglip2.json          # SigLIP-2 experiment manifests
│   └── viso_matrix_kernel.py             # Multi-GPU Kaggle runner
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

| Split | Total Items | `choice` | `score` | `noul` | Images |
|---|---|---|---|---|---|
| **Train** | 31,647 | 15,910 | 11,048 | 4,689 | 24,964 |
| **Calib** | 3,455 | 1,745 | 1,202 | 508 | 2,755 |
| **Dev** | 4,392 | 2,210 | 1,530 | 652 | 3,466 |
| **Test** | 4,409 | 2,176 | 1,539 | 694 | 3,438 |
| **Total** | **43,903** | **22,041** | **15,319** | **6,543** | **34,623** |

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

To train a ViSO model with SigLIP-2 visual features and BaMI text representations:

```bash
python train_viso.py \
    --config-file configs/viso_siglip2_bami.yaml \
    --opts MODEL.DEVICE cuda \
           MODEL.NAME viso_siglip2_run \
           LOSS.NAME rlcd \
           TRAINING.EPOCHS_SUP 25
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

---

## 🔬 Model Architectures & ViSO Ablations

| Variant | Vision Backbone | Text Backbone | Decision Head | Loss / Training | Test Acc | Core Acc | ECE | Latency |
|---|---|---|---|---|---|---|---|---|
| **A0 (Flat Baseline)** | SigLIP-2 (768d) | BaMI / WordEmb | Flat 3,500-way Softmax | Cross-Entropy | 43.2% | 43.2% | 0.106 | ~3 ms |
| **A1** | SigLIP-2 (768d) | BaMI | Typed Heads (`noul`/`score`/`choice`) | Hard-CE | 43.8% | 43.9% | 0.052 | ~3 ms |
| **A2 (ViSO Base)** | SigLIP-2 (768d) | WordEmb | Typed Heads + SetAttention | Soft-CE | 44.6% | 44.7% | 0.020 | ~3 ms |
| **A2bami (ViSO + BaMI)**| SigLIP-2 (768d) | BaMI BERT (768d) | Typed Heads + SetAttention | Soft-CE + Standardize | **45.6%** | **45.7%** | **0.023** | ~3.5 ms |
| **A3 (ViSO + RLCD)** | SigLIP-2 (768d) | BaMI BERT | Typed Heads + SetAttention | Dual-stage RLCD | 45.1% | 45.3% | 0.027 | ~3.5 ms |
| **Cross-Attn Head** | SigLIP-2 (768d) | BaMI BERT | Candidate-to-Patch Cross-Attention | Soft-CE | **46.2%** | **46.4%** | **0.022** | ~4 ms |
| **ViSO Cascade (50% Cov)** | SigLIP-2 (768d) | BaMI + Qwen3.5 | Calibrated Escalation Head | Selective Routing | **61.3%** | **61.3%** | 0.030 | ~14 ms |
| **ViSO Cascade (20% Cov)** | SigLIP-2 (768d) | BaMI + Qwen3.5 | Calibrated Escalation Head | Selective Routing | **69.0%** | **69.2%** | 0.029 | ~21 ms |
| **ViSO Cascade (10% Cov)** | SigLIP-2 (768d) | BaMI + Qwen3.5 | Calibrated Escalation Head | Selective Routing | **71.2%** | **71.4%** | **0.022** | ~23 ms |

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
| **Qwen3.5-2B (VLM)** | 2.27 B | LoRA + Head | **75.7%** | **75.9%** | **76.4%** | **81.1%** | **70.0%** | **72.9%** | 0.025 | 0.044s |
| **Qwen3.5-0.8B (VLM)** | 0.87 B | LoRA + Head | **72.1%** | **72.3%** | **72.8%** | **78.4%** | **65.5%** | **68.8%** | 0.038 | 0.025s |
| **ViSO Cascade (10% Cov)** | ~103 M + 0.8B | System-1 $\rightarrow$ System-2 | **71.2%** | **71.4%** | **71.8%** | 77.8% | 64.9% | 68.2% | **0.022** | ~0.023s |
| *Qwen3-VL-8B (VLM)* | 8.77 B | Zero-shot Ref | 69.5% | 69.8% | 70.3% | 82.5% | 69.3% | 59.4% | 0.031 | 0.122s |
| **ViSO Cascade (20% Cov)** | ~103 M + 0.8B | System-1 $\rightarrow$ System-2 | **69.0%** | **69.2%** | **69.8%** | 76.1% | 63.4% | 66.0% | 0.029 | ~0.021s |
| *Intern-Decision-4B (VLM)* | 4.54 B | Zero-shot Ref | 68.1% | 68.5% | 69.1% | 81.8% | 63.3% | 58.5% | 0.044 | 0.082s |
| *Intern-Decision-2B (VLM)* | 2.21 B | Zero-shot Ref | 64.9% | 65.3% | 65.8% | 77.6% | 57.2% | 56.9% | 0.025 | 0.047s |
| **OneJev-0.8B (VLM)** | 1.11 B | Head-only | 61.2% | 61.6% | 61.9% | 74.6% | 61.3% | 50.7% | 0.035 | 0.029s |
| **ViSO Cascade (50% Cov)** | ~103 M + 0.8B | System-1 $\rightarrow$ System-2 | **61.3%** | **61.3%** | **61.5%** | 70.2% | 55.4% | 57.8% | 0.030 | ~0.014s |
| **jpt-0.8B (VLM)** | 0.85 B | Head-only | 60.5% | 60.9% | 60.9% | 74.1% | 59.2% | 50.1% | 0.037 | 0.029s |
| **Intern-Decision-0.8B (VLM)** | 0.85 B | Head-only | 60.1% | 60.4% | 61.3% | 73.5% | 57.2% | 50.3% | 0.031 | 0.029s |
| **Qwen3.5-0.8B (VLM)** | 0.87 B | Head-only | 59.7% | 60.1% | 60.5% | 72.5% | 56.8% | 50.4% | 0.030 | 0.029s |
| **OneJev-0.8B (VLM)** | 1.11 B | Zero-shot | 60.2% | 60.5% | 60.8% | 74.4% | 54.7% | 50.4% | 0.037 | 0.029s |
| **jpt-0.8B (VLM)** | 0.85 B | Zero-shot | 58.5% | 58.9% | 58.8% | 72.1% | 52.1% | 49.4% | 0.038 | 0.029s |
| **Intern-Decision-0.8B (VLM)** | 0.85 B | Zero-shot | 57.9% | 58.3% | 58.9% | 72.5% | 49.9% | 48.5% | 0.030 | 0.029s |
| **Vintern-1B-v3.5 (VLM)** | 0.94 B | Head-only | 57.1% | 57.3% | 57.9% | 69.7% | 51.3% | 48.6% | 0.051 | 0.070s |
| **Qwen3.5-0.8B (VLM)** | 0.87 B | Zero-shot | 56.0% | 56.4% | 57.1% | 70.5% | 41.9% | 48.0% | 0.030 | 0.029s |
| **Vintern-1B-v3.5 (VLM)** | 0.94 B | Zero-shot | 53.3% | 53.5% | 53.8% | 69.1% | 48.0% | 42.3% | 0.046 | 0.070s |
| **ViSO (Cross-Attn)** | ~106 M | Token Cross-Attn | **46.2%** | **46.4%** | **46.6%** | **54.8%** | **37.1%** | **41.6%** | **0.022** | ~0.004s |
| **ViSO (A2bami: BaMI+SigLIP2)**| ~103 M | Typed System-1 | **45.6%** | **45.7%** | **45.9%** | **54.5%** | **36.5%** | **40.9%** | **0.023** | ~0.003s |
| **ViSO (A2: SigLIP2 Base)** | ~103 M | Typed System-1 | **44.6%** | **44.7%** | **44.9%** | 52.9% | 33.5% | 40.8% | **0.020** | ~0.003s |
| **FlatS1 (A0 Baseline)** | ~100 M | Flat Softmax | 43.2% | 43.2% | 43.4% | 52.0% | 30.3% | 39.4% | 0.106 | ~0.003s |
| *ViSO Zero-Vision (A5)* | ~103 M | Text-only Control | 40.8% | 40.8% | 41.0% | 51.1% | 26.8% | 36.2% | 0.033 | ~0.002s |
| *B-MLP (Simple Floor)* | ~15 M | Question Bag x Image | 35.5% | 35.5% | 35.6% | 51.0% | 28.0% | 25.0% | - | <0.001s |

### 📌 Benchmark Insights
1. **ViSO Efficiency & Latency Advantage**: ViSO alone operates at an ultra-low inference latency of **~3 ms** per question on CPU/Edge devices with only **~103M parameters**—more than $10\times$ faster and $15\times$ smaller than sub-billion parameter VLMs, while comfortably outperforming the standard Flat classification baseline (+2.4% overall, +6.2% on counting).
2. **Selective Escalation (System-One $\rightarrow$ System-Two)**: By capitalizing on ViSO's sharp calibration (ECE $\approx 0.020$), the escalation head identifies confident answers locally and defers ambiguous cases to System-Two (Qwen3.5). At **10% ViSO coverage**, the cascade reaches **71.2% accuracy** (99% of full VLM performance) with significant latency reduction on confident queries. At **50% coverage**, accuracy reaches **61.3%** while requiring only **0.64x** of the VLM's compute budget.
3. **Cross-Attention & Architectural Innovations**: Introducing token-level Candidate-to-Visual Cross-Attention and Vietnamese BaMI BERT features pushes pure System-One accuracy to **46.2%**, effectively mitigating visual feature pooling bottlenecks.
4. **LoRA Fine-tuning**: For full generative models, parameter-efficient LoRA tuning on Qwen3.5-0.8B yields a +16.1% leap over zero-shot to **72.1%**, while Qwen3.5-2B achieves the ceiling at **75.7%**.


