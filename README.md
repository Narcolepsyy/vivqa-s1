# ViVQA-S1: System-One Typed Decision Architecture & Benchmarking for Vietnamese Visual Question Answering

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Narcolepsyy%2Fvivqa--s1-black.svg?logo=github)](https://github.com/Narcolepsyy/vivqa-s1)

**ViVQA-S1** is an efficient **System-One ($O(1)$ single forward pass)** visual question answering framework and benchmark tailored for Vietnamese. Unlike standard Vision-Language Models (VLMs) that perform slow autoregressive decoding ($O(L)$ steps), ViVQA-S1 parametrizes VQA as a **typed decision task**—achieving high accuracy, low latency, and formal confidence calibration with selective fallback.

Developed as part of the **IE307 Capstone Project**, University of Information Technology (UIT - VNUHCM).

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

## 🔬 Model Architectures & Ablations

| Variant | Vision Backbone | Text Backbone | Decision Head | Loss |
|---|---|---|---|---|
| **A0 (Baseline)** | SigLIP-2 (768d) | BaMI / WordEmb | Flat Softmax | Cross-Entropy |
| **A1** | SigLIP-2 (768d) | BaMI | Typed Heads (`noul`/`score`/`choice`) | Hard-CE |
| **A2** | SigLIP-2 (768d) | BaMI | Typed Heads + SetAttention | Soft-CE |
| **A3 (ViSO)** | SigLIP-2 (768d) | BaMI | Typed Heads + SetAttention | **RLCD** |
| **A4 (ViSO+Calib)**| SigLIP-2 (768d) | BaMI | Typed Heads + Temperature Scaling | RLCD + Calib |
| **Cross-Attn** | SigLIP-2 (768d) | BaMI | Candidate-to-Patch Cross-Attention | RLCD |

---

## 📬 Contact & Citation

**Author**: Khai Tran ([@Narcolepsyy](https://github.com/Narcolepsyy))  
**Email**: [23520680@gm.uit.edu.vn](mailto:23520680@gm.uit.edu.vn)  
**Affiliation**: Department of Information Systems, University of Information Technology (UIT - VNUHCM)  
**Advisor**: Phan Nguyet Duy ([duypn@uit.edu.vn](mailto:duypn@uit.edu.vn))  

```bibtex
@misc{tran2026vivqas1,
  title={ViVQA-S1: System-One Typed Decision Architecture & Benchmarking for Vietnamese Visual Question Answering},
  author={Khai Tran},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Narcolepsyy/vivqa-s1}}
}
```
