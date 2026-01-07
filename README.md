# 🧠 Chagas Disease Screening: CNN-BiLSTM with Attention

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Challenge](https://img.shields.io/badge/PhysioNet-Challenge%202025-green.svg)](https://physionetchallenges.org/2025/)
[![Docker Ready](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](#-docker)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](https://www.researchgate.net/publication/398674423_Sequential_Deep_Learning_for_Chagas_Disease_Screening_A_CNN-BiLSTM_Approach_with_an_Attention_Mechanism)

---

### 👥 **Team Chagas UBT**

**🏅 Rank:** `10 / 41` | **📊 Test Score:** `0.238`


_A deep learning approach for automated Chagas disease detection from 12-lead ECGs using CNN, BiLSTM, and Attention
mechanisms._

> 📝 **[Read our paper](https://www.researchgate.net/publication/398674423_Sequential_Deep_Learning_for_Chagas_Disease_Screening_A_CNN-BiLSTM_Approach_with_an_Attention_Mechanism)**

---

## 🏆 Performance

| Dataset      | TPR@5%    | AUROC | AUPRC | Accuracy |
|--------------|-----------|-------|-------|----------|
| REDS-II Test | **0.314** | 0.730 | 0.099 | 0.967    |
| SaMi-Trop 3  | 0.295     | 0.722 | 0.105 | 0.974    |
| ELSA-Brasil  | 0.104     | 0.555 | 0.036 | 0.979    |

---

## 🎗️ Model Architecture

**Pipeline Overview:**

```
12-lead ECG (12×4000 @ 400Hz)
    ↓
CNN Feature Extraction (12→32→64→128)
    ↓
BiLSTM Encoder (2 layers, hidden=128)
    ↓
Additive Attention Mechanism
    ↓
Classification Head (256→128→1)
    ↓
Chagas Risk Probability
```

**Key Features**

- Focal Loss (γ=2.0, α=0.25) for severe class imbalance
- Robust augmentations: time shift, amplitude scaling, lead dropout, mixup
- Weighted sampling emphasizing validated datasets
- Lightweight inference: <0.1 s per sample, ~15 MB model

---

## 🚀 Quick Start

### 🔧 Installation

```bash

git clone https://github.com/sjelodari/Chagas_UBT_Physionet_Challenge.git
cd Chagas_UBT_Physionet_Challenge
pip install -r requirements.txt
```

### 📦 Data Preparation

```bash

# CODE-15%
python prepare_code15_data.py -i code15_input/ -d exams.csv -l labels.csv -o code15_output/

# SaMi-Trop
python prepare_samitrop_data.py -i samitrop_input/ -d exams.csv -o samitrop_output/

# PTB-XL
python prepare_ptbxl_data.py -i ptbxl_input/records500/ -d database.csv -o ptbxl_output/
```

### 🧩 Training & Inference

```bash

# Train model
python train_model.py -d training_data -m model -v

# Run inference
python run_model.py -d holdout_data -m model -o holdout_outputs -v

# Evaluate
python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv
```

---

## 🗂️ Repository Structure

```
Chagas_UBT_Physionet_Challenge/
├── team_code.py          # CNN-BiLSTM-Attention implementation
├── train_model.py        # Training script (official)
├── run_model.py          # Inference script (official)
├── helper_code.py        # Utilities and metrics
├── prepare_*.py          # Dataset preparation scripts
├── Dockerfile
└── README.md
```

---

## 🔬 Methods

**Data Processing**

- Resampled to 400 Hz, 10 s windows (12×4000)
- Per-lead z-score normalization
- Augmentations: time shift (±0.5 s), amplitude scaling (0.9–1.1×), lead dropout, Gaussian noise (σ = 0.01), mixup

**Training**

- Optimizer: Adam (lr = 1e-3), batch = 32, up to 40 epochs
- Loss: Focal Loss with early stopping (patience = 7)
- Weighted sampling: 2× for SaMi-Trop/PTB-XL

---

## 🐳 Docker

```bash
docker build -t chagas_ubt .
docker run -it -v ~/data:/challenge/data chagas_ubt bash
```

---

## 📖 Citation

```bibtex
@inproceedings{inproceedings,
author = {Mamaghani, Saber and Bokun, Adam and Leutheuser, Heike},
year = {2025},
month = {12},
pages = {},
title = {Sequential Deep Learning for Chagas Disease Screening: A CNN-BiLSTM Approach with an Attention Mechanism},
doi = {10.22489/CinC.2025.126}
}
```

---

## 🙏 Acknowledgments

- PhysioNet Challenge 2025 organizers
- NHR@FAU for HPC resources (NVIDIA A100)
- Funded by DFG grant 440719683

---

## 📧 Contact

**Saber Jelodari Mamaghani**  
University of Bayreuth  
📩 [saber.jelodari@uni-bayreuth.de](mailto:saber.jelodari@uni-bayreuth.de)

