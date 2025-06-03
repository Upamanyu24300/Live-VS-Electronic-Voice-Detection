# 🎙️ Live vs Electronic Voice Detection

This project focuses on distinguishing **live (biological)** voices from **electronic (synthesized)** voices using machine learning and deep learning techniques. It implements a **Self-Attentional ResNet** and a **LightGBM** classifier trained on GFCC features extracted from the POCO (Fake-or-Real) dataset.

---

## 🚀 Overview

The goal is to classify audio into two categories:
- **Live Voice**: Human-spoken voice
- **Electronic Voice**: Machine-generated or replayed audio

### 🔍 Key Features
- Gammatone Frequency Cepstral Coefficients (GFCC) feature extraction
- Low-Frequency Average Energy (LFAE) based frame selection
- Self-Attentional ResNet model
- LightGBM classifier for feature-level boosting
- Visualizations of preprocessing, GFCCs, and model performance
- Evaluation using Accuracy, EER, FAR, and FRR

---

## 🛠️ Tech Stack & Tools

| Tool/Library | Description |
|--------------|-------------|
| <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" /> | Programming Language |
| <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" /> | Deep Learning Framework |
| <img src="https://img.shields.io/badge/LightGBM-Boosting-green?logo=lightgbm" /> | Gradient Boosted Trees |
| <img src="https://img.shields.io/badge/Librosa-Audio-blue?logo=librosa" /> | Audio Feature Extraction |
| <img src="https://img.shields.io/badge/NumPy-Scientific-blue?logo=numpy" /> | Numerical Computation |
| <img src="https://img.shields.io/badge/Matplotlib-Visualization-red?logo=matplotlib" /> | Plotting Library |
| <img src="https://img.shields.io/badge/SciKit Learn-ML-yellow?logo=scikit-learn" /> | Classical ML Tools |
| <img src="https://img.shields.io/badge/Gammatone--filters-Audio-black" /> | Auditory Filtering |

---

## 📁 Dataset

- **POCO Dataset (Fake-or-Real Deepfake Audio)**  
  Structured into:
  - `training/real`, `training/fake`
  - `testing/real`, `testing/fake`
  - `validation/real`, `validation/fake`

---

## 🧪 Methodology

### 1. **Preprocessing**
- **STFT** for time-frequency analysis
- **LFAE** for frame selection (top 10 high-energy frames)
- **GFCC** extraction using Gammatone filterbanks + DCT

### 2. **Models**
#### 🎯 Self-Attentional ResNet
- PreAct ResBlocks
- Self-Attention Pooling
- Binary classification via sigmoid output

#### 🌲 LightGBM
- Input from ResNet penultimate layer
- GridSearchCV for tuning
- Trained on extracted GFCC embeddings

---

## 📊 Evaluation Metrics

| Metric | ResNet | LightGBM |
|--------|--------|----------|
| **Test Accuracy** | 78.40% | **79.32%** |
| **Equal Error Rate (EER)** | 22.61% | **21.51%** |
| **FAR / FRR** | 22.61% / 22.06% | **21.51% / 21.51%** |

---

## 📈 Visual Outputs

- Waveform plots of raw audio
- STFT Spectrograms
- GFCC heatmaps
- Attention weight distribution
- Confusion matrix and ROC curve

---

## 📦 Installation

```bash
pip install librosa soundfile gammatone lightgbm tensorflow matplotlib scikit-learn
