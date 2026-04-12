# 🌐 Multimodal Hate Speech Detection in Low-Resource Dravidian Languages

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange?logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Meta--Learner-brightgreen)](https://lightgbm.readthedocs.io/)
[![Languages](https://img.shields.io/badge/Languages-Tamil%20%7C%20Telugu%20%7C%20Malayalam-purple)](#supported-languages)
[![Multimodal](https://img.shields.io/badge/Modalities-Text%20%2B%20Audio-red)](#system-architecture)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

> **Automatically detect hate speech in Tamil, Telugu, and Malayalam social media content by combining the power of text understanding and audio emotion analysis.**

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Supported Languages](#supported-languages)
4. [System Architecture](#system-architecture)
   - [Phase 1 – Text Stream](#phase-1--text-stream-nlp)
   - [Phase 2 – Audio Stream](#phase-2--audio-stream-speech-signal-processing)
   - [Phase 3 – Late Fusion](#phase-3--late-fusion-meta-learner)
5. [Dataset Information](#dataset-information)
6. [Model Details](#model-details)
7. [Installation](#installation)
8. [Usage Guide](#usage-guide)
9. [Results & Performance](#results--performance)
10. [File Structure](#file-structure)
11. [Dependencies](#dependencies)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)
14. [License](#license)
15. [Contact & References](#contact--references)

---

## Executive Summary

This research project tackles the automated detection of **hate speech in Dravidian languages** — Tamil, Telugu, and Malayalam — which are widely spoken across South India but are significantly under-represented in mainstream AI safety research.

Social media platforms host millions of posts in these languages every day, many containing hate speech that goes undetected because existing tools are built primarily for English. This project addresses that gap using a **Multimodal Late Fusion Architecture** that combines:

- 🔤 **Text Analysis** — Semantic understanding of transcripts using state-of-the-art multilingual Transformer models.
- 🔊 **Audio Analysis** — Paralinguistic cue extraction (tone, pitch, emotion) from raw audio waveforms using Speech Foundation Models.
- 🔗 **Intelligent Fusion** — A LightGBM meta-learner that learns to weigh and combine the confidence of all models dynamically.

The result is a robust hate speech classifier that handles **code-mixed language** (e.g., Tamil written in English characters — "Tanglish"), **noisy social media audio**, and the unique linguistic structures of Dravidian languages.

---

## Problem Statement

Hate speech detection in Indian regional languages faces several unique challenges that make it harder than English-based detection:

| Challenge | Description |
|-----------|-------------|
| **Code-Mixing** | Users frequently mix English with native scripts (e.g., Tamil words written in English characters). Traditional NLP models struggle to handle this linguistic mixing. |
| **Tonal Context** | Text transcripts alone miss sarcasm, aggression, or emotion present in the speaker's voice — critical cues for distinguishing "offensive" from "hate" speech. |
| **Data Scarcity** | Unlike English, high-quality, human-labeled datasets for Dravidian languages are extremely limited ("low-resource" problem). |
| **Script Diversity** | Tamil, Telugu, and Malayalam each have entirely distinct scripts, grammar, and phonetics, requiring language-specific fine-tuning. |

A unimodal (text-only or audio-only) system is insufficient. This project's **multimodal approach** directly addresses all four challenges.

---

## Supported Languages

The project currently supports the following Dravidian languages:

| Language | Script | Language Code | Notebook |
|----------|--------|---------------|----------|
| **Tamil** | தமிழ் | `ta` | `tamil-hsd.ipynb` |
| **Telugu** | తెలుగు | `te` | `telugu-hsd.ipynb` |
| **Malayalam** | മലയാളം | `ml` | `malyalam-hsd.ipynb` |

> **Note:** The Dravidian language family also includes Kannada and Tulu, among others. Future extensions of this work may incorporate those languages.

---

## System Architecture

The system implements a **Stacking Ensemble (Meta-Learning)** approach. Multiple specialist models are trained independently and their outputs are fused by a final meta-classifier.

```
                         ┌─────────────────────────────────────┐
                         │         INPUT: Social Media Post      │
                         │      (Text Transcript + Audio File)   │
                         └──────────────┬──────────────────────┘
                                        │
               ┌────────────────────────┴────────────────────────┐
               │                                                  │
               ▼                                                  ▼
   ┌───────────────────────┐                        ┌───────────────────────┐
   │   TEXT STREAM (NLP)   │                        │  AUDIO STREAM (Speech)│
   │                       │                        │                       │
   │  ┌─────────────────┐  │                        │  ┌─────────────────┐  │
   │  │     mBERT       │  │                        │  │   Wav2Vec2      │  │
   │  │  (Multilingual) │  │                        │  │  (XLSR-53)      │  │
   │  └────────┬────────┘  │                        │  └────────┬────────┘  │
   │           │           │                        │           │           │
   │  ┌─────────────────┐  │  ← Augmentation:  →   │  ┌─────────────────┐  │
   │  │  XLM-RoBERTa   │  │     Gaussian Noise     │  │     WavLM       │  │
   │  │  (Cross-lingual)│  │     Time Stretch       │  │  (Non-semantic) │  │
   │  └────────┬────────┘  │     Pitch Shift        │  └────────┬────────┘  │
   │           │           │                        │           │           │
   │  ┌─────────────────┐  │                        │  ┌─────────────────┐  │
   │  │  IndicBERT v2  │  │                        │  │      MMS        │  │
   │  │ (Indian-native) │  │                        │  │  (1000+ langs)  │  │
   │  └────────┬────────┘  │                        │  └────────┬────────┘  │
   └───────────┼───────────┘                        └───────────┼───────────┘
               │ Probability logits                             │ Probability logits
               │ (3 scores)                                     │ (3 scores)
               └────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │  LATE FUSION           │
                       │  Meta-Learner          │
                       │  (LightGBM / GBDT)     │
                       │                        │
                       │  Input: 6 logit vectors│
                       │  Output: Final label   │
                       └────────────┬───────────┘
                                    │
                                    ▼
                     ┌──────────────────────────┐
                     │  PREDICTION              │
                     │  0 = Non-Hate Speech     │
                     │  1 = Hate Speech         │
                     └──────────────────────────┘
```

### Phase 1 – Text Stream (NLP)

The system processes text transcripts through **three multilingual Transformer models** in parallel, each capturing different linguistic nuances:

| Model | HuggingFace ID | Strength |
|-------|---------------|----------|
| **mBERT** | `google-bert/bert-base-multilingual-cased` | Strong baseline, 104-language coverage |
| **XLM-RoBERTa** | `FacebookAI/xlm-roberta-base` | Optimized for cross-lingual and code-mixed data |
| **IndicBERT v2** | `ai4bharat/IndicBERTv2-MLM-only` | Pre-trained natively on Indian languages; superior Dravidian syntax grasp |

**Data split:** 80% Training / 10% Validation / 10% Test (stratified)

**Label mapping:**

| Original Label | Mapped Label | Meaning |
|----------------|-------------|---------|
| `N` | 0 | Non-Hate |
| `C`, `G`, `P`, `R`, `O` | 1 | Hate Speech (various sub-categories) |

### Phase 2 – Audio Stream (Speech Signal Processing)

Raw audio waveforms are processed to extract **paralinguistic features** — the tonal, emotional, and prosodic cues missed by text alone.

**Data augmentation** is applied before training to make the model robust to real-world noise:
- 🔉 **Gaussian Noise** — Simulates background noise common in social media recordings
- ⏱️ **Time Stretch** — Handles variations in speech speed
- 🎵 **Pitch Shift** — Handles variations in speaker pitch

Three Speech Foundation Models (SFMs) are fine-tuned:

| Model | Description |
|-------|-------------|
| **Wav2Vec2 (XLSR-53)** | Massive multilingual speech model fine-tuned with language-specific checkpoints for Tamil/Telugu |
| **WavLM** | Captures non-semantic speech details (speaker identity, background noise) for richer context detection |
| **MMS (Massively Multilingual Speech)** | Meta's model supporting 1,000+ languages including Dravidian dialects |

### Phase 3 – Late Fusion (Meta-Learner)

The probability scores (logits) from all **6 models** (3 Text + 3 Audio) are concatenated into a single feature vector. A **LightGBM** (Gradient Boosting Decision Tree) classifier acts as the meta-learner:

- It learns to **dynamically weigh** each model's confidence based on context
- Prioritizes **audio models** when text is ambiguous or highly code-mixed
- Prioritizes **text models** when audio quality is poor or noisy
- Trained on the same 10% validation split used by the base models

---

## Dataset Information

The project uses the **"Hate Speech Detection in Dravidian Languages"** dataset, available on Kaggle.

**Dataset Source:**
```
/kaggle/input/hate-speech-detection-in-dravidian-languages/
└── Hate Speech Detection in Dravidian languages/
    ├── train/
    │   ├── tamil/
    │   │   ├── text/   → TA-AT-train.csv
    │   │   └── audio/  → .wav files
    │   ├── telugu/
    │   │   ├── text/   → TE-AT-train.csv
    │   │   └── audio/  → .wav files
    │   └── malayalam/
    │       ├── text/   → MA-AT-train.csv
    │       └── audio/  → .wav files
    └── test/
        └── (same structure)
```

**CSV Format (Text Files):**

| Column | Description |
|--------|-------------|
| `Transcript` | The text content of the social media post |
| `Class Label Short` | Hate speech label: `N` (Non-Hate), `C`, `G`, `P`, `R`, or `O` |

**Dataset Characteristics:**
- Covers **social media** content (YouTube comments, Twitter posts)
- Contains naturally occurring **code-mixed** text (e.g., Tanglish — Tamil + English)
- Binary classification: **Hate vs. Non-Hate**
- Stratified 80/10/10 train/validation/test split

---

## Model Details

### Text Models

```python
MODELS_TO_TRAIN = {
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
    "xlm-r":     "FacebookAI/xlm-roberta-base",
    "mbert":     "google-bert/bert-base-multilingual-cased"
}
```

All text models use:
- `AutoTokenizer` + `AutoModelForSequenceClassification` from Hugging Face
- `EarlyStoppingCallback` to prevent overfitting
- Class-weight balancing for imbalanced datasets

### Audio Models

```python
AUDIO_MODELS = [
    "facebook/wav2vec2-large-xlsr-53",   # Wav2Vec2 XLSR-53
    "microsoft/wavlm-base-plus",          # WavLM
    "facebook/mms-300m"                   # MMS
]
```

All audio models use:
- `AutoFeatureExtractor` + `AutoModelForAudioClassification`
- `audiomentations` for on-the-fly data augmentation
- Librosa for audio loading and preprocessing

### Fusion (Meta-Learner)

```python
from lightgbm import LGBMClassifier

meta_learner = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    class_weight='balanced'
)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended; a single T4 or better suffices)
- [Kaggle account](https://www.kaggle.com/) with dataset access

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Rameshsain070/Multimodal-Hate-Speech-Detection-in-Dravidian-Languages-Using-Text-and-Audio.git
cd Multimodal-Hate-Speech-Detection-in-Dravidian-Languages-Using-Text-and-Audio
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### Step 3 — Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

> **CPU-only (no GPU):**
> ```bash
> pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
> ```

### Step 4 — Install Remaining Dependencies

```bash
pip install transformers accelerate datasets evaluate \
            librosa soundfile audiomentations \
            scikit-learn lightgbm joblib \
            pandas numpy matplotlib tqdm \
            huggingface_hub
```

### Step 5 — Log in to Hugging Face

Some models (e.g., IndicBERT v2) require a Hugging Face account. Create a free account at [huggingface.co](https://huggingface.co/) and log in:

```bash
huggingface-cli login
```

### Step 6 — Download the Dataset

1. Go to the [Hate Speech Detection in Dravidian Languages](https://www.kaggle.com/datasets/) dataset on Kaggle.
2. Download and extract it to a local path.
3. Update the `BASE_PATH` in the notebook's `Config` class to point to your local copy:

```python
class Config:
    BASE_PATH = "/path/to/your/hate-speech-detection-in-dravidian-languages"
```

---

## Usage Guide

Each language has its own self-contained Jupyter notebook. Open the notebook for the language you want to work with and run the cells in order.

### Running on Kaggle (Recommended)

1. Upload the notebooks directly to Kaggle.
2. Attach the **"Hate Speech Detection in Dravidian Languages"** dataset as a Kaggle input.
3. Enable **GPU accelerator** (T4 x2 or P100).
4. Run all cells sequentially.

### Running Locally

```bash
jupyter notebook tamil-hsd.ipynb
# or
jupyter notebook telugu-hsd.ipynb
# or
jupyter notebook malyalam-hsd.ipynb
```

### Notebook Cell Overview

| Cell | Purpose |
|------|---------|
| **Cell 1** | Environment setup — installs/updates PyTorch and dependencies |
| **Cell 2** | Configuration — sets paths, label maps, model names, and splits data |
| **Cell 3** | Text model training — fine-tunes mBERT, XLM-R, IndicBERT with early stopping |
| **Cell 4** | Audio setup — installs audio-specific libraries (`audiomentations`, etc.) |
| **Cell 5** | Audio model training — fine-tunes Wav2Vec2, WavLM, MMS with augmentation |
| **Cell 6** | Meta-learner — builds feature vectors from all 6 models, trains LightGBM |
| **Final** | Evaluation — generates classification report, confusion matrix, ROC curve |

### Making a Prediction on New Data

After training, you can load the saved models and run inference:

```python
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load trained meta-learner
meta_learner = joblib.load("meta_learner.pkl")

# Example: Get text logits from one model
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("./xlm-r-finetuned")
model.eval()

text = "உன்னை பார்க்கவே வெறுப்பாக இருக்கிறது"  # Tamil example
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    logits = model(**inputs).logits.softmax(dim=-1).numpy()

# Combine with other model logits and pass to meta-learner
# prediction = meta_learner.predict([combined_logits])
# print("Hate Speech" if prediction[0] == 1 else "Non-Hate Speech")
```

---

## Results & Performance

The multimodal fusion approach consistently outperforms unimodal (text-only or audio-only) baselines.

### Key Metrics

| Approach | Accuracy | F1-Score (Macro) |
|----------|----------|-----------------|
| Text-only (best single model) | ~78–82% | ~0.74–0.79 |
| Audio-only (best single model) | ~71–76% | ~0.67–0.72 |
| **Multimodal Fusion (LightGBM)** | **~84–88%** | **~0.81–0.86** |

> **Note:** Exact numbers vary by language and dataset split. The figures above represent the typical improvement range observed across Tamil, Telugu, and Malayalam experiments.

### Why Multimodal Wins

- **Reduces false positives**: Audio tone corrects text-based misclassifications of sarcasm.
- **Handles code-mixing**: XLM-R and IndicBERT together cover romanized and native-script text.
- **Noise robustness**: Data augmentation + WavLM ensure the system works on real social media audio.

---

## File Structure

```
Multimodal-Hate-Speech-Detection-in-Dravidian-Languages-Using-Text-and-Audio/
│
├── tamil-hsd.ipynb          # Full pipeline for Tamil hate speech detection
├── telugu-hsd.ipynb         # Full pipeline for Telugu hate speech detection
├── malyalam-hsd.ipynb       # Full pipeline for Malayalam hate speech detection
└── README.md                # Project documentation (this file)
```

Each notebook is self-contained and follows the same structure:

```
notebook
├── Cell 1  – Environment setup
├── Cell 2  – Configuration & data loading
├── Cell 3  – Text model training (mBERT, XLM-R, IndicBERT)
├── Cell 4  – Audio library setup
├── Cell 5  – Audio model training (Wav2Vec2, WavLM, MMS)
├── Cell 6  – Meta-learner training (LightGBM fusion)
└── Cell 7+ – Evaluation & visualization
```

**Generated artifacts (saved by notebooks):**

| File | Description |
|------|-------------|
| `tamil_test_split.csv` | Held-out test set for Tamil |
| `telugu_test_split.csv` | Held-out test set for Telugu |
| `malayalam_test_split.csv` | Held-out test set for Malayalam |
| `meta_learner.pkl` | Trained LightGBM meta-learner |
| `./xlm-r-finetuned/` | Fine-tuned XLM-R checkpoint |
| `./indicbert-finetuned/` | Fine-tuned IndicBERT checkpoint |
| `./mbert-finetuned/` | Fine-tuned mBERT checkpoint |
| `./wav2vec2-finetuned/` | Fine-tuned Wav2Vec2 checkpoint |
| `./wavlm-finetuned/` | Fine-tuned WavLM checkpoint |
| `./mms-finetuned/` | Fine-tuned MMS checkpoint |

---

## Dependencies

### Core Frameworks

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.6.0 | Deep learning backend |
| `torchvision` | 0.21.0 | (Required by PyTorch) |
| `torchaudio` | 2.6.0 | Audio tensor operations |
| `transformers` | Latest | Pre-trained Transformer models |
| `accelerate` | Latest | Distributed training support |

### NLP & Data

| Library | Version | Purpose |
|---------|---------|---------|
| `datasets` | Latest | Hugging Face Dataset API |
| `evaluate` | Latest | Metrics (F1, accuracy) |
| `pandas` | Latest | DataFrame manipulation |
| `numpy` | Latest | Numerical operations |
| `scikit-learn` | Latest | Train/test split, class weights |

### Audio Processing

| Library | Version | Purpose |
|---------|---------|---------|
| `librosa` | Latest | Audio loading & feature extraction |
| `soundfile` | Latest | Audio file I/O |
| `audiomentations` | Latest | Data augmentation for audio |
| `numpy-minmax` | Latest | Required by audiomentations |
| `numpy-rms` | Latest | Required by audiomentations |
| `python-stretch` | Latest | Required by audiomentations |

### Machine Learning & Fusion

| Library | Version | Purpose |
|---------|---------|---------|
| `lightgbm` | Latest | LightGBM meta-learner |
| `joblib` | Latest | Model serialization |
| `tqdm` | Latest | Progress bars |
| `matplotlib` | Latest | Confusion matrices & ROC curves |

### Complete Installation

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install transformers accelerate datasets evaluate \
            librosa soundfile \
            audiomentations --no-deps && \
pip install numpy-minmax numpy-rms python-stretch \
            scikit-learn lightgbm joblib \
            pandas numpy matplotlib tqdm \
            huggingface_hub
```

---

## Troubleshooting

###  `CUDA memory`
- Reduce batch size in `TrainingArguments` (e.g., `per_device_train_batch_size=8`).
- Use gradient accumulation: `gradient_accumulation_steps=4`.
- Enable `fp16=True` in `TrainingArguments` for mixed-precision training.

### Module: audiomentations`
```bash
pip install audiomentations --no-deps
pip install numpy-minmax numpy-rms python-stretch
```

### tokenizer for 'ai4bharat/IndicBERTv2-MLM-only'`
- Ensure you are logged in to Hugging Face: `huggingface-cli login`
- The model requires accepting terms on the Hugging Face model page.

###  Dataset path 
- Make `BASE_PATH` in `Config` points to the correct local directory.
- On Kaggle, the path is automatically `/kaggle/input/hate-speech-detection-in-dravidian-languages/`.

###  `DtypeWarning` or empty DataFrame
- Verify CSV column names match `Transcript` and `Class Label Short`.
- Check for extra spaces in column names: use `df.columns.str.strip()`.

### Web app does not predict
- Ensure backend is running and the frontend Backend URL points to `/predict`.
- If request is long-running, enable frontend "real-time job mode" (uses `/predict/jobs` polling API).
- If frontend is on GitHub Pages, configure backend CORS for your Pages origin:
  - `ALLOWED_ORIGINS=https://<username>.github.io`
  - or use `ALLOWED_ORIGIN_REGEX` (defaults to a GitHub Pages origin pattern).
- If predictions time out, increase `PREDICT_TIMEOUT_SECONDS` and/or disable `WARMUP_ALL_MODELS`.
- If error says backend is unreachable, use a public backend host URL (localhost will not work from deployed GitHub Pages).

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository on GitHub.
2. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/add-kannada-support
   ```
3. **Make your changes** and commit with a clear message:
   ```bash
   git commit -m "feat: add Kannada language pipeline"
   ```
4. **Push** to your fork and open a **Pull Request**.

### Ideas for Contribution

- 🌐 Add support for **Kannada** or other Dravidian languages
- 📊 Implement additional evaluation metrics (Precision, Recall, AUC-ROC per class)
- 🧪 Add unit tests for data preprocessing functions
- 🚀 Optimize inference speed for production deployment
- 📱 Build a simple web demo with Gradio or Streamlit
- 📝 Improve dataset documentation and data cards

---

## License

This project is licensed under the **MIT License** — see below for details.

```
MIT License

Copyright (c) 2024 Rameshsain070

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact & References

### Author

**Ramesh Sain**
- GitHub: [@Rameshsain070](https://github.com/Rameshsain070)
- Hugging Face: [@ramesh070](https://huggingface.co/ramesh070)

### Academic References

1. **mBERT** — Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019. [[Paper]](https://arxiv.org/abs/1810.04805)

2. **XLM-RoBERTa** — Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. [[Paper]](https://arxiv.org/abs/1911.02116)

3. **IndicBERT v2** — Doddapaneni, S., et al. (2022). *IndicBERT v2: A new multilingual BERT for Indian languages.* ai4bharat. [[Paper]](https://arxiv.org/abs/2212.05409)

4. **Wav2Vec 2.0** — Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS 2020. [[Paper]](https://arxiv.org/abs/2006.11477)

5. **WavLM** — Chen, S., et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.* IEEE JSTSP. [[Paper]](https://arxiv.org/abs/2110.13900)

6. **MMS** — Pratap, V., et al. (2023). *Scaling Speech Technology to 1,000+ Languages.* Meta AI. [[Paper]](https://arxiv.org/abs/2305.13516)

7. **LightGBM** — Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017. [[Paper]](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)

8. **Dravidian Language NLP** — Chakravarthi, B. R., et al. (2021). *Findings of the Shared Task on Offensive Language Identification in Tamil, Malayalam, and Kannada.* ACL Anthology. [[Paper]](https://aclanthology.org/2021.dravidianlangtech-1.32/)

---

<div align="center">

**⭐ If this project helps your research, please consider starring the repository! ⭐**

Made with ❤️ for the Dravidian NLP community

</div>

---

## 🌍 Web App (Realtime + 3D UI)

This repository now includes a deployable web application:

- `backend/` → FastAPI inference API (uses your trained model files if present under `backend/models/<language>/`)
- `frontend/` → modern 3D web UI (Three.js) for realtime prediction

### Supported production models

Each language exposes all 6 trained base variants and returns both per-model outputs and fused output.

| Language | Text models | Audio models |
|---|---|---|
| Tamil | `indicbert`, `xlm-r`, `mbert` | `wav2vec2`, `wavlm`, `mms` |
| Telugu | `indicbert`, `xlm-r`, `mbert` | `wav2vec2`, `wavlm`, `mms` |
| Malayalam | `indicbert`, `xlm-r`, `mbert` | `wav2vec2`, `wavlm`, `mms` |

### Backend (API)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Optional backend environment variables:

- `ALLOWED_ORIGINS` (comma-separated frontend origins; default: local dev origins)
- `ALLOWED_ORIGIN_REGEX` (regex for dynamic origins; default: `^https://[a-zA-Z0-9-]+\\.github\\.io$`)
- `MAX_AUDIO_BYTES` (upload limit in bytes; default: `10485760`)
- `PREDICT_TIMEOUT_SECONDS` (per-request timeout; default: `120`)
- `JOB_TTL_SECONDS` (async job retention time; default: `3600`)
- `WARMUP_MODELS` (`true|false`, warm default models at startup; default: `true`)
- `WARMUP_ALL_MODELS` (`true|false`, warm all variants at startup; default: `false`)

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction endpoint:

- `POST /predict`
- form-data fields:
  - `language`: `tamil` | `telugu` | `malayalam`
  - `text`: optional input text
  - `audio`: optional audio file
  - `text_models`: optional (`default`, `all`, or comma-separated keys like `xlm-r,mbert`)
  - `audio_models`: optional (`default`, `all`, or comma-separated keys like `wav2vec2,mms`)

At least one modality (`text` or `audio`) is required.

Realtime/polling endpoints:

- `POST /predict/jobs` → creates async prediction job (same form-data as `/predict`)
- `GET /predict/jobs/{job_id}` → returns `queued` | `running` | `completed` | `failed` and result/error
- `GET /models` → returns language-wise model catalog and default selections

Response highlights from `/predict`:

- fused prediction: `prediction`, `label`, `confidence`, `hate_probability`, `fusion_method`
- model usage: `selected_models`, `used_text`, `used_audio`
- per-model outputs: `model_outputs.text` and `model_outputs.audio`

### Using your saved meta models

If you exported LightGBM meta models from notebooks, place them like:

```text
backend/models/tamil/meta_lightgbm_tamil_from_eval_safe.joblib
backend/models/telugu/meta_lightgbm_telugu_from_eval_safe.joblib
backend/models/malayalam/meta_lightgbm_malayalam_from_eval_safe.joblib
```

The backend will auto-load these files and use them for fusion.

### Frontend (3D UI)

Serve frontend locally (any static server):

```bash
cd frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500` and keep backend running at `http://127.0.0.1:8000`.

Frontend backend URL configuration priority:

1. `?apiUrl=<BACKEND_PREDICT_URL>` query parameter
2. `frontend/config.js` via `window.APP_CONFIG.apiUrl`
3. Saved value from the UI (localStorage)
4. Fallback default: `http://127.0.0.1:8000/predict`

### Deploy on your GitHub account

Frontend is configured for GitHub Pages via:

- `.github/workflows/deploy-frontend-pages.yml`

To enable:

1. Push this branch to your repository.
2. In GitHub repo settings, open **Pages** and set source to **GitHub Actions**.
3. Run the workflow (or push to `main` with frontend changes).
4. Your UI will be hosted on `https://<your-username>.github.io/<repo-name>/`

> Note: GitHub Pages is static hosting only. Keep FastAPI backend deployed on a backend host (Render/Railway/Hugging Face Spaces) and set backend URL in `frontend/config.js`, query param, or UI input.
